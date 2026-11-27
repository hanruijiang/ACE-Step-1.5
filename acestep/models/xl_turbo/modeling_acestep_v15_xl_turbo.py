# Copyright 2025 The ACESTEO Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import math
import time
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange

from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.utils import logging


try:
    from .configuration_acestep_v15 import AceStepConfig
except ImportError:
    from configuration_acestep_v15 import AceStepConfig

from acestep.models.common.dcw_correction import DCWCorrector


logger = logging.get_logger(__name__)

from acestep.models.common.modeling_layers import (
    AceStepPreTrainedModel,
    AudioTokenDetokenizer,
    _repaint_step_injection,
    _repaint_boundary_blend,
    XLAceStepAudioTokenizer as AceStepAudioTokenizer,
    XLAceStepDiTModel as AceStepDiTModel,
    XLAceStepConditionEncoder as AceStepConditionEncoder,
)


class AceStepConditionGenerationModel(AceStepPreTrainedModel):
    """
    Main conditional generation model for AceStep.
    
    End-to-end model for generating audio conditioned on text, lyrics, and timbre.
    Combines encoder (for conditioning), decoder (diffusion model), tokenizer
    (for discrete tokenization), and detokenizer (for reconstruction).
    Generates audio with flow matching samplers.
    """
    def __init__(self, config: AceStepConfig):
        super().__init__(config)
        self.config = config
        # Diffusion model components
        self.decoder = AceStepDiTModel(config)  # Main diffusion transformer
        # Build encoder config: XL models use separate encoder dimensions
        # (encoder_hidden_size=2048 != decoder hidden_size=2560)
        encoder_config = copy.deepcopy(config)
        encoder_config.hidden_size = config.encoder_hidden_size
        encoder_config.intermediate_size = config.encoder_intermediate_size
        encoder_config.num_attention_heads = config.encoder_num_attention_heads
        encoder_config.num_key_value_heads = config.encoder_num_key_value_heads
        self.encoder = AceStepConditionEncoder(encoder_config)  # Condition encoder
        self.tokenizer = AceStepAudioTokenizer(encoder_config)  # Audio tokenizer
        self.detokenizer = AudioTokenDetokenizer(encoder_config)  # Audio detokenizer
        # Null condition embedding for classifier-free guidance
        self.null_condition_emb = nn.Parameter(torch.randn(1, 1, encoder_config.hidden_size))

        # Initialize weights and apply final processing
        self.post_init()

    def tokenize(self, x, silence_latent, attention_mask):
        if x.shape[1] % self.config.pool_window_size != 0:
            pad_len = self.config.pool_window_size - (x.shape[1] % self.config.pool_window_size)
            x = torch.cat([x,  silence_latent[:1,:pad_len].repeat(x.shape[0],1,1)], dim=1)
            attention_mask = F.pad(attention_mask, (0, pad_len), mode='constant', value=0)
        x = rearrange(x, 'n (t_patch p) d -> n t_patch p d', p=self.config.pool_window_size)
        seq_len = x.shape[1]
        chunk = math.ceil(attention_mask.shape[1] / seq_len)
        attention_mask = attention_mask.to(x.dtype)
        attention_mask = F.max_pool1d(attention_mask.unsqueeze(1), kernel_size=chunk, stride=chunk, ceil_mode=True).squeeze(1)
        quantized, indices = self.tokenizer(x)
        return quantized, indices, attention_mask

    def detokenize(self, quantized):
        """
        Detokenize quantized audio tokens back to continuous representations.

        Args:
            quantized: Quantized tokens of shape [N, T//pool_window_size, d]

        Returns:
            Detokenized hidden states of shape [N, T, d]
        """
        hidden_states = self.detokenizer(quantized)
        return hidden_states

    @torch.no_grad()
    def prepare_condition(
        self,
        text_hidden_states: torch.FloatTensor,
        text_attention_mask: torch.Tensor,
        lyric_hidden_states: torch.FloatTensor,
        lyric_attention_mask: torch.Tensor,
        refer_audio_acoustic_hidden_states_packed: torch.FloatTensor,
        refer_audio_order_mask: torch.Tensor,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor,
        silence_latent: torch.FloatTensor,
        src_latents: torch.FloatTensor,
        chunk_masks: torch.Tensor,
        is_covers: torch.Tensor,
        precomputed_lm_hints_25Hz: Optional[torch.FloatTensor] = None,
        audio_codes: torch.FloatTensor = None,
    ):
        
        dtype = hidden_states.dtype
        encoder_hidden_states, encoder_attention_mask = self.encoder(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
        )

        # N x T x d -> N x T//pool_window_size x pool_window_size x d
        # tokenize and detokenize to get LM hints for cover songs (when is_covers=True)
        # Use precomputed hints if provided (e.g., from audio codes), otherwise tokenize hidden_states
        if precomputed_lm_hints_25Hz is not None:
            print("Using precomputed LM hints")
            lm_hints_25Hz = precomputed_lm_hints_25Hz[:, :src_latents.shape[1], :]
        else:
            if audio_codes is not None:
                lm_hints_5Hz = self.tokenize.quantizer.get_output_from_indices(audio_codes)
            else:
                lm_hints_5Hz, indices, llm_mask = self.tokenize(hidden_states, silence_latent, attention_mask)
            lm_hints_25Hz = self.detokenize(lm_hints_5Hz)
            # Crop lm_hints_25Hz to match src_latents length (tokenize may have added padding)
            lm_hints_25Hz = lm_hints_25Hz[:, :src_latents.shape[1], :]
        src_latents = torch.where(is_covers.unsqueeze(-1).unsqueeze(-1) > 0, lm_hints_25Hz, src_latents)
        # Concatenate source latents with chunk masks as context
        context_latents = torch.cat([src_latents, chunk_masks.to(dtype)], dim=-1)
        return encoder_hidden_states, encoder_attention_mask, context_latents

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use generate_audio() for inference.")

    def prepare_noise(self, context_latents: torch.FloatTensor, seed: Union[int, List[int], None] = None):
        """
        Prepare noise tensor for generation with optional seeding.

        Args:
            context_latents: Context latents to determine noise shape
            seed: Can be int, List[int], or None. If None, uses random noise.

        Returns:
            Noise tensor of appropriate shape
        """
        device = context_latents.device
        dtype = context_latents.dtype
        # Handle seed: can be int, List[int], or None
        src_latents_shape = (context_latents.shape[0], context_latents.shape[1], context_latents.shape[-1] // 2)
        if seed is None:
            # No seed provided - use random
            noise = torch.randn(src_latents_shape, device=device, dtype=dtype)
        elif isinstance(seed, list):
            # List of seeds - generate noise for each sample separately
            noise_list = []
            for i, s in enumerate(seed):
                if s is None or s < 0:
                    # Random seed for this sample
                    noise_i = torch.randn(1, src_latents_shape[1], src_latents_shape[2], device=device, dtype=dtype)
                else:
                    # Use specific seed for this sample
                    generator = torch.Generator(device=device).manual_seed(int(s))
                    noise_i = torch.randn(1, src_latents_shape[1], src_latents_shape[2], generator=generator, device=device, dtype=dtype)
                noise_list.append(noise_i)
            noise = torch.cat(noise_list, dim=0)
        else:
            # Single seed for all samples
            generator = torch.Generator(device=device).manual_seed(int(seed))
            noise = torch.randn(src_latents_shape, generator=generator, device=device, dtype=dtype)

        return noise

    def get_x0_from_noise(self, zt, vt, t):
        return zt - vt * t.unsqueeze(-1).unsqueeze(-1)

    def renoise(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        if isinstance(t, torch.Tensor) and t.ndim != x.ndim:
            t = t.unsqueeze(-1).unsqueeze(-1)
        xt = t * noise + (1 - t) * x
        return xt

    def flowedit_generate_audio(self, **kwargs):
        """Flow-edit overlay on the CFG-distilled XL-turbo variant.

        Force ``diffusion_guidance_scale=1.0`` because XL-turbo bakes
        CFG into the velocity head — paired-CFG would amplify a delta
        the model wasn't trained to produce.  Otherwise delegate to
        the shared pipeline.
        """
        from acestep.models.common.flow_edit_pipeline import (
            flowedit_generate_audio as _flowedit_impl,
        )

        gs = kwargs.get("diffusion_guidance_scale", 1.0)
        if gs != 1.0:
            from loguru import logger
            logger.info(
                "[xl-turbo flowedit] xl-turbo is CFG-distilled; forcing "
                "diffusion_guidance_scale=1.0 (was {:.2f}).", gs,
            )
            kwargs["diffusion_guidance_scale"] = 1.0
        return _flowedit_impl(self, **kwargs)

    def generate_audio(
        self,
        text_hidden_states: torch.FloatTensor,
        text_attention_mask: torch.FloatTensor,
        lyric_hidden_states: torch.FloatTensor,
        lyric_attention_mask: torch.FloatTensor,
        refer_audio_acoustic_hidden_states_packed: torch.FloatTensor,
        refer_audio_order_mask: torch.LongTensor,
        src_latents: torch.FloatTensor,
        chunk_masks: torch.FloatTensor,
        is_covers: torch.Tensor,
        silence_latent: Optional[torch.FloatTensor] = None,
        attention_mask: torch.Tensor = None,
        seed: int = None,
        retake_seed: Optional[Union[int, List[int]]] = None,
        retake_variance: float = 0.0,
        fix_nfe: int = 8,
        infer_steps: Optional[int] = None,
        infer_method: str = "ode",
        use_cache: bool = True,
        audio_cover_strength: float = 1.0,
        non_cover_text_hidden_states: Optional[torch.FloatTensor] = None,
        non_cover_text_attention_mask: Optional[torch.FloatTensor] = None,
        precomputed_lm_hints_25Hz: Optional[torch.FloatTensor] = None,
        audio_codes: Optional[torch.FloatTensor] = None,
        shift: float = 3.0,
        timesteps: Optional[torch.Tensor] = None,
        cover_noise_strength: float = 0.0,
        repaint_mask: Optional[torch.Tensor] = None,
        clean_src_latents: Optional[torch.FloatTensor] = None,
        repaint_crossfade_frames: int = 10,
        repaint_injection_ratio: float = 0.5,
        sampler_mode: str = "euler",
        velocity_norm_threshold: float = 0.0,
        velocity_ema_factor: float = 0.0,
        dcw_enabled: bool = True,
        dcw_mode: str = "double",
        dcw_scaler: float = 0.05,
        dcw_high_scaler: float = 0.02,
        dcw_wavelet: str = "haar",
        # Turbo uses distilled guidance.
        diffusion_guidance_scale: float = 1.0,
        use_adg: bool = False,
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        **kwargs,
    ):
        if diffusion_guidance_scale != 1.0 or use_adg or cfg_interval_start != 0.0 or cfg_interval_end != 1.0:
            logger.info(
                "XL-Turbo DiT ignores CFG params (guidance_scale=%.2f, use_adg=%s, "
                "cfg_interval=[%.2f, %.2f]); turbo is CFG-distilled.",
                diffusion_guidance_scale, use_adg, cfg_interval_start, cfg_interval_end,
            )

        # Valid shifts: only discrete values 1, 2, 3 are supported
        VALID_SHIFTS = [1.0, 2.0, 3.0]
        
        # Valid timesteps: all unique timesteps from shift=1,2,3 with fix_nfe=8 (total 20 values)
        VALID_TIMESTEPS = [
            1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875, 
            0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75, 
            0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454, 
            0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125
        ]
        
        # Pre-defined timestep schedules for each valid shift (fix_nfe=8, excluding final 0)
        SHIFT_TIMESTEPS = {
            1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
            2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693, 0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
            3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3],
        }
        
        # Determine the timestep schedule to use
        t_schedule_list = None
        
        if timesteps is not None:
            # Process custom timesteps: map each value to nearest valid timestep
            timesteps_list = timesteps.tolist() if isinstance(timesteps, torch.Tensor) else list(timesteps)
            
            # Remove trailing zeros
            while len(timesteps_list) > 0 and timesteps_list[-1] == 0:
                timesteps_list.pop()
            
            # Validate length: 1-20
            if len(timesteps_list) < 1:
                logger.warning(f"timesteps length is too short after removing trailing zeros, using default shift={shift}")
            elif len(timesteps_list) > 20:
                logger.warning(f"timesteps length={len(timesteps_list)} exceeds maximum 20, truncating to 20")
                timesteps_list = timesteps_list[:20]
                t_schedule_list = timesteps_list
            else:
                t_schedule_list = timesteps_list
            
            if t_schedule_list is not None:
                # Map each timestep to nearest valid timestep
                original_timesteps = t_schedule_list.copy()
                mapped_timesteps = []
                for t in t_schedule_list:
                    nearest = min(VALID_TIMESTEPS, key=lambda x: abs(x - t))
                    mapped_timesteps.append(nearest)
                
                if original_timesteps != mapped_timesteps:
                    logger.warning(f"timesteps mapped to nearest valid values: {original_timesteps} -> {mapped_timesteps}")
                
                t_schedule_list = mapped_timesteps

        # Build a schedule from the requested number of steps.
        if t_schedule_list is None and infer_steps is not None and infer_steps > 0:
            n = min(int(infer_steps), 20)
            if n != int(infer_steps):
                logger.warning(
                    "infer_steps=%d exceeds maximum 20, clamping to 20.", infer_steps
                )
            raw = [1.0 - i / n for i in range(n)]
            if shift != 1.0:
                raw = [shift * t / (1.0 + (shift - 1.0) * t) for t in raw]
            t_schedule_list = raw

        if t_schedule_list is None:
            # Use shift-based schedule: round to nearest valid shift
            original_shift = shift
            shift = min(VALID_SHIFTS, key=lambda x: abs(x - shift))
            if original_shift != shift:
                logger.warning(f"shift={original_shift} not supported, rounded to nearest valid shift={shift}")
            t_schedule_list = SHIFT_TIMESTEPS[shift]
        
        if attention_mask is None:
            latent_length = src_latents.shape[1]
            attention_mask = torch.ones(src_latents.shape[0], latent_length, device=src_latents.device, dtype=src_latents.dtype)
        
        time_costs = {}
        start_time = time.time()
        total_start_time = start_time
        encoder_hidden_states, encoder_attention_mask, context_latents = self.prepare_condition(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=src_latents,
            attention_mask=attention_mask,
            silence_latent=silence_latent,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
            audio_codes=audio_codes,
        )
        
        encoder_hidden_states_non_cover, encoder_attention_mask_non_cover, context_latents_non_cover = None, None, None
        if audio_cover_strength < 1.0:
            non_is_covers = torch.zeros_like(is_covers, device=is_covers.device, dtype=is_covers.dtype)
            # Use silence_latent for non-cover condition to simulate text2music mode (no reference audio)
            silence_latent_expanded = silence_latent[:, :src_latents.shape[1], :].expand(src_latents.shape[0], -1, -1)
            encoder_hidden_states_non_cover, encoder_attention_mask_non_cover, context_latents_non_cover = self.prepare_condition(
            text_hidden_states=non_cover_text_hidden_states,
            text_attention_mask=non_cover_text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=silence_latent_expanded,
            attention_mask=attention_mask,
            silence_latent=silence_latent,
            src_latents=silence_latent_expanded,
            chunk_masks=chunk_masks,
            is_covers=non_is_covers,
            precomputed_lm_hints_25Hz=None,
            audio_codes=None,
        )
        
        end_time = time.time()
        time_costs["encoder_time_cost"] = end_time - start_time
        start_time = end_time
        
        noise = self.prepare_noise(context_latents, seed)
        # Retake mixing: variance-preserving blend with an independent noise draw.
        # v=0 -> noise unchanged; v=1 -> equivalent to using retake_seed as the main seed.
        if retake_variance > 0.0:
            retake_noise = self.prepare_noise(context_latents, retake_seed)
            v_rad = retake_variance * (math.pi / 2.0)
            noise = math.cos(v_rad) * noise + math.sin(v_rad) * retake_noise
        bsz, device, dtype = context_latents.shape[0], context_latents.device, context_latents.dtype
        past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        
        # Cover noise initialization: blend noise with src_latents
        if cover_noise_strength > 0.0:
            # cover_noise_strength=1 means closest to src, so noise_level should be low
            effective_noise_level = 1.0 - cover_noise_strength
            # Find nearest valid timestep from t_schedule
            nearest_t = min(t_schedule_list, key=lambda x: abs(x - effective_noise_level))
            # xt = nearest_t * noise + (1 - nearest_t) * src_latents
            xt = self.renoise(src_latents, nearest_t, noise)
            # Truncate t_schedule to start from nearest_t
            start_idx = t_schedule_list.index(nearest_t)
            t_schedule_list = t_schedule_list[start_idx:]
            logger.info(
                f"[generate_audio] Cover mode: cover_noise_strength={cover_noise_strength}, "
                f"effective_noise_level={effective_noise_level:.4f}, nearest_t={nearest_t:.4f}, "
                f"remaining_steps={len(t_schedule_list)}"
            )
        else:
            xt = noise
        
        # Use pre-computed t_schedule_list (already validated and mapped to valid timesteps)
        t_schedule = torch.tensor(t_schedule_list, device=device, dtype=dtype)
        num_steps = len(t_schedule)
        
        # Recalculate cover_steps based on actual num_steps
        use_heun = sampler_mode == "heun"
        use_norm_clamp = velocity_norm_threshold > 0.0
        use_ema = velocity_ema_factor > 0.0
        prev_vt = None
        if use_heun and infer_method == "sde":
            logger.warning("Heun sampler is not compatible with SDE; falling back to Euler.")
            use_heun = False

        # DCW — opt-in per-band wavelet-domain correction (CVPR 2026).
        # No-op unless `dcw_enabled=True` and a non-zero scaler is configured.
        dcw_corrector = DCWCorrector(
            enabled=dcw_enabled,
            mode=dcw_mode,
            scaler=dcw_scaler,
            high_scaler=dcw_high_scaler,
            wavelet=dcw_wavelet,
        )

        cover_steps = int(num_steps * audio_cover_strength)
        _switched_to_non_cover = False
        for step_idx in range(num_steps):
            current_timestep = t_schedule[step_idx].item()
            t_curr_tensor = current_timestep * torch.ones((bsz,), device=device, dtype=dtype)
            
            if step_idx >= cover_steps and not _switched_to_non_cover:
                _switched_to_non_cover = True
                encoder_hidden_states = encoder_hidden_states_non_cover
                encoder_attention_mask = encoder_attention_mask_non_cover
                context_latents = context_latents_non_cover
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            
            with torch.no_grad():        
                decoder_outputs = self.decoder(
                    hidden_states=xt,
                    timestep=t_curr_tensor,
                    timestep_r=t_curr_tensor,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                
            vt = decoder_outputs[0]
            past_key_values = decoder_outputs[1]

            # Velocity norm clamping — prevents outlier predictions
            if use_norm_clamp:
                vt_norm = torch.norm(vt, dim=(1, 2), keepdim=True)
                xt_norm = torch.norm(xt, dim=(1, 2), keepdim=True) + 1e-10
                scale = torch.clamp(velocity_norm_threshold * xt_norm / (vt_norm + 1e-10), max=1.0)
                vt = vt * scale

            # Velocity EMA smoothing — stabilises denoising trajectory
            if use_ema and prev_vt is not None:
                vt = (1.0 - velocity_ema_factor) * vt + velocity_ema_factor * prev_vt

            # Cache pre-step latent so DCW can reconstruct the predicted
            # clean sample `denoised = x - v * t` after the sampler update.
            # Also stash the raw velocity (pre-Heun-averaging) so the x0
            # reconstruction uses the single-evaluation v(t_curr), matching
            # the reference FLUX scheduler's `x0 = sample - sigma * v`.
            xt_before_step = xt
            vt_for_denoise = vt

            # On final step, directly compute x0 from noise
            if step_idx == num_steps - 1:
                xt = self.get_x0_from_noise(xt, vt, t_curr_tensor)
                if dcw_corrector.is_active:
                    t_unsq = current_timestep * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                    denoised = xt_before_step - vt_for_denoise * t_unsq
                    xt = dcw_corrector.apply(xt, denoised, current_timestep)
                prev_vt = vt
                break

            # Update x_t based on inference method
            if infer_method == "sde":
                # Stochastic Differential Equation: predict clean, then re-add noise
                pred_clean = self.get_x0_from_noise(xt, vt, t_curr_tensor)
                next_timestep = t_schedule[step_idx + 1].item()
                xt = self.renoise(pred_clean, next_timestep)
                t_after_step = next_timestep
            elif use_heun and infer_method == "ode":
                # Heun (second-order) ODE step via trapezoidal rule
                next_timestep = t_schedule[step_idx + 1].item()
                dt = current_timestep - next_timestep
                dt_tensor = dt * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                xt_predicted = xt - vt * dt_tensor
                # Corrector: evaluate model at the predicted point
                t_next_tensor = next_timestep * torch.ones((bsz,), device=device, dtype=dtype)
                corrector_kv = EncoderDecoderCache(DynamicCache(), DynamicCache())
                with torch.no_grad():
                    decoder_outputs2 = self.decoder(
                        hidden_states=xt_predicted,
                        timestep=t_next_tensor,
                        timestep_r=t_next_tensor,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        context_latents=context_latents,
                        use_cache=False,
                        past_key_values=corrector_kv,
                    )
                vt2 = decoder_outputs2[0]
                if use_norm_clamp:
                    vt2_norm = torch.norm(vt2, dim=(1, 2), keepdim=True)
                    xt_pred_norm = torch.norm(xt_predicted, dim=(1, 2), keepdim=True) + 1e-10
                    scale2 = torch.clamp(velocity_norm_threshold * xt_pred_norm / (vt2_norm + 1e-10), max=1.0)
                    vt2 = vt2 * scale2
                if use_ema:
                    vt2 = (1.0 - velocity_ema_factor) * vt2 + velocity_ema_factor * vt
                vt_avg = 0.5 * (vt + vt2)
                xt = xt - vt_avg * dt_tensor
                vt = vt_avg
                t_after_step = next_timestep
            elif infer_method == "ode":
                # Ordinary Differential Equation: Euler method
                # dx/dt = -v, so x_{t+1} = x_t - v_t * dt
                next_timestep = t_schedule[step_idx + 1].item()
                dt = current_timestep - next_timestep
                dt_tensor = dt * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                xt = xt - vt * dt_tensor
                t_after_step = next_timestep

            # DCW correction — push x_next's frequency band(s) away from
            # the predicted clean sample.  Scaler decays with t_curr.
            if dcw_corrector.is_active:
                t_unsq = current_timestep * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                denoised = xt_before_step - vt_for_denoise * t_unsq
                xt = dcw_corrector.apply(xt, denoised, current_timestep)

            prev_vt = vt

            injection_cutoff = round(repaint_injection_ratio * num_steps)
            if repaint_mask is not None and clean_src_latents is not None and step_idx < injection_cutoff:
                xt = _repaint_step_injection(
                    xt, clean_src_latents, repaint_mask, t_after_step, noise,
                )
        
        x_gen = xt
        if repaint_mask is not None and clean_src_latents is not None and repaint_crossfade_frames > 0:
            x_gen = _repaint_boundary_blend(
                x_gen, clean_src_latents, repaint_mask, repaint_crossfade_frames,
            )
        end_time = time.time()
        time_costs["diffusion_time_cost"] = end_time - start_time
        time_costs["diffusion_per_step_time_cost"] = time_costs["diffusion_time_cost"] / num_steps
        time_costs["total_time_cost"] = end_time - total_start_time
        return {
            "target_latents": x_gen,
            "time_costs": time_costs,
        }
