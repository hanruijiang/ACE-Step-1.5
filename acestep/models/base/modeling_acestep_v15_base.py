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

from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.utils import logging
from tqdm import tqdm


try:
    from .configuration_acestep_v15 import AceStepConfig
    from .apg_guidance import adg_forward, apg_forward, cfg_forward, MomentumBuffer
except ImportError:
    from configuration_acestep_v15 import AceStepConfig
    from apg_guidance import adg_forward, apg_forward, cfg_forward, MomentumBuffer

from acestep.models.common.dcw_correction import DCWCorrector


logger = logging.get_logger(__name__)

from acestep.models.common.modeling_layers import (
    create_4d_mask,
    AceStepPreTrainedModel,
    AudioTokenDetokenizer,
    _repaint_step_injection,
    _repaint_boundary_blend,
    pack_sequences,
    AceStepTimbreEncoder,
    AceStepAudioTokenizer,
    AceStepDiTModel,
    AceStepConditionEncoder,
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
        # Build encoder config: use separate encoder_hidden_size when available
        # (4B models have encoder_hidden_size=2048 != hidden_size=2560)
        _enc_hs = getattr(config, "encoder_hidden_size", None) or config.hidden_size
        if _enc_hs != config.hidden_size:
            encoder_config = copy.deepcopy(config)
            encoder_config.hidden_size = _enc_hs
            encoder_config.intermediate_size = getattr(config, "encoder_intermediate_size", None) or config.intermediate_size
            encoder_config.num_attention_heads = getattr(config, "encoder_num_attention_heads", None) or config.num_attention_heads
            encoder_config.num_key_value_heads = getattr(config, "encoder_num_key_value_heads", None) or config.num_key_value_heads
        else:
            encoder_config = config
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
        bsz = context_latents.shape[0]
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
        if t.shape[0] != zt.shape[0]:
            raise ValueError(f"Batch size mismatch: t has {t.shape[0]}, zt has {zt.shape[0]}")

        return zt - vt * t.unsqueeze(-1).unsqueeze(-1)

    def renoise(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        if isinstance(t, torch.Tensor) and t.ndim != x.ndim:
            t = t.unsqueeze(-1).unsqueeze(-1)
        xt = t * noise + (1 - t) * x
        return xt

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
        infer_method: str = "ode",
        use_cache: bool = True,
        infer_steps: int = 30,
        diffusion_guidance_scale: float = 7.0,
        audio_cover_strength: float = 1.0,
        non_cover_text_hidden_states: Optional[torch.FloatTensor] = None,
        non_cover_text_attention_mask: Optional[torch.FloatTensor] = None,
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        precomputed_lm_hints_25Hz: Optional[torch.FloatTensor] = None,
        audio_codes: Optional[torch.FloatTensor] = None,
        use_progress_bar: bool = True,
        use_adg: bool = False,
        shift: float = 1.0,
        cover_noise_strength: float = 0.0,
        repaint_mask: Optional[torch.Tensor] = None,
        clean_src_latents: Optional[torch.FloatTensor] = None,
        repaint_crossfade_frames: int = 10,
        repaint_injection_ratio: float = 0.5,
        sampler_mode: str = "euler",
        velocity_norm_threshold: float = 0.0,
        velocity_ema_factor: float = 0.0,
        timesteps: Optional[torch.Tensor] = None,
        dcw_enabled: bool = True,
        dcw_mode: str = "double",
        dcw_scaler: float = 0.05,
        dcw_high_scaler: float = 0.02,
        dcw_wavelet: str = "haar",
        **kwargs,
    ):
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

        # Calculate cover steps based on audio_cover_strength
        cover_steps = int(infer_steps * audio_cover_strength)
        device, dtype = context_latents.device, context_latents.dtype
        # Use custom timesteps if provided, otherwise compute from infer_steps and shift
        if timesteps is not None:
            t = timesteps.to(device=device, dtype=dtype)
            infer_steps = len(t) - 1  # Override infer_steps based on timesteps length
        else:
            t = torch.linspace(1.0, 0.0, infer_steps + 1, device=device, dtype=dtype)
            # Apply shift transformation to timesteps if shift != 1.0
            if shift != 1.0:
                t = shift * t / (1 + (shift - 1) * t)
        if use_progress_bar:
            iterator = tqdm(zip(t[:-1], t[1:]), total=infer_steps)
        else:
            iterator = zip(t[:-1], t[1:])

        noise = self.prepare_noise(context_latents, seed)
        # Retake mixing: variance-preserving blend with an independent noise draw.
        # v=0 -> noise unchanged; v=1 -> equivalent to using retake_seed as the main seed.
        if retake_variance > 0.0:
            retake_noise = self.prepare_noise(context_latents, retake_seed)
            v_rad = retake_variance * (math.pi / 2.0)
            noise = math.cos(v_rad) * noise + math.sin(v_rad) * retake_noise
        bsz, device, dtype = context_latents.shape[0], context_latents.device, context_latents.dtype
        past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        momentum_buffer = MomentumBuffer()

        # Cover noise initialization: blend noise with src_latents
        if cover_noise_strength > 0.0:
            # cover_noise_strength=1 means closest to src, so noise_level should be low
            effective_noise_level = 1.0 - cover_noise_strength
            # Find nearest timestep in the schedule t[:-1]
            t_values = t[:-1].tolist()
            nearest_t = min(t_values, key=lambda x: abs(x - effective_noise_level))
            start_idx = t_values.index(nearest_t)
            # xt = nearest_t * noise + (1 - nearest_t) * src_latents
            xt = self.renoise(src_latents, nearest_t, noise)
            # Truncate schedule to start from nearest_t
            t = t[start_idx:]
            infer_steps = len(t) - 1
            cover_steps = int(infer_steps * audio_cover_strength)
            if use_progress_bar:
                iterator = tqdm(zip(t[:-1], t[1:]), total=infer_steps)
            else:
                iterator = zip(t[:-1], t[1:])
            logger.info(
                f"[generate_audio] Cover mode: cover_noise_strength={cover_noise_strength}, "
                f"effective_noise_level={effective_noise_level:.4f}, nearest_t={nearest_t:.4f}, "
                f"remaining_steps={infer_steps}"
            )
        else:
            xt = noise

        # main task condition
        do_cfg_guidance = diffusion_guidance_scale > 1.0
        if do_cfg_guidance:
            encoder_hidden_states = torch.cat([encoder_hidden_states, self.null_condition_emb.expand_as(encoder_hidden_states)], dim=0)
            encoder_attention_mask = torch.cat([encoder_attention_mask, encoder_attention_mask], dim=0)
            # src_latents
            context_latents = torch.cat([context_latents, context_latents], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

        use_heun = sampler_mode == "heun"
        use_norm_clamp = velocity_norm_threshold > 0.0
        use_ema = velocity_ema_factor > 0.0
        prev_vt = None
        if use_heun and infer_method == "sde":
            logger.warning("Heun sampler is not compatible with SDE; falling back to Euler.")
            use_heun = False

        # DCW — Differential Correction in Wavelet domain (CVPR 2026).
        # No-op unless `dcw_enabled=True` and a non-zero scaler is configured.
        dcw_corrector = DCWCorrector(
            enabled=dcw_enabled,
            mode=dcw_mode,
            scaler=dcw_scaler,
            high_scaler=dcw_high_scaler,
            wavelet=dcw_wavelet,
        )

        _switched_to_non_cover = False
        with torch.no_grad():
            for step_idx, (t_curr, t_prev) in enumerate(iterator):
                if step_idx >= cover_steps and not _switched_to_non_cover:
                    _switched_to_non_cover = True
                    if do_cfg_guidance:
                        encoder_hidden_states_non_cover = torch.cat([encoder_hidden_states_non_cover, self.null_condition_emb.expand_as(encoder_hidden_states_non_cover)], dim=0)
                        encoder_attention_mask_non_cover = torch.cat([encoder_attention_mask_non_cover, encoder_attention_mask_non_cover], dim=0)
                        # src_latents
                        context_latents_non_cover = torch.cat([context_latents_non_cover, context_latents_non_cover], dim=0)

                    encoder_hidden_states = encoder_hidden_states_non_cover
                    encoder_attention_mask = encoder_attention_mask_non_cover
                    context_latents = context_latents_non_cover
                    past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

                x = torch.cat([xt, xt], dim=0) if do_cfg_guidance else xt
                t_curr_tensor = t_curr * torch.ones((x.shape[0],), device=device, dtype=dtype)
                decoder_outputs = self.decoder(
                    hidden_states=x,
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
                apply_cfg_guidance = t_curr >= cfg_interval_start and t_curr <= cfg_interval_end
                if do_cfg_guidance:
                    pred_cond, pred_null_cond = vt.chunk(2)
                    if apply_cfg_guidance:
                        if not use_adg:
                            vt = apg_forward(
                                pred_cond=pred_cond,
                                pred_uncond=pred_null_cond,
                                guidance_scale=diffusion_guidance_scale,
                                momentum_buffer=momentum_buffer,
                                dims=[1],
                            )
                        else:
                            vt = adg_forward(
                                latents=xt,
                                noise_pred_cond=pred_cond,
                                noise_pred_uncond=pred_null_cond,
                                sigma=t_curr,
                                guidance_scale=diffusion_guidance_scale,
                            )
                    else:
                        vt = pred_cond
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
                # Also stash the *raw* velocity (pre-Heun-averaging) so the
                # reconstruction uses the single-evaluation `v(t_curr)`, which
                # is what the reference FLUX scheduler's `x0 = sample - sigma *
                # model_output` assumes; Heun below overwrites `vt` with the
                # trapezoidal average, which would bias the x0 estimate.
                xt_before_step = xt
                vt_for_denoise = vt

                # Update x_t based on inference method
                if infer_method == "sde":
                    # Stochastic Differential Equation: predict clean, then re-add noise
                    t_curr_bsz = t_curr * torch.ones((bsz,), device=device, dtype=dtype)
                    pred_clean = self.get_x0_from_noise(xt, vt, t_curr_bsz)
                    next_timestep = 1.0 - (float(step_idx + 1) / infer_steps)
                    xt = self.renoise(pred_clean, next_timestep)
                    t_after_step = next_timestep
                elif use_heun and infer_method == "ode":
                    # Heun (second-order) ODE step via trapezoidal rule
                    dt = t_curr - t_prev
                    dt_tensor = dt * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                    # Predictor: Euler step to get xt_predicted at t_prev
                    xt_predicted = xt - vt * dt_tensor
                    # Corrector: evaluate model at the predicted point
                    x2 = torch.cat([xt_predicted, xt_predicted], dim=0) if do_cfg_guidance else xt_predicted
                    t_prev_tensor = t_prev * torch.ones((x2.shape[0],), device=device, dtype=dtype)
                    # Reset KV cache for corrector pass (Heun needs fresh evaluation)
                    corrector_kv = EncoderDecoderCache(DynamicCache(), DynamicCache())
                    decoder_outputs2 = self.decoder(
                        hidden_states=x2,
                        timestep=t_prev_tensor,
                        timestep_r=t_prev_tensor,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        context_latents=context_latents,
                        use_cache=False,
                        past_key_values=corrector_kv,
                    )
                    vt2 = decoder_outputs2[0]
                    if do_cfg_guidance:
                        pred_cond2, pred_null_cond2 = vt2.chunk(2)
                        # Recompute CFG interval for corrector timestep (t_prev, not t_curr)
                        apply_cfg_corrector = t_prev >= cfg_interval_start and t_prev <= cfg_interval_end
                        if apply_cfg_corrector:
                            if not use_adg:
                                # Use basic CFG for corrector to avoid mutating APG momentum twice per step
                                vt2 = cfg_forward(pred_cond2, pred_null_cond2, diffusion_guidance_scale)
                            elif t_prev > 0:
                                # Guard against sigma=0 which causes NaN in ADG division
                                vt2 = adg_forward(
                                    latents=xt_predicted,
                                    noise_pred_cond=pred_cond2,
                                    noise_pred_uncond=pred_null_cond2,
                                    sigma=t_prev,
                                    guidance_scale=diffusion_guidance_scale,
                                )
                            else:
                                vt2 = cfg_forward(pred_cond2, pred_null_cond2, diffusion_guidance_scale)
                        else:
                            vt2 = pred_cond2
                    if use_norm_clamp:
                        vt2_norm = torch.norm(vt2, dim=(1, 2), keepdim=True)
                        xt_pred_norm = torch.norm(xt_predicted, dim=(1, 2), keepdim=True) + 1e-10
                        scale2 = torch.clamp(velocity_norm_threshold * xt_pred_norm / (vt2_norm + 1e-10), max=1.0)
                        vt2 = vt2 * scale2
                    if use_ema:
                        vt2 = (1.0 - velocity_ema_factor) * vt2 + velocity_ema_factor * vt
                    # Average the two velocity predictions (trapezoidal rule)
                    vt_avg = 0.5 * (vt + vt2)
                    xt = xt - vt_avg * dt_tensor
                    vt = vt_avg
                    t_after_step = t_prev
                elif infer_method == "ode":
                    # Ordinary Differential Equation: Euler method
                    # dx/dt = -v, so x_{t+1} = x_t - v_t * dt
                    dt = t_curr - t_prev
                    dt_tensor = dt * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                    xt = xt - vt * dt_tensor
                    t_after_step = t_prev

                # DCW correction: push x_next's frequency band(s) away from
                # the predicted clean sample to compensate for SNR-t drift.
                # Scaler is t_curr-modulated inside the corrector, so this
                # naturally decays to identity as the trajectory approaches 0.
                if dcw_corrector.is_active:
                    t_curr_f = float(t_curr) if torch.is_tensor(t_curr) else t_curr
                    t_unsq = t_curr_f * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                    denoised = xt_before_step - vt_for_denoise * t_unsq
                    xt = dcw_corrector.apply(xt, denoised, t_curr_f)

                prev_vt = vt

                injection_cutoff = round(repaint_injection_ratio * infer_steps)
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
        time_costs["diffusion_per_step_time_cost"] = time_costs["diffusion_time_cost"] / infer_steps
        time_costs["total_time_cost"] = end_time - total_start_time
        return {
            "target_latents": x_gen,
            "time_costs": time_costs,
        }

    def flowedit_generate_audio(self, **kwargs):
        """Flow-edit: morph src toward a target prompt/lyrics.

        Thin delegate to ``models.common.flow_edit_pipeline`` so all four
        base variants share the same implementation.  See that module's
        docstring for the algorithm and supported sampler settings.
        """
        from acestep.models.common.flow_edit_pipeline import (
            flowedit_generate_audio as _flowedit_impl,
        )

        return _flowedit_impl(self, **kwargs)
