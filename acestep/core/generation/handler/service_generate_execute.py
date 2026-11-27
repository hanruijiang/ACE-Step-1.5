"""Execution helpers for service generation diffusion and output assembly."""

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger


class ServiceGenerateExecuteMixin:
    """Run diffusion execution for normalized service-generation requests."""

    def _unpack_service_processed_data(self, processed_data: Tuple[Any, ...]) -> Dict[str, Any]:
        """Convert batch preprocessing tuple into a keyed payload."""
        (
            keys,
            text_inputs,
            src_latents,
            target_latents,
            text_hidden_states,
            text_attention_mask,
            lyric_hidden_states,
            lyric_attention_mask,
            _audio_attention_mask,
            refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask,
            chunk_mask,
            spans,
            is_covers,
            _audio_codes,
            lyric_token_idss,
            precomputed_lm_hints_25Hz,
            non_cover_text_hidden_states,
            non_cover_text_attention_masks,
            repaint_mask,
        ) = processed_data
        return {
            "keys": keys,
            "text_inputs": text_inputs,
            "src_latents": src_latents,
            "target_latents": target_latents,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "lyric_hidden_states": lyric_hidden_states,
            "lyric_attention_mask": lyric_attention_mask,
            "refer_audio_acoustic_hidden_states_packed": refer_audio_acoustic_hidden_states_packed,
            "refer_audio_order_mask": refer_audio_order_mask,
            "chunk_mask": chunk_mask,
            "spans": spans,
            "is_covers": is_covers,
            "lyric_token_idss": lyric_token_idss,
            "precomputed_lm_hints_25Hz": precomputed_lm_hints_25Hz,
            "non_cover_text_hidden_states": non_cover_text_hidden_states,
            "non_cover_text_attention_masks": non_cover_text_attention_masks,
            "repaint_mask": repaint_mask,
        }

    def _resolve_service_seed_param(self, seed_list: Optional[List[int]]) -> Any:
        """Return model seed parameter: per-item seed list or random single seed."""
        if seed_list is not None:
            return seed_list
        return random.randint(0, 2**32 - 1)

    def _resolve_service_dcw_enabled(self, generate_kwargs: Dict[str, Any]) -> bool:
        """Return an explicit or loaded-model DCW value for backend execution."""
        dcw_enabled = generate_kwargs.get("dcw_enabled")
        if dcw_enabled is not None:
            return bool(dcw_enabled)
        config = getattr(self, "config", None)
        return bool(getattr(config, "is_turbo", False))

    def _build_service_generate_kwargs(
        self,
        payload: Dict[str, Any],
        seed_param: Any,
        infer_steps: int,
        guidance_scale: float,
        audio_cover_strength: float,
        cover_noise_strength: float,
        infer_method: str,
        use_adg: bool,
        cfg_interval_start: float,
        cfg_interval_end: float,
        shift: float,
        timesteps: Optional[List[float]],
        repaint_crossfade_frames: int = 10,
        repaint_injection_ratio: float = 0.5,
        sampler_mode: str = "euler",
        velocity_norm_threshold: float = 0.0,
        velocity_ema_factor: float = 0.0,
        dcw_enabled: Optional[bool] = None,
        dcw_mode: str = "double",
        dcw_scaler: float = 0.05,
        dcw_high_scaler: float = 0.02,
        dcw_wavelet: str = "haar",
        retake_seed: Any = None,
        retake_variance: float = 0.0,
    ) -> Dict[str, Any]:
        """Build kwargs passed to model generation backends."""
        repaint_mask = payload.get("repaint_mask")
        clean_src_latents = payload.get("target_latents") if repaint_mask is not None else None

        kwargs = {
            "text_hidden_states": payload["text_hidden_states"],
            "text_attention_mask": payload["text_attention_mask"],
            "lyric_hidden_states": payload["lyric_hidden_states"],
            "lyric_attention_mask": payload["lyric_attention_mask"],
            "refer_audio_acoustic_hidden_states_packed": payload["refer_audio_acoustic_hidden_states_packed"],
            "refer_audio_order_mask": payload["refer_audio_order_mask"],
            "src_latents": payload["src_latents"],
            "chunk_masks": payload["chunk_mask"],
            "is_covers": payload["is_covers"],
            "silence_latent": self.silence_latent,
            "seed": seed_param,
            "non_cover_text_hidden_states": payload["non_cover_text_hidden_states"],
            "non_cover_text_attention_mask": payload["non_cover_text_attention_masks"],
            "precomputed_lm_hints_25Hz": payload["precomputed_lm_hints_25Hz"],
            "audio_cover_strength": audio_cover_strength,
            "cover_noise_strength": cover_noise_strength,
            "infer_method": infer_method,
            "infer_steps": infer_steps,
            "diffusion_guidance_scale": guidance_scale,
            "use_adg": use_adg,
            "cfg_interval_start": cfg_interval_start,
            "cfg_interval_end": cfg_interval_end,
            "shift": shift,
            "repaint_mask": repaint_mask,
            "clean_src_latents": clean_src_latents,
            "repaint_crossfade_frames": repaint_crossfade_frames,
            "repaint_injection_ratio": repaint_injection_ratio,
            "sampler_mode": sampler_mode,
            "velocity_norm_threshold": velocity_norm_threshold,
            "velocity_ema_factor": velocity_ema_factor,
            "dcw_enabled": dcw_enabled,
            "dcw_mode": dcw_mode,
            "dcw_scaler": dcw_scaler,
            "dcw_high_scaler": dcw_high_scaler,
            "dcw_wavelet": dcw_wavelet,
            "retake_seed": retake_seed,
            "retake_variance": retake_variance,
        }
        if timesteps is not None:
            kwargs["timesteps"] = torch.tensor(timesteps, dtype=torch.float32, device=self.device)
        return kwargs

    def _execute_service_generate_diffusion(
        self,
        payload: Dict[str, Any],
        generate_kwargs: Dict[str, Any],
        seed_param: Any,
        infer_method: str,
        shift: float,
        audio_cover_strength: float,
        retake_seed: Any = None,
        retake_variance: float = 0.0,
        flow_edit_ctx: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute condition preparation and PyTorch diffusion."""
        generate_kwargs = {
            **generate_kwargs,
            "dcw_enabled": self._resolve_service_dcw_enabled(generate_kwargs),
        }
        if flow_edit_ctx is not None and flow_edit_ctx.get("morph"):
            from .service_generate_flow_edit import dispatch_flow_edit_overlay

            return dispatch_flow_edit_overlay(
                self, payload=payload, generate_kwargs=generate_kwargs,
                seed_param=seed_param, flow_edit_ctx=flow_edit_ctx,
            )
        logger.info(f"[service_generate] Generating audio with PyTorch ({self.device})")
        with torch.inference_mode():
            with self._load_model_context("model"):
                encoder_hidden_states, encoder_attention_mask, context_latents = self.model.prepare_condition(
                    text_hidden_states=payload["text_hidden_states"],
                    text_attention_mask=payload["text_attention_mask"],
                    lyric_hidden_states=payload["lyric_hidden_states"],
                    lyric_attention_mask=payload["lyric_attention_mask"],
                    refer_audio_acoustic_hidden_states_packed=payload["refer_audio_acoustic_hidden_states_packed"],
                    refer_audio_order_mask=payload["refer_audio_order_mask"],
                    hidden_states=payload["src_latents"],
                    attention_mask=torch.ones(
                        payload["src_latents"].shape[0],
                        payload["src_latents"].shape[1],
                        device=payload["src_latents"].device,
                        dtype=payload["src_latents"].dtype,
                    ),
                    silence_latent=self.silence_latent,
                    src_latents=payload["src_latents"],
                    chunk_masks=payload["chunk_mask"],
                    is_covers=payload["is_covers"],
                    precomputed_lm_hints_25Hz=payload["precomputed_lm_hints_25Hz"],
                )

                outputs = self.model.generate_audio(**generate_kwargs)

        return outputs, encoder_hidden_states, encoder_attention_mask, context_latents
