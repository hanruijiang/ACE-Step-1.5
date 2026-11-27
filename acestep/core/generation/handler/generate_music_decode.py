"""Decode/validation helpers for ``generate_music`` orchestration."""

import gc
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger

from acestep.gpu_config import get_effective_free_vram_gb


class GenerateMusicDecodeMixin:
    """Validate generated latents and decode them into waveform tensors."""

    def _prepare_generate_music_decode_state(
        self,
        outputs: Dict[str, Any],
        latent_shift: float,
        latent_rescale: float,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Collect decode inputs and validate raw diffusion latents.

        Args:
            outputs: ``service_generate`` output payload containing target latents and timings.
            latent_shift: Additive latent post-processing shift.
            latent_rescale: Multiplicative latent post-processing scale.

        Returns:
            Tuple containing validated ``pred_latents`` and mutable ``time_costs``.

        Raises:
            RuntimeError: If latents contain NaN/Inf values or collapse to all zeros.
        """
        logger.info("[generate_music] Model generation completed. Decoding latents...")
        pred_latents = outputs["target_latents"]
        time_costs = outputs["time_costs"]
        time_costs["offload_time_cost"] = self.current_offload_cost


        if self.debug_stats:
            logger.debug(
                f"[generate_music] pred_latents: {pred_latents.shape}, dtype={pred_latents.dtype} "
                f"{pred_latents.min()=}, {pred_latents.max()=}, {pred_latents.mean()=} "
                f"{pred_latents.std()=}"
            )
        else:
            logger.debug(f"[generate_music] pred_latents: {pred_latents.shape}, dtype={pred_latents.dtype}")
        logger.debug(f"[generate_music] time_costs: {time_costs}")

        if not torch.isfinite(pred_latents).all():
            raise RuntimeError("Generation produced NaN or Inf latents.")
        if pred_latents.numel() > 0 and pred_latents.abs().sum() == 0:
            raise RuntimeError("Generation produced zero latents.")
        if latent_shift != 0.0 or latent_rescale != 1.0:
            logger.info(
                f"[generate_music] Applying latent post-processing: shift={latent_shift}, "
                f"rescale={latent_rescale}"
            )
            if self.debug_stats:
                logger.debug(
                    f"[generate_music] Latent BEFORE shift/rescale: min={pred_latents.min():.4f}, "
                    f"max={pred_latents.max():.4f}, mean={pred_latents.mean():.4f}, "
                    f"std={pred_latents.std():.4f}"
                )
            pred_latents = pred_latents * latent_rescale + latent_shift
            if self.debug_stats:
                logger.debug(
                    f"[generate_music] Latent AFTER shift/rescale: min={pred_latents.min():.4f}, "
                    f"max={pred_latents.max():.4f}, mean={pred_latents.mean():.4f}, "
                    f"std={pred_latents.std():.4f}"
                )
        return pred_latents, time_costs

    def _decode_generate_music_pred_latents(
        self,
        pred_latents: torch.Tensor,
        progress: Any,
        use_tiled_decode: bool,
        time_costs: Dict[str, Any],
        return_latents: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Decode predicted latents and update decode timing metrics.

        Args:
            pred_latents: Predicted latent tensor shaped ``[batch, frames, dim]``.
            progress: Optional progress callback.
            use_tiled_decode: Whether tiled VAE decode should be used.
            time_costs: Mutable time-cost payload from service generation.
            return_latents: Include a copy of the predicted latents on CPU.

        Returns:
            Tuple of decoded waveforms, CPU latents, and updated time-cost payload.
        """
        if progress:
            progress(0.8, desc="Decoding audio...")
        logger.info("[generate_music] Decoding latents with VAE...")
        start_time = time.time()
        with torch.inference_mode():
            with self._load_model_context("vae"):
                pred_latents_cpu = pred_latents.detach().cpu() if return_latents else None
                pred_latents_for_decode = pred_latents.transpose(1, 2).contiguous().to(self.vae.dtype)

                logger.debug(
                    "[generate_music] Before VAE decode: "
                    f"allocated={self._memory_allocated()/1024**3:.2f}GB, "
                    f"max={self._max_memory_allocated()/1024**3:.2f}GB"
                )
                vae_device = None
                vae_cpu = os.environ.get("ACESTEP_VAE_ON_CPU", "0").lower() in ("1", "true", "yes")
                if not vae_cpu and self.device != "mps":
                    vae_cpu = get_effective_free_vram_gb() < 0.5
                if vae_cpu:
                    logger.info("[generate_music] Decoding with the VAE on CPU.")
                    vae_device = next(self.vae.parameters()).device
                    self.vae = self.vae.cpu()
                    pred_latents_for_decode = pred_latents_for_decode.cpu()
                    self._empty_cache()
                try:
                    if use_tiled_decode:
                        logger.info("[generate_music] Using tiled VAE decode to reduce VRAM usage...")
                        pred_wavs = self.tiled_decode(pred_latents_for_decode)
                    else:
                        decoder_output = self.vae.decode(pred_latents_for_decode)
                        pred_wavs = decoder_output.sample
                        del decoder_output
                finally:
                    if vae_cpu and vae_device is not None:
                        logger.info("[generate_music] Restoring VAE to original device after CPU decode path...")
                        self.vae = self.vae.to(vae_device)
                    self._empty_cache()
                logger.debug(
                    "[generate_music] After VAE decode: "
                    f"allocated={self._memory_allocated()/1024**3:.2f}GB, "
                    f"max={self._max_memory_allocated()/1024**3:.2f}GB"
                )
                del pred_latents_for_decode
                if pred_wavs.dtype != torch.float32:
                    pred_wavs = pred_wavs.float()
                peak = pred_wavs.abs().amax(dim=[1, 2], keepdim=True)
                if torch.any(peak > 1.0):
                    pred_wavs = pred_wavs / peak.clamp(min=1.0)
                self._empty_cache()
        gc.collect()
        self._empty_cache()
        end_time = time.time()
        time_costs["vae_decode_time_cost"] = end_time - start_time
        time_costs["total_time_cost"] = time_costs["total_time_cost"] + time_costs["vae_decode_time_cost"]
        time_costs["offload_time_cost"] = self.current_offload_cost
        return pred_wavs, pred_latents_cpu, time_costs
