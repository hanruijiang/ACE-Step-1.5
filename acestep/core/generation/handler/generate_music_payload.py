"""Success payload builders for ``generate_music`` orchestration."""

from typing import Any, Dict

from acestep.gpu_config import get_global_gpu_config
from loguru import logger


class GenerateMusicPayloadMixin:
    """Build audio/metadata payload structures returned by ``generate_music``."""

    def _build_generate_music_success_payload(
        self,
        outputs: Dict[str, Any],
        pred_wavs,
        pred_latents_cpu,
        time_costs: Dict[str, Any],
        seed_value_for_ui: int,
        actual_batch_size: int,
        progress: Any,
        retake_seed_value_for_ui: str = "",
        retake_variance: float = 0.0,
    ) -> Dict[str, Any]:
        """Assemble final success response from decoded tensors and model outputs.

        Args:
            outputs: Service output payload containing intermediate generation tensors.
            pred_wavs: Decoded waveform tensor shaped ``[batch, channels, samples]``.
            pred_latents_cpu: CPU latent tensor preserved for extra outputs.
            time_costs: Updated time-cost payload including decode/offload timings.
            seed_value_for_ui: Seed value displayed in UI outputs.
            actual_batch_size: Effective generation batch size.
            progress: Optional progress callback.

        Returns:
            Dict[str, Any]: Standard success payload returned by ``generate_music``.
        """
        logger.info("[generate_music] VAE decode completed. Preparing audio tensors...")
        if progress:
            progress(0.99, desc="Preparing audio data...")

        audio_tensors = []
        for index in range(actual_batch_size):
            audio_tensor = pred_wavs[index].cpu()
            audio_tensors.append(audio_tensor)
        # Free the GPU waveform tensor now that all per-sample CPU copies are done.
        del pred_wavs

        status_message = "Generation completed successfully!"
        logger.info(f"[generate_music] Done! Generated {len(audio_tensors)} audio tensors.")

        extra_outputs = {
            "spans": outputs.get("spans", []),
            "time_costs": time_costs,
            "seed_value": seed_value_for_ui,
            "retake_seed_value": retake_seed_value_for_ui,
            "retake_variance": retake_variance,
        }
        if not get_global_gpu_config().save_memory_mode:
            extra_outputs["pred_latents"] = pred_latents_cpu

        audios = []
        for audio_tensor in audio_tensors:
            audios.append({"tensor": audio_tensor, "sample_rate": self.sample_rate})

        return {
            "audios": audios,
            "status_message": status_message,
            "extra_outputs": extra_outputs,
            "success": True,
            "error": None,
        }
