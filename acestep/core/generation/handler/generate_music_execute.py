"""Execution helper for ``generate_music`` service invocation with timeout handling."""

import os
import threading
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

# A positive timeout is opt-in; zero runs synchronously.


class GenerateMusicExecuteMixin:
    """Run service generation with an optional timeout."""

    def _run_generate_music_service(
        self,
        progress: Any,
        actual_batch_size: int,
        audio_duration: Optional[float],
        inference_steps: int,
        timesteps: Optional[Sequence[float]],
        service_inputs: Dict[str, Any],
        refer_audios: Optional[List[Any]],
        guidance_scale: float,
        actual_seed_list: Optional[List[int]],
        audio_cover_strength: float,
        cover_noise_strength: float,
        use_adg: bool,
        cfg_interval_start: float,
        cfg_interval_end: float,
        shift: float,
        infer_method: str,
        sampler_mode: str = "euler",
        velocity_norm_threshold: float = 0.0,
        velocity_ema_factor: float = 0.0,
        dcw_enabled: bool = True,
        dcw_mode: str = "double",
        dcw_scaler: float = 0.05,
        dcw_high_scaler: float = 0.02,
        dcw_wavelet: str = "haar",
        repaint_crossfade_frames: int = 10,
        repaint_injection_ratio: float = 0.5,
        source_repaint_latents: Any = None,
        task_type: str = "",
        actual_retake_seed_list: Optional[List[int]] = None,
        retake_variance: float = 0.0,
        flow_edit_morph: bool = False,
        flow_edit_source_caption: str = "",
        flow_edit_source_lyrics: str = "",
        flow_edit_n_min: float = 0.0,
        flow_edit_n_max: float = 1.0,
        flow_edit_n_avg: int = 1,
    ) -> Dict[str, Any]:
        """Invoke service generation with a timeout and optional stage callback."""
        if progress:
            progress(0.52, desc=f"Generating music (batch size: {actual_batch_size})...")

        _result: Dict[str, Any] = {}
        _error: Dict[str, BaseException] = {}

        def _service_target():
            try:
                _result["outputs"] = self.service_generate(
                    captions=service_inputs["captions_batch"],
                    global_captions=service_inputs.get("global_captions_batch"),
                    lyrics=service_inputs["lyrics_batch"],
                    metas=service_inputs["metas_batch"],
                    vocal_languages=service_inputs["vocal_languages_batch"],
                    refer_audios=refer_audios,
                    target_wavs=service_inputs["target_wavs_tensor"],
                    infer_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    seed=actual_seed_list,
                    repainting_start=service_inputs["repainting_start_batch"],
                    repainting_end=service_inputs["repainting_end_batch"],
                    instructions=service_inputs["instructions_batch"],
                    audio_cover_strength=audio_cover_strength,
                    cover_noise_strength=cover_noise_strength,
                    use_adg=use_adg,
                    cfg_interval_start=cfg_interval_start,
                    cfg_interval_end=cfg_interval_end,
                    shift=shift,
                    infer_method=infer_method,
                    sampler_mode=sampler_mode,
                    velocity_norm_threshold=velocity_norm_threshold,
                    velocity_ema_factor=velocity_ema_factor,
                    dcw_enabled=dcw_enabled,
                    dcw_mode=dcw_mode,
                    dcw_scaler=dcw_scaler,
                    dcw_high_scaler=dcw_high_scaler,
                    dcw_wavelet=dcw_wavelet,
                    audio_code_hints=service_inputs["audio_code_hints_batch"],
                    return_intermediate=service_inputs["should_return_intermediate"],
                    timesteps=timesteps,
                    chunk_mask_modes=service_inputs.get("chunk_mask_modes_batch"),
                    repaint_crossfade_frames=repaint_crossfade_frames,
                    repaint_injection_ratio=repaint_injection_ratio,
                    source_repaint_latents=source_repaint_latents,
                    task_type=task_type,
                    retake_seed=actual_retake_seed_list,
                    retake_variance=retake_variance,
                    flow_edit_morph=flow_edit_morph,
                    flow_edit_source_caption=flow_edit_source_caption,
                    flow_edit_source_lyrics=flow_edit_source_lyrics,
                    flow_edit_n_min=flow_edit_n_min,
                    flow_edit_n_max=flow_edit_n_max,
                    flow_edit_n_avg=flow_edit_n_avg,
                )
            except Exception as exc:
                _error["exc"] = exc

        timeout = int(os.environ.get("ACESTEP_GENERATION_TIMEOUT", "0"))
        if timeout <= 0:
            _service_target()
            if "exc" in _error:
                raise _error["exc"]
            return {"outputs": _result["outputs"]}

        gen_thread = threading.Thread(
            target=_service_target,
            name="service-generate",
            daemon=True,
        )
        gen_thread.start()
        gen_thread.join(timeout=timeout)

        if gen_thread.is_alive():
            logger.error(
                f"[generate_music] service_generate exceeded {timeout}s "
                f"timeout (batch={actual_batch_size}, steps={inference_steps}, "
                f"duration={audio_duration}s).  The CUDA operation may still be "
                f"running in the background."
            )
            raise TimeoutError(
                f"Music generation timed out after {timeout} seconds.  "
                f"This usually means the GPU ran out of VRAM or the diffusion loop "
                f"stalled.  Try reducing batch size, duration, or inference steps."
            )
        if "exc" in _error:
            raise _error["exc"]

        return {"outputs": _result["outputs"]}
