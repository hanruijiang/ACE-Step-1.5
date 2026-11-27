"""Runtime setup helpers for initialization orchestration."""

from typing import Any, Optional, Tuple

import torch
from loguru import logger

from acestep import gpu_config


class InitServiceSetupMixin:
    """Device/runtime normalization and status helpers."""

    def _resolve_initialize_device(self, requested_device: str) -> str:
        """Resolve a concrete runtime device, applying backend fallback rules."""
        device = requested_device
        if device == "auto":
            if gpu_config.is_cuda_available():
                return "cuda"
            if gpu_config.is_mps_available():
                return "mps"
            if gpu_config.is_xpu_available():
                return "xpu"
            return "cpu"

        if device == "cuda" and not gpu_config.is_cuda_available():
            if gpu_config.is_mps_available():
                logger.warning("[initialize_service] CUDA requested but unavailable. Falling back to MPS.")
                return "mps"
            if gpu_config.is_xpu_available():
                logger.warning("[initialize_service] CUDA requested but unavailable. Falling back to XPU.")
                return "xpu"
            logger.warning("[initialize_service] CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"

        if device == "mps" and not gpu_config.is_mps_available():
            if gpu_config.is_cuda_available():
                logger.warning("[initialize_service] MPS requested but unavailable. Falling back to CUDA.")
                return "cuda"
            if gpu_config.is_xpu_available():
                logger.warning("[initialize_service] MPS requested but unavailable. Falling back to XPU.")
                return "xpu"
            logger.warning("[initialize_service] MPS requested but unavailable. Falling back to CPU.")
            return "cpu"

        if device == "xpu" and not gpu_config.is_xpu_available():
            if gpu_config.is_cuda_available():
                logger.warning("[initialize_service] XPU requested but unavailable. Falling back to CUDA.")
                return "cuda"
            if gpu_config.is_mps_available():
                logger.warning("[initialize_service] XPU requested but unavailable. Falling back to MPS.")
                return "mps"
            logger.warning("[initialize_service] XPU requested but unavailable. Falling back to CPU.")
            return "cpu"

        return device

    def _configure_initialize_runtime(
        self,
        *,
        device: str,
        compile_model: bool,
        quantization: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        """Apply backend constraints and return normalized compile/quantization settings."""
        normalized_compile = compile_model
        normalized_quantization = quantization

        if device == "mps":
            if normalized_compile:
                logger.warning("[initialize_service] torch.compile is disabled on MPS.")
                normalized_compile = False
            if normalized_quantization is not None:
                logger.warning("[initialize_service] Quantization (torchao) is not supported on MPS; disabling.")
                normalized_quantization = None

        return normalized_compile, normalized_quantization

    @staticmethod
    def _ensure_len_for_compile(model: Any, method_name: str) -> None:
        """Inject a fallback ``__len__`` implementation for torch.compile introspection.

        Args:
            model: Model instance whose class may need a ``__len__`` shim.
            method_name: Label used in debug logs to identify the target model.
        """
        if hasattr(model.__class__, "__len__"):
            return

        def _len_impl(_model_self):
            """Return a neutral length for torch.compile compatibility."""
            return 0

        model.__class__.__len__ = _len_impl
        logger.debug(f"[initialize_service] Injected __len__ into {method_name} class for torch.compile")

    def _validate_quantization_setup(self, *, quantization: Optional[str], compile_model: bool) -> None:
        """Validate quantization prerequisites before model loading."""
        if quantization is None:
            return
        try:
            import torchao  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "torchao is required for quantization but is unavailable or incompatible "
                "with this PyTorch build. Please install a compatible torchao version."
            ) from exc

    @staticmethod
    def _build_initialize_status_message(
        *,
        device: str,
        model_path: str,
        vae_path: str,
        text_encoder_path: str,
        dtype: torch.dtype,
        attention: str,
        compile_model: bool,
        offload_to_cpu: bool,
        offload_dit_to_cpu: bool,
        quantization: Optional[str],
    ) -> str:
        """Format initialize_service status output for UI/API consumers."""
        status_msg = f"[OK] Model initialized successfully on {device}\n"
        status_msg += f"Main model: {model_path}\n"
        status_msg += f"VAE: {vae_path}\n"
        status_msg += f"Text encoder: {text_encoder_path}\n"
        status_msg += f"Dtype: {dtype}\n"
        status_msg += f"Attention: {attention}\n"
        compiled_label = str(compile_model)
        status_msg += f"Compiled: {compiled_label}\n"
        status_msg += f"Quantization: {quantization or 'Disabled'}\n"
        status_msg += f"Offload to CPU: {offload_to_cpu}\n"
        status_msg += f"Offload DiT to CPU: {offload_dit_to_cpu}\n"
        return status_msg

