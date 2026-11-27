"""Load and cache wavelet transforms for DCW."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from loguru import logger


class _LazyWavelet:
    """Lazy loader for ``pytorch_wavelets`` DWT1D modules.

    We cache one ``DWT1DForward`` / ``DWT1DInverse`` pair per
    ``(device, dtype, wavelet)`` triple so repeated sampler steps don't
    keep rebuilding the filter banks.
    """

    def __init__(self) -> None:
        self._cache: dict = {}
        self._import_failed = False

    def _try_import(self):
        if self._import_failed:
            return None
        try:
            from pytorch_wavelets import DWT1DForward, DWT1DInverse
        except ImportError:
            self._import_failed = True
            logger.warning(
                "DCW is enabled but 'pytorch_wavelets' is not installed. "
                "Install with `pip install pytorch_wavelets PyWavelets` to "
                "use Differential Correction in Wavelet domain. Falling "
                "back to no-op for this generation."
            )
            return None
        return DWT1DForward, DWT1DInverse

    def get(
        self,
        device: torch.device,
        dtype: torch.dtype,
        wavelet: str,
    ) -> Optional[Tuple["torch.nn.Module", "torch.nn.Module"]]:
        """Return ``(dwt, iwt)`` for the requested device/dtype/wavelet.

        Returns ``None`` when ``pytorch_wavelets`` is missing — callers
        must treat this as a "skip the correction" signal.
        """
        modules = self._try_import()
        if modules is None:
            return None
        DWT1DForward, DWT1DInverse = modules
        key = (str(device), str(dtype), wavelet)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        # DCW's math is numerically sensitive; always run the DWT in fp32 on
        # the latent's device and cast results back to the caller's dtype.
        dwt = DWT1DForward(J=1, mode="zero", wave=wavelet).to(device=device, dtype=torch.float32)
        iwt = DWT1DInverse(mode="zero", wave=wavelet).to(device=device, dtype=torch.float32)
        self._cache[key] = (dwt, iwt)
        return dwt, iwt


# Module-level singleton — one cache per process is fine.
WAVELET_CACHE = _LazyWavelet()
