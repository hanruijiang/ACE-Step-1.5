"""Audio-code parsing and conversion helpers for handler decomposition."""

import re
from typing import List, Optional

import torch
from loguru import logger


class AudioCodesMixin:
    """Mixin containing audio-code parsing and latent conversion helpers.

    """

    def _parse_audio_code_string(self, code_str: str) -> List[int]:
        """Extract integer audio codes from tokens like ``<|audio_code_123|>``."""
        if not code_str:
            return []
        max_audio_code = 63999
        values = [int(value) for value in re.findall(r"<\|audio_code_(\d+)\|>", code_str)]
        codes = [min(value, max_audio_code) for value in values]
        clamped_count = sum(value > max_audio_code for value in values)
        if clamped_count:
            logger.warning("Clamped {} audio codes to the valid range.", clamped_count)
        return codes

    def _decode_audio_codes_to_latents(self, code_str: str) -> Optional[torch.Tensor]:
        """Convert serialized audio-code string into 25Hz latents."""
        if self.model is None or not hasattr(self.model, "tokenizer") or not hasattr(self.model, "detokenizer"):
            return None

        code_ids = self._parse_audio_code_string(code_str)
        if len(code_ids) == 0:
            return None

        with self._load_model_context("model"):
            quantizer = self.model.tokenizer.quantizer
            detokenizer = self.model.detokenizer
            indices = torch.tensor(code_ids, device=self.device, dtype=torch.long)
            indices = indices.unsqueeze(0).unsqueeze(-1)

            quantized = quantizer.get_output_from_indices(indices)
            if quantized.dtype != self.dtype:
                quantized = quantized.to(self.dtype)
            lm_hints_25hz = detokenizer(quantized)
            return lm_hints_25hz
