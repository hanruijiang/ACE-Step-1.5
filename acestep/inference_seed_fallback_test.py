"""Verify that singular and random seeds reach DiT after resolution."""

from __future__ import annotations

import unittest
from typing import Any, Optional
from unittest.mock import MagicMock

from acestep.inference import GenerationConfig, GenerationParams, generate_music

# Path used by the recording handler.
SOURCE_AUDIO = "source.wav"


class RecordingDitHandler:
    """Minimal stand-in that captures the seed kwargs ``generate_music`` forwards."""

    def __init__(self) -> None:
        """Initialize with no captured call; ``generate_kwargs`` stays None until called."""
        self.generate_kwargs: Optional[dict[str, Any]] = None

    def prepare_seeds(self, batch_size: int, seed: str, use_random_seed: bool) -> tuple[list[int], None]:
        """Resolve an explicit seed or a fixed stand-in for a random seed.

        Returns:
            ``(seeds, None)`` — a list of ``batch_size`` identical seeds, and the
            padding info the real handler returns and ``generate_music`` discards.
        """
        return [int(seed) if not use_random_seed and seed not in ("", "-1") else 1234] * batch_size, None

    def generate_music(
        self,
        seed: Optional[str] = None,
        use_random_seed: Optional[bool] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Capture the resolved seed arguments without running a model."""
        self.generate_kwargs = {"seed": seed, "use_random_seed": use_random_seed, **kwargs}
        return {
            "success": False,
            "status_message": "stub handler — stopped after capturing seed kwargs",
            "error": "stub",
            "audios": [],
            "extra_outputs": {},
        }


def _make_llm_handler() -> MagicMock:
    """LLM handler that looks fully initialized, so skipping (via task_type) is a real decision."""
    llm_handler = MagicMock()
    llm_handler.llm_initialized = True
    llm_handler.generate_with_stop_condition.return_value = {
        "success": False,
        "error": "stub LM should not have been reached",
    }
    return llm_handler


class SingularSeedFallbackTests(unittest.TestCase):
    """``params.seed`` must reach the DiT when ``config.seeds`` is unset."""

    def test_bare_seed_reaches_dit_with_random_seed_disabled(self) -> None:
        """config.seeds=None + params.seed=42 + use_random_seed=False must forward seed '42'."""
        dit_handler = RecordingDitHandler()
        llm_handler = _make_llm_handler()
        params = GenerationParams(
            task_type="complete",  # direct-conditioning task: skips the LM, keeps this test focused on seeds
            src_audio=SOURCE_AUDIO,
            caption="warm analog drums and bass",
            lyrics="[Instrumental]",
            seed=42,
        )
        config = GenerationConfig(batch_size=1, seeds=None, use_random_seed=False)

        generate_music(dit_handler, llm_handler, params, config)

        self.assertIsNotNone(dit_handler.generate_kwargs, "DiT handler was never called")
        self.assertEqual("42", dit_handler.generate_kwargs["seed"])
        self.assertFalse(dit_handler.generate_kwargs["use_random_seed"])

    def test_missing_seed_still_uses_random_seed(self) -> None:
        """A random seed is resolved before DiT and forwarded explicitly."""
        dit_handler = RecordingDitHandler()
        llm_handler = _make_llm_handler()
        params = GenerationParams(
            task_type="complete",
            src_audio=SOURCE_AUDIO,
            caption="warm analog drums and bass",
            lyrics="[Instrumental]",
        )
        config = GenerationConfig(batch_size=1, seeds=None, use_random_seed=True)

        generate_music(dit_handler, llm_handler, params, config)

        self.assertIsNotNone(dit_handler.generate_kwargs, "DiT handler was never called")
        self.assertEqual("1234", dit_handler.generate_kwargs["seed"])
        self.assertFalse(dit_handler.generate_kwargs["use_random_seed"])


if __name__ == "__main__":
    unittest.main()
