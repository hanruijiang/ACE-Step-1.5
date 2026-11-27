"""Tests for cached repaint-source latent loading."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from acestep.inference import (
    _load_cached_repaint_source,
    _load_cached_repaint_source_latents,
    _resample_matching_source_seeds,
)


class RepaintCacheLoadTests(unittest.TestCase):
    """Verify repaint source latent sidecar loading."""

    def test_missing_sidecar_returns_none(self):
        """Uploaded audio without a generated sidecar should use the normal path."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_load_cached_repaint_source_latents(str(Path(tmp) / "upload.wav")))

    def test_loads_relative_latent_file_from_sidecar(self):
        """Generated sidecars should resolve relative cached latent paths."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "sample.wav"
            audio_path.write_bytes(b"")
            np.save(root / "sample.repaint_latents.npy", np.ones((4, 3), dtype=np.float32))
            (root / "sample.json").write_text(
                json.dumps({"repaint_source_latents_file": "sample.repaint_latents.npy"}),
                encoding="utf-8",
            )

            latents = _load_cached_repaint_source_latents(str(audio_path))

            self.assertIsNotNone(latents)
            self.assertEqual((4, 3), tuple(latents.shape))

    def test_loads_source_seed_from_sidecar(self):
        """Generated-source cache should expose the original generation seed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "sample.wav"
            audio_path.write_bytes(b"")
            np.save(root / "sample.repaint_latents.npy", np.ones((4, 3), dtype=np.float32))
            (root / "sample.json").write_text(
                json.dumps({
                    "repaint_source_latents_file": "sample.repaint_latents.npy",
                    "seed": 123,
                }),
                encoding="utf-8",
            )

            cached_source = _load_cached_repaint_source(str(audio_path))

            self.assertIsNotNone(cached_source)
            self.assertEqual(123, cached_source.source_seed)

    def test_resample_matching_source_seed_only_changes_collisions(self):
        """Cached-source repaint should avoid reusing the original generation seed."""
        seeds = _resample_matching_source_seeds([1, 2, 3], source_seed=2)

        self.assertEqual(1, seeds[0])
        self.assertEqual(3, seeds[2])
        self.assertNotEqual(2, seeds[1])


if __name__ == "__main__":
    unittest.main()
