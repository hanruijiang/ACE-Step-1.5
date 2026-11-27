"""Tests for deterministic UUID generation and audio-code preservation."""
import unittest
from acestep.audio_utils import generate_uuid_from_params


class UuidGenerationTest(unittest.TestCase):
    def test_same_params_produce_same_uuid(self):
        """Same params should always produce the same UUID (deterministic)."""
        params = {
            "seed": 3631605787,
            "caption": "A beautiful melody",
        }

        uuid1 = generate_uuid_from_params(params)
        uuid2 = generate_uuid_from_params(params)

        # UUIDs should be identical
        self.assertEqual(uuid1, uuid2, "Same params should produce same UUID")

    def test_uuid_format_is_valid(self):
        """Generated UUID should follow standard format."""
        params = {
            "seed": 3631605787,
        }

        uuid = generate_uuid_from_params(params)

        # Check format: 8-4-4-4-12
        parts = uuid.split("-")
        self.assertEqual(len(parts), 5, "UUID should have 5 parts")
        self.assertEqual(len(parts[0]), 8, "First part should be 8 chars")
        self.assertEqual(len(parts[1]), 4, "Second part should be 4 chars")
        self.assertEqual(len(parts[2]), 4, "Third part should be 4 chars")
        self.assertEqual(len(parts[3]), 4, "Fourth part should be 4 chars")
        self.assertEqual(len(parts[4]), 12, "Fifth part should be 12 chars")



class AudioCodesPreservationTest(unittest.TestCase):
    """Test that user-provided audio_codes are not overwritten by empty LM codes.

    Reproduces the logic from inference.py lines 654-658 where
    lm_generated_audio_codes_list may contain empty strings (from
    infer_type='dit') that would incorrectly overwrite user-provided codes.
    """

    @staticmethod
    def _build_audio_params(base_params, lm_codes_list, idx):
        """Replicate the audio_params building logic from inference.py."""
        audio_params = base_params.copy()
        if lm_codes_list and idx < len(lm_codes_list):
            lm_code = lm_codes_list[idx]
            if lm_code and str(lm_code).strip():
                audio_params["audio_codes"] = lm_code
        return audio_params

    def test_user_codes_preserved_when_lm_returns_empty(self):
        """User-provided audio_codes must survive when LM returns empty codes."""
        user_codes = "<|audio_code_1|><|audio_code_2|>"
        base = {"audio_codes": user_codes, "seed": 42}
        # LM ran in "dit" mode → empty string in list
        result = self._build_audio_params(base, [""], 0)
        self.assertEqual(result["audio_codes"], user_codes)

    def test_lm_codes_overwrite_when_non_empty(self):
        """LM-generated codes should overwrite when non-empty."""
        lm_codes = "<|audio_code_99|>"
        base = {"audio_codes": "", "seed": 42}
        result = self._build_audio_params(base, [lm_codes], 0)
        self.assertEqual(result["audio_codes"], lm_codes)

    def test_user_codes_preserved_when_lm_list_empty(self):
        """User codes preserved when lm_generated list is empty."""
        user_codes = "<|audio_code_1|>"
        base = {"audio_codes": user_codes, "seed": 42}
        result = self._build_audio_params(base, [], 0)
        self.assertEqual(result["audio_codes"], user_codes)


if __name__ == "__main__":
    unittest.main()
