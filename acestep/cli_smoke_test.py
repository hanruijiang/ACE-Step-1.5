"""CLI smoke test: imports and help without downloading checkpoints."""
import subprocess
import sys
import unittest


class CliSmokeTest(unittest.TestCase):
    def test_help(self):
        proc = subprocess.run([sys.executable, "-m", "acestep.cli", "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--config", proc.stdout)
        self.assertIn("--configure", proc.stdout)


if __name__ == "__main__":
    unittest.main()
