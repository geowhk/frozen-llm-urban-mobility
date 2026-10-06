"""CLI import regression: do not shadow the installed/source package."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

class EntryPoints(unittest.TestCase):
    def test_source_launcher_and_module(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
        for args in ([str(root / "run_paper1.py"), "--help"], ["-m", "paper1", "--help"]):
            result = subprocess.run([sys.executable, *args], cwd=root, env=env,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("verify", result.stdout)
