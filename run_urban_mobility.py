#!/usr/bin/env python3
"""Source-tree launcher; named differently from the urban_mobility package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from urban_mobility.cli import main

if __name__ == "__main__":
    main()
