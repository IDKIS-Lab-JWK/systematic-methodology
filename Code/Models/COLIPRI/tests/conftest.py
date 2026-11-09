from __future__ import annotations

import sys
from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parents[2]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
