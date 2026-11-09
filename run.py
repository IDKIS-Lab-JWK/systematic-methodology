"""Repository command-line entry point (heavy libraries are imported lazily)."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'Code'))
from framework.cli import main
if __name__ == '__main__':
    raise SystemExit(main())
