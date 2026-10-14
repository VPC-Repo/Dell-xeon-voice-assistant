import sys
from pathlib import Path

# Make the gateway package importable without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
