from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"
FIGS_DIR = OUTPUTS_DIR / "figs"

for p in [RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, FIGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
