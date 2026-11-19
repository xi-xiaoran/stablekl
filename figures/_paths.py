from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUT = Path(__file__).resolve().parent / 'out'
OUT.mkdir(exist_ok=True)
