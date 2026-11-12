"""Download the freMTPL2 frequency and severity tables from OpenML into data/."""

from pathlib import Path

from sklearn.datasets import fetch_openml

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

for data_id, name in [(41214, "freMTPL2freq"), (41215, "freMTPL2sev")]:
    df = fetch_openml(data_id=data_id, as_frame=True, parser="auto").frame
    df.to_csv(DATA_DIR / f"{name}.csv", index=False)
    print(f"Saved {name}: {df.shape[0]:,} rows")
