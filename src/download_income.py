from pathlib import Path
import argparse
import pandas as pd
import requests

def fetch_acs_income(year: int) -> pd.DataFrame:
    # County median household income (ACS 5-year): B19013_001E
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME,B19013_001E",
        "for": "county:*",
        "in": "state:*",
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data[1:], columns=data[0])

    df["income"] = pd.to_numeric(df["B19013_001E"], errors="coerce")
    df["state"] = df["state"].astype(str).str.zfill(2)
    df["county"] = df["county"].astype(str).str.zfill(3)
    df["fips"] = df["state"] + df["county"]

    df = df.dropna(subset=["income"]).copy()
    return df[["fips", "NAME", "income"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]  # project root
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "income_acs.csv"
    df = fetch_acs_income(args.year)
    df.to_csv(out_path, index=False)

    print(f"[OK] wrote {out_path} rows={len(df)}")

if __name__ == "__main__":
    main()
