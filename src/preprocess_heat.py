import argparse
import pandas as pd
from src.paths import RAW_DIR, PROCESSED_DIR

def guess_col(cols, must_contain_any):
    cols_l = [c.lower() for c in cols]
    for key in must_contain_any:
        for c, cl in zip(cols, cols_l):
            if key in cl:
                return c
    return None

def preprocess_heat(in_path, fips_col=None, year_col=None, value_col=None) -> pd.DataFrame:
    df = pd.read_csv(in_path)

    # Auto-detect if not provided
    if fips_col is None:
        fips_col = guess_col(df.columns, ["fips", "countyfips", "geoid"])
    if year_col is None:
        year_col = guess_col(df.columns, ["year"])
    if value_col is None:
        value_col = guess_col(df.columns, ["value", "data", "measure", "heat", "days"])

    if fips_col is None or year_col is None or value_col is None:
        raise ValueError(
            "Could not auto-detect columns. "
            "Pass --fips_col, --year_col, --value_col explicitly."
        )

    out = df[[fips_col, year_col, value_col]].copy()
    out.columns = ["fips", "year", "extreme_heat_days"]

    out["fips"] = out["fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["extreme_heat_days"] = pd.to_numeric(out["extreme_heat_days"], errors="coerce")

    out = out.dropna(subset=["year", "extreme_heat_days"]).copy()
    out["year"] = out["year"].astype(int)

    # Deduplicate: mean if repeated
    out = out.groupby(["fips", "year"], as_index=False)["extreme_heat_days"].mean()

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=str(RAW_DIR / "cdc_extreme_heat_days.csv"))
    ap.add_argument("--fips_col", default=None)
    ap.add_argument("--year_col", default=None)
    ap.add_argument("--value_col", default=None)
    args = ap.parse_args()

    out_path = PROCESSED_DIR / "heat_clean.csv"
    clean = preprocess_heat(args.infile, args.fips_col, args.year_col, args.value_col)
    clean.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} rows={len(clean)} cols={list(clean.columns)}")

if __name__ == "__main__":
    main()
