from pathlib import Path
import numpy as np
import pandas as pd

def z(x):
    return (x - x.mean()) / x.std()

def main():
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    heat_path = processed / "heat_clean.csv"
    income_path = processed / "income_acs.csv"

    print("[INFO] root:", root)
    print("[INFO] reading:", heat_path)
    print("[INFO] reading:", income_path)

    if not heat_path.exists():
        raise FileNotFoundError(f"Missing {heat_path}")
    if not income_path.exists():
        raise FileNotFoundError(f"Missing {income_path}")

    heat = pd.read_csv(heat_path)
    income = pd.read_csv(income_path)

    print("[INFO] heat rows:", len(heat), "cols:", list(heat.columns))
    print("[INFO] income rows:", len(income), "cols:", list(income.columns))

    # Ensure correct types
    heat["fips"] = heat["fips"].astype(str).str.zfill(5)
    income["fips"] = income["fips"].astype(str).str.zfill(5)

    df = heat.merge(income[["fips", "income"]], on="fips", how="inner").dropna()

    # Features
    df["log_income"] = np.log(df["income"].clip(lower=1000))
    df["year_c"] = df["year"] - df["year"].median()

    df["income_q"] = pd.qcut(df["log_income"], 5, labels=False)  # 0..4
    df["log_income_z"] = z(df["log_income"])
    df["year_z"] = z(df["year_c"])

    out1 = processed / "model_df.csv"
    df.to_csv(out1, index=False)
    print(f"[OK] wrote {out1} rows={len(df)}")

    # Optional: also write model.csv (if you specifically want that name)
    out2 = processed / "model.csv"
    df.to_csv(out2, index=False)
    print(f"[OK] wrote {out2} rows={len(df)}")

if __name__ == "__main__":
    main()
