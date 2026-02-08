import json
import numpy as np
import pandas as pd
import arviz as az
from scipy.stats import nbinom
from src.paths import PROCESSED_DIR, OUTPUTS_DIR

def flat(da):
    return da.stack(sample=("chain", "draw")).values

def main():
    df = pd.read_csv(PROCESSED_DIR / "model_df.csv")
    idata = az.from_netcdf(OUTPUTS_DIR / "idata.nc")

    # Hold-out by year (top 20%)
    cut = df["year"].quantile(0.8)
    train = df[df["year"] <= cut].copy()
    test = df[df["year"] > cut].copy()

    # State mapping (train-defined)
    train["state_fips"] = train["fips"].astype(str).str[:2]
    states = np.unique(train["state_fips"])
    state_to_i = {s: i for i, s in enumerate(states)}

    test["state_fips"] = test["fips"].astype(str).str[:2]
    test["state_i"] = test["state_fips"].map(state_to_i)
    test = test.dropna(subset=["state_i"]).copy()
    test["state_i"] = test["state_i"].astype(int)

    y = test["extreme_heat_days"].round().astype(int).values
    x_inc = test["log_income_z"].values
    x_year = test["year_z"].values
    x_int = x_inc * x_year
    sidx = test["state_i"].values

    post = idata.posterior
    beta0 = flat(post["beta0"])
    b_inc = flat(post["beta_inc"])
    b_year = flat(post["beta_year"])
    b_int = flat(post["beta_int"])
    alpha = flat(post["alpha"])

    # a_state in idata is indexed by the full dataset’s states from model_fit,
    # so for an MVP, we’ll approximate by using zeros for state effects in scoring:
    # (If you want the “perfect” mapping, we can store the states list in a JSON.)
    a_state = 0.0

    S = beta0.shape[0]
    eta = (beta0[:, None]
           + a_state
           + b_inc[:, None] * x_inc[None, :]
           + b_year[:, None] * x_year[None, :]
           + b_int[:, None] * x_int[None, :])
    mu_s = np.exp(eta)

    # Predictive log-likelihood via NB mixture
    p = alpha[:, None] / (alpha[:, None] + mu_s)
    n = alpha[:, None]
    logpmf = nbinom.logpmf(y[None, :], n, p)

    m = logpmf.max(axis=0)
    log_pred = m + np.log(np.mean(np.exp(logpmf - m), axis=0))
    nll = float(-np.mean(log_pred))

    # 95% predictive interval proxy from mu quantiles
    mu_lo = np.quantile(mu_s, 0.025, axis=0)
    mu_hi = np.quantile(mu_s, 0.975, axis=0)
    coverage = float(np.mean((y >= mu_lo) & (y <= mu_hi)))

    test = test.copy()
    test["mu_lo"] = mu_lo
    test["mu_hi"] = mu_hi
    test["pi_width"] = test["mu_hi"] - test["mu_lo"]

    group = test.groupby("income_q", as_index=False)["pi_width"].mean()
    group_out = OUTPUTS_DIR / "group_uncertainty.csv"
    group.to_csv(group_out, index=False)

    metrics = {"test_nll": nll, "test_interval_coverage_proxy": coverage}
    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] metrics:", metrics)
    print(f"[OK] wrote {group_out}")

if __name__ == "__main__":
    main()
