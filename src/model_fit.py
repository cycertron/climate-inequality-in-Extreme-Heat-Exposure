import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from src.paths import PROCESSED_DIR, OUTPUTS_DIR

def main():
    df = pd.read_csv(PROCESSED_DIR / "model_df.csv")

    df["state_fips"] = df["fips"].astype(str).str[:2]
    states, state_idx = np.unique(df["state_fips"], return_inverse=True)

    y = df["extreme_heat_days"].round().astype(int).values
    x_inc = df["log_income_z"].values
    x_year = df["year_z"].values
    x_int = x_inc * x_year

    coords = {"obs_id": np.arange(len(df)), "state": states}

    with pm.Model(coords=coords) as model:
        state_i = pm.Data("state_i", state_idx, dims="obs_id")
        X_inc = pm.Data("X_inc", x_inc, dims="obs_id")
        X_year = pm.Data("X_year", x_year, dims="obs_id")
        X_int = pm.Data("X_int", x_int, dims="obs_id")

        beta0 = pm.Normal("beta0", 0, 1.5)
        beta_inc = pm.Normal("beta_inc", 0, 1.0)
        beta_year = pm.Normal("beta_year", 0, 1.0)
        beta_int = pm.Normal("beta_int", 0, 1.0)

        sigma_state = pm.HalfNormal("sigma_state", 1.0)
        a_state = pm.Normal("a_state", 0, sigma_state, dims="state")

        eta = beta0 + a_state[state_i] + beta_inc*X_inc + beta_year*X_year + beta_int*X_int
        mu = pm.Deterministic("mu", pm.math.exp(eta), dims="obs_id")

        alpha = pm.Exponential("alpha", 1.0)
        pm.NegativeBinomial("y_obs", mu=mu, alpha=alpha, observed=y, dims="obs_id")

        idata = pm.sample(1000, tune=1000, chains=2, target_accept=0.9, random_seed=0)

    out = OUTPUTS_DIR / "idata.nc"
    idata.to_netcdf(out)
    print(f"[OK] saved {out}")
    print(az.summary(idata, var_names=["beta0","beta_inc","beta_year","beta_int","sigma_state","alpha"]))

if __name__ == "__main__":
    main()
