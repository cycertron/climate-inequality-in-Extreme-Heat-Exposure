import pandas as pd
import matplotlib.pyplot as plt
from src.paths import OUTPUTS_DIR, FIGS_DIR

def main():
    group = pd.read_csv(OUTPUTS_DIR / "group_uncertainty.csv")
    plt.figure()
    plt.bar(group["income_q"].astype(str), group["pi_width"])
    plt.xlabel("income quintile (0=lowest, 4=highest)")
    plt.ylabel("avg 95% interval width (proxy)")
    plt.title("Predictive uncertainty by income group")
    out = FIGS_DIR / "interval_width_by_income_q.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[OK] saved {out}")

if __name__ == "__main__":
    main()
