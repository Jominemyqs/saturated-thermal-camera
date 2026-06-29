from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    for method, group in df.groupby("method"):
        group = group.sort_values("actual_frac_saturated")
        ax.plot(100 * group["actual_frac_saturated"], group[metric], marker="o", label=method)
    ax.set_xlabel("Saturated pixels (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " vs censoring level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "outputs"
    csv_path = out_dir / "sweep_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Run scripts/02_sweep_censoring.py first.")
    df = pd.read_csv(csv_path)

    plot_metric(df, "field_rel_l2", "Relative L2 field error", out_dir / "sweep_field_error.png")
    plot_metric(df, "peak_rel_error", "Relative peak temperature error", out_dir / "sweep_peak_error.png")
    plot_metric(df, "A_rel_error", "Relative amplitude parameter error", out_dir / "sweep_amplitude_error.png")
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
