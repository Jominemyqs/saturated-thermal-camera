from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "by_experiment"
SPDE_OUTPUT = OUTPUT_ROOT / "19_stochastic_spde_ablation"
PROPAGATION_OUTPUT = OUTPUT_ROOT / "20_physics_mean_propagation"
ADVECTION_OUTPUT = OUTPUT_ROOT / "24_posterior_physics_mean_advection"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "25_broad_model_comparison"

METHODS = [
    "ambient mean + RBF",
    "ambient mean + stochastic ST",
    "legacy latent-clipped physics mean + RBF",
    "legacy latent-clipped physics mean + stochastic ST",
    "legacy latent-clipped physics mean + advective stochastic ST",
    "observed-clipped physics mean + RBF",
    "posterior physics mean + RBF",
    "advective posterior physics mean + RBF",
    "full-posterior propagation + RBF",
]

PRIMARY_METHODS = [
    "observed-clipped physics mean + RBF",
    "posterior physics mean + RBF",
    "advective posterior physics mean + RBF",
    "legacy latent-clipped physics mean + stochastic ST",
    "legacy latent-clipped physics mean + advective stochastic ST",
]

METHOD_METADATA = {
    "ambient mean + RBF": {
        "source_experiment": "19_stochastic_spde_ablation",
        "mean_information": "ambient only",
        "residual_model": "current-only RBF",
        "previous_frame_in_residual": False,
        "presentation_group": "ambient baseline",
    },
    "ambient mean + stochastic ST": {
        "source_experiment": "19_stochastic_spde_ablation",
        "mean_information": "ambient only",
        "residual_model": "joint two-frame stationary stochastic heat covariance",
        "previous_frame_in_residual": True,
        "presentation_group": "ambient baseline",
    },
    "legacy latent-clipped physics mean + RBF": {
        "source_experiment": "19_stochastic_spde_ablation",
        "mean_information": "noise-free simulation field clipped at c",
        "residual_model": "current-only RBF",
        "previous_frame_in_residual": False,
        "presentation_group": "legacy latent-clipped",
    },
    "legacy latent-clipped physics mean + stochastic ST": {
        "source_experiment": "19_stochastic_spde_ablation",
        "mean_information": "noise-free simulation field clipped at c",
        "residual_model": "joint two-frame stationary stochastic heat covariance",
        "previous_frame_in_residual": True,
        "presentation_group": "legacy latent-clipped",
    },
    "legacy latent-clipped physics mean + advective stochastic ST": {
        "source_experiment": "19_stochastic_spde_ablation",
        "mean_information": "noise-free simulation field clipped at c",
        "residual_model": "joint two-frame advective stationary stochastic heat covariance",
        "previous_frame_in_residual": True,
        "presentation_group": "legacy latent-clipped",
    },
    "observed-clipped physics mean + RBF": {
        "source_experiment": "20_physics_mean_propagation",
        "mean_information": "noisy camera observation clipped at c",
        "residual_model": "current-only RBF",
        "previous_frame_in_residual": False,
        "presentation_group": "camera-realistic",
    },
    "posterior physics mean + RBF": {
        "source_experiment": "24_posterior_physics_mean_advection",
        "mean_information": "E[T_(n-1) | Y_(n-1)]",
        "residual_model": "current-only RBF",
        "previous_frame_in_residual": False,
        "presentation_group": "camera-realistic",
    },
    "advective posterior physics mean + RBF": {
        "source_experiment": "24_posterior_physics_mean_advection",
        "mean_information": "advected E[T_(n-1) | Y_(n-1)]",
        "residual_model": "current-only RBF",
        "previous_frame_in_residual": False,
        "presentation_group": "camera-realistic",
    },
    "full-posterior propagation + RBF": {
        "source_experiment": "20_physics_mean_propagation",
        "mean_information": "E[T_(n-1) | Y_(n-1)] and propagated covariance",
        "residual_model": "current-only RBF plus propagated low-rank covariance",
        "previous_frame_in_residual": False,
        "presentation_group": "camera-realistic",
    },
}

SOURCE_METHODS = {
    SPDE_OUTPUT: {
        "regular mean + RBF": "ambient mean + RBF",
        "regular mean + stochastic space-time": "ambient mean + stochastic ST",
        "physics mean + RBF": "legacy latent-clipped physics mean + RBF",
        "physics mean + stochastic space-time": (
            "legacy latent-clipped physics mean + stochastic ST"
        ),
        "physics mean + advective stochastic space-time": (
            "legacy latent-clipped physics mean + advective stochastic ST"
        ),
    },
    PROPAGATION_OUTPUT: {
        "clipped propagation": "observed-clipped physics mean + RBF",
        "full-posterior propagation": "full-posterior propagation + RBF",
    },
    ADVECTION_OUTPUT: {
        "posterior physics mean + RBF": "posterior physics mean + RBF",
        "advective posterior physics mean + RBF": (
            "advective posterior physics mean + RBF"
        ),
    },
}

METRICS = [
    "excess_field_rel_l2",
    "mean_crps_K",
    "fixed_top_01_crps_K",
    "peak_absolute_error_K",
    "hot_region_95_coverage",
    "hot_region_95_interval_width_K",
]

METRIC_LABELS = {
    "excess_field_rel_l2": "Relative excess-field L2 error",
    "mean_crps_K": "All-domain CRPS (K)",
    "fixed_top_01_crps_K": "Fixed top-1% CRPS (K)",
    "peak_absolute_error_K": "Peak absolute error (K)",
    "hot_region_95_coverage": "Hot-region 95% coverage",
    "hot_region_95_interval_width_K": "Hot-region 95% width (K)",
}

SHORT_LABELS = {
    "ambient mean + RBF": "ambient + RBF",
    "ambient mean + stochastic ST": "ambient + stochastic ST",
    "legacy latent-clipped physics mean + RBF": "latent-clipped + RBF",
    "legacy latent-clipped physics mean + stochastic ST": (
        "latent-clipped + stochastic ST"
    ),
    "legacy latent-clipped physics mean + advective stochastic ST": (
        "latent-clipped + advective stochastic ST"
    ),
    "observed-clipped physics mean + RBF": "observed-clipped + RBF",
    "posterior physics mean + RBF": "posterior physics mean + RBF",
    "advective posterior physics mean + RBF": "advective posterior mean + RBF",
    "full-posterior propagation + RBF": "full-posterior propagation + RBF",
}

GROUP_COLORS = {
    "ambient baseline": "#7A7A7A",
    "legacy latent-clipped": "#0072B2",
    "camera-realistic": "#009E73",
}

PAIRINGS = [
    (
        "ambient mean + stochastic ST",
        "ambient mean + RBF",
        "stochastic covariance with ambient mean",
    ),
    (
        "legacy latent-clipped physics mean + stochastic ST",
        "legacy latent-clipped physics mean + RBF",
        "stochastic covariance with latent-clipped physics mean",
    ),
    (
        "legacy latent-clipped physics mean + advective stochastic ST",
        "legacy latent-clipped physics mean + stochastic ST",
        "residual advection with latent-clipped physics mean",
    ),
    (
        "posterior physics mean + RBF",
        "observed-clipped physics mean + RBF",
        "previous-frame inequality information",
    ),
    (
        "advective posterior physics mean + RBF",
        "posterior physics mean + RBF",
        "advection in posterior physics mean",
    ),
    (
        "full-posterior propagation + RBF",
        "posterior physics mean + RBF",
        "propagated previous-frame covariance",
    ),
]


def load_results() -> pd.DataFrame:
    frames = []
    for source, mapping in SOURCE_METHODS.items():
        results = pd.read_csv(source / "results.csv")
        selected = results[results["method"].isin(mapping)].copy()
        selected["method"] = selected["method"].map(mapping)
        frames.append(selected)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if set(combined["method"]) != set(METHODS):
        missing = sorted(set(METHODS) - set(combined["method"]))
        raise AssertionError(f"Broad comparison is missing methods: {missing}")
    duplicated = combined.duplicated(["trajectory", "method"])
    if duplicated.any():
        raise AssertionError("A trajectory-method pair appears more than once")
    for method, metadata in METHOD_METADATA.items():
        mask = combined["method"] == method
        for key, value in metadata.items():
            combined.loc[mask, key] = value
    return combined


def verify_compatibility(results: pd.DataFrame) -> pd.DataFrame:
    config_paths = {
        "stochastic_spde": SPDE_OUTPUT / "fixed_configuration.csv",
        "propagation": PROPAGATION_OUTPUT / "fixed_configuration.csv",
        "posterior_mean_advection": ADVECTION_OUTPUT / "fixed_configuration.csv",
    }
    configs = {name: pd.read_csv(path).iloc[0] for name, path in config_paths.items()}
    checks = {
        "fraction_saturated": [float(row["fraction_saturated"]) for row in configs.values()],
        "observation_stride": [int(row["observation_stride"]) for row in configs.values()],
        "seed": [int(row["seed"]) for row in configs.values()],
        "diffusivity": [float(row["diffusivity"]) for row in configs.values()],
        "cooling_rate": [float(row["cooling_rate"]) for row in configs.values()],
        "signal_sd": [float(row["signal_sd"]) for row in configs.values()],
        "source_coupling": [float(row["source_coupling"]) for row in configs.values()],
        "length_multiplier": [float(row["length_multiplier"]) for row in configs.values()],
        "crps_estimator": [str(row["crps_estimator"]) for row in configs.values()],
    }
    rows = []
    for parameter, values in checks.items():
        if parameter == "crps_estimator":
            matches = len(set(values)) == 1 and values[0] == "unbiased_M_times_M_minus_1"
        else:
            matches = bool(np.allclose(values, values[0], rtol=0.0, atol=1e-12))
        rows.append(
            {
                "parameter": parameter,
                "values": " | ".join(str(value) for value in values),
                "matches_across_source_experiments": matches,
            }
        )
    heldout = results[results["role"] == "evaluation"]
    method_counts = heldout.groupby("method")["trajectory"].nunique()
    rows.append(
        {
            "parameter": "heldout_trajectory_count",
            "values": " | ".join(str(int(value)) for value in method_counts),
            "matches_across_source_experiments": bool((method_counts == 30).all()),
        }
    )

    old = pd.read_csv(PROPAGATION_OUTPUT / "results.csv")
    old = old[old["method"] == "posterior-mean propagation"].set_index("trajectory")
    new = pd.read_csv(ADVECTION_OUTPUT / "results.csv")
    new = new[new["method"] == "posterior physics mean + RBF"].set_index("trajectory")
    maximum_difference = max(
        float(np.max(np.abs(old.loc[new.index, metric] - new[metric])))
        for metric in METRICS
    )
    rows.append(
        {
            "parameter": "retained_posterior_RBF_reproduction",
            "values": f"maximum metric difference {maximum_difference:.3e}",
            "matches_across_source_experiments": maximum_difference < 1e-10,
        }
    )
    audit = pd.DataFrame(rows)
    if not audit["matches_across_source_experiments"].all():
        raise AssertionError("At least one broad-comparison compatibility check failed")
    return audit


def aggregate(results: pd.DataFrame, *, by_family: bool) -> pd.DataFrame:
    heldout = results[results["role"] == "evaluation"]
    groups = ["method"]
    if by_family:
        groups.insert(0, "family")
    summary = heldout.groupby(groups, sort=False)[METRICS].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def paired_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    heldout = results[results["role"] == "evaluation"]
    indexed = heldout.set_index(["trajectory", "method"])
    rows = []
    for family, names in [
        ("all", sorted(heldout["trajectory"].unique())),
        *[
            (family, sorted(group["trajectory"].unique()))
            for family, group in heldout.groupby("family", sort=False)
        ],
    ]:
        for method, baseline, comparison in PAIRINGS:
            row = {
                "family": family,
                "comparison": comparison,
                "method": method,
                "baseline": baseline,
                "n_trajectories": len(names),
            }
            for metric in METRICS:
                changes = np.asarray(
                    [
                        indexed.loc[(name, method), metric]
                        - indexed.loc[(name, baseline), metric]
                        for name in names
                    ]
                )
                row[f"{metric}_mean_change"] = float(np.mean(changes))
                row[f"{metric}_median_change"] = float(np.median(changes))
                if metric == "hot_region_95_coverage":
                    row[f"{metric}_win_count"] = int(np.sum(changes > 0.0))
                else:
                    row[f"{metric}_win_count"] = int(np.sum(changes < 0.0))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(
    summary: pd.DataFrame,
    out_path: Path,
    *,
    methods: list[str] = METHODS,
    title: str = "Broad thermal-camera GP comparison",
) -> None:
    indexed = summary.set_index("method")
    y = np.arange(len(methods))
    figure, axes = plt.subplots(2, 3, figsize=(18.0, 12.0), constrained_layout=True)
    for axis, metric in zip(axes.ravel(), METRICS):
        means = np.asarray([indexed.loc[method, f"{metric}_mean"] for method in methods])
        errors = np.asarray(
            [
                indexed.loc[method, f"{metric}_std"]
                / np.sqrt(indexed.loc[method, f"{metric}_count"])
                for method in methods
            ]
        )
        colors = [GROUP_COLORS[METHOD_METADATA[method]["presentation_group"]] for method in methods]
        axis.errorbar(
            means,
            y,
            xerr=np.nan_to_num(errors),
            linestyle="none",
            color="#444444",
            capsize=3,
            zorder=1,
        )
        axis.scatter(means, y, c=colors, s=64, edgecolor="white", linewidth=0.7, zorder=2)
        axis.set_yticks(y, [SHORT_LABELS[method] for method in methods])
        axis.invert_yaxis()
        axis.set_xlabel(METRIC_LABELS[metric])
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        if metric == "hot_region_95_coverage":
            axis.axvline(0.95, color="#8C564B", linestyle="--", linewidth=1.0)
    figure.suptitle(
        f"{title}: 30 held-out trajectories at 3% censoring",
        fontsize=17,
    )
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def write_readme(
    output_dir: Path,
    *,
    overall: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    table = overall.set_index("method")
    pair_table = paired[paired["family"] == "all"].set_index("comparison")
    lines = [
        "# Broad model comparison",
        "",
        "This summary places the retained mean and covariance experiments in one table. "
        "All rows use 3% censoring, the same 30 held-out trajectories, development-only "
        "physical calibration, seed 41, and unbiased M(M-1) CRPS.",
        "",
        "The table is descriptive rather than a single-factor ablation. Current-only RBF "
        "models update on Y_n after constructing their mean. Stochastic space-time rows "
        "use a joint two-frame residual GP and the legacy noise-free simulation field "
        "clipped at c. They therefore use a different information architecture and should "
        "not be declared universally superior from raw cross-row scores alone.",
        "",
        "## Held-out results",
        "",
        "| Method | Field | All CRPS | Top-1% CRPS | Peak error | Hot coverage | Hot width |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = table.loc[method]
        lines.append(
            f"| {method} | {row['excess_field_rel_l2_mean']:.3f} | "
            f"{row['mean_crps_K_mean']:.3f} | "
            f"{row['fixed_top_01_crps_K_mean']:.3f} | "
            f"{row['peak_absolute_error_K_mean']:.3f} | "
            f"{row['hot_region_95_coverage_mean']:.3f} | "
            f"{row['hot_region_95_interval_width_K_mean']:.3f} |"
        )

    inequality = pair_table.loc["previous-frame inequality information"]
    mean_advection = pair_table.loc["advection in posterior physics mean"]
    stochastic = pair_table.loc[
        "stochastic covariance with latent-clipped physics mean"
    ]
    residual_advection = pair_table.loc[
        "residual advection with latent-clipped physics mean"
    ]
    lines.extend(
        [
            "",
            "## Main comparisons",
            "",
            "- Replacing the observed-clipped physics mean by the posterior physics mean "
            f"changes field error by {inequality['excess_field_rel_l2_mean_change']:+.4f} "
            f"and top-1% CRPS by {inequality['fixed_top_01_crps_K_mean_change']:+.4f} K.",
            "- Adding advection to the posterior physics mean changes field error by "
            f"{mean_advection['excess_field_rel_l2_mean_change']:+.4f} and top-1% CRPS "
            f"by {mean_advection['fixed_top_01_crps_K_mean_change']:+.4f} K.",
            "- In the matched legacy mean ablation, stochastic ST versus RBF changes "
            f"all-domain CRPS by {stochastic['mean_crps_K_mean_change']:+.4f} K and "
            f"field error by {stochastic['excess_field_rel_l2_mean_change']:+.4f}.",
            "- In the matched stochastic-ST ablation, residual advection changes "
            f"all-domain CRPS by {residual_advection['mean_crps_K_mean_change']:+.4f} K "
            f"and field error by {residual_advection['excess_field_rel_l2_mean_change']:+.4f}.",
            "",
            "## Interpretation",
            "",
            "The posterior physics mean remains the clean primary architecture for the "
            "camera problem. It improves the hidden hot tail while preserving a strictly "
            "causal current-only update. Mean advection offers a small global-field gain "
            "but slightly worsens top-1% CRPS. The stochastic space-time covariance gives "
            "the lowest probabilistic scores in its legacy two-frame setup, and residual "
            "advection improves it further, but that result should motivate a later clean "
            "camera-realistic temporal model rather than replace the current primary model.",
            "",
            "Files:",
            "",
            "- results.csv: all trajectory-level rows and architecture labels.",
            "- heldout30_overall.csv: overall held-out summaries.",
            "- primary_comparison.csv: the five rows requested for the broad comparison.",
            "- family_summary.csv and paired_comparisons.csv: family and paired results.",
            "- configuration_audit.csv: cross-experiment compatibility checks.",
            "- comparison.png: complete nine-row overview.",
            "- primary_comparison.png: zoomed view of the five requested models.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    audit = verify_compatibility(results)
    overall = aggregate(results, by_family=False)
    family = aggregate(results, by_family=True)
    paired = paired_comparisons(results)
    primary = overall[overall["method"].isin(PRIMARY_METHODS)].copy()
    primary["method"] = pd.Categorical(
        primary["method"], categories=PRIMARY_METHODS, ordered=True
    )
    primary = primary.sort_values("method")

    results.to_csv(DEFAULT_OUTPUT_DIR / "results.csv", index=False)
    overall.to_csv(DEFAULT_OUTPUT_DIR / "heldout30_overall.csv", index=False)
    primary.to_csv(DEFAULT_OUTPUT_DIR / "primary_comparison.csv", index=False)
    family.to_csv(DEFAULT_OUTPUT_DIR / "family_summary.csv", index=False)
    paired.to_csv(DEFAULT_OUTPUT_DIR / "paired_comparisons.csv", index=False)
    audit.to_csv(DEFAULT_OUTPUT_DIR / "configuration_audit.csv", index=False)
    plot_summary(overall, DEFAULT_OUTPUT_DIR / "comparison.png")
    plot_summary(
        overall,
        DEFAULT_OUTPUT_DIR / "primary_comparison.png",
        methods=PRIMARY_METHODS,
        title="Primary thermal-camera GP comparison",
    )
    write_readme(DEFAULT_OUTPUT_DIR, overall=overall, paired=paired)
    print(f"Saved broad comparison to {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
