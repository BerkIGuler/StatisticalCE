import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import yaml


SCENARIOS = [
    "high_delay_high_mobility",
    "high_delay_moderate_mobility",
    "high_delay_low_mobility",
    "moderate_delay_high_mobility",
    "moderate_delay_moderate_mobility",
    "moderate_delay_low_mobility",
    "low_delay_high_mobility",
    "low_delay_moderate_mobility",
    "low_delay_low_mobility"
]


def load_yaml(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)


def plot_ind_dist_results(
    delay_profile: str,
    pilot_type: str,
    results_dir: Path = Path("results"),
    output_path: Optional[Path] = None,
    metric: str = "db",
):
    """
    Plot 3x3 subfigures for all 9 scenarios for a single configuration:
    - delay_profile in {A,...,E}
    - pilot_type in {\"2\", \"23\", \"2711\"}
    - metric: \"db\" (nmse_mean_db) or \"linear\" (nmse_mean_linear)
    Always plots: bilinear, LMMSE(all_test), LMMSE(per_test).
    """
    # Normalize inputs
    delay_profile = delay_profile.upper()
    pilot_suffix = pilot_type.replace(" ", "")
    if metric not in ("db", "linear"):
        raise ValueError("metric must be 'db' or 'linear'")
    value_key = "nmse_mean_db" if metric == "db" else "nmse_mean_linear"
    ylabel = "NMSE [dB]" if metric == "db" else "NMSE (linear)"

    base_dir = results_dir / "in_dist" / "numerical"

    # File naming convention established by evaluate_* scripts
    bilinear_file = base_dir / f"bilinear_interp_{delay_profile}_{pilot_suffix}.yaml"
    lmmse_all_test_file = base_dir / f"lmmse_{delay_profile}_all_test_{pilot_suffix}.yaml"
    lmmse_per_test_file = base_dir / f"lmmse_{delay_profile}_per_test_{pilot_suffix}.yaml"

    bilinear_data = load_yaml(bilinear_file)
    lmmse_all_test_data = load_yaml(lmmse_all_test_file)
    lmmse_per_test_data = load_yaml(lmmse_per_test_file)

    # Keys are SNRs (0,5,10,...) → ensure sorted order
    snrs = sorted(int(s) for s in bilinear_data.keys())

    # Verify all algorithms share the same SNR grid
    snrs_lmmse_all = sorted(int(s) for s in lmmse_all_test_data.keys())
    snrs_lmmse_per = sorted(int(s) for s in lmmse_per_test_data.keys())
    if snrs != snrs_lmmse_all or snrs != snrs_lmmse_per:
        raise ValueError(
            f"SNR grids differ between bilinear ({bilinear_file}) and LMMSE "
            f"({lmmse_all_test_file}, {lmmse_per_test_file})"
        )

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.reshape(-1)

    for idx, scenario in enumerate(SCENARIOS):
        ax = axes[idx]

        # Some scenarios might be absent (defensive)
        if scenario not in next(iter(bilinear_data.values())).keys():
            ax.set_visible(False)
            continue

        y_bilin = [bilinear_data[snr][scenario][value_key] for snr in snrs]
        y_lmmse_all = [
            lmmse_all_test_data[snr][scenario][value_key] for snr in snrs
        ]
        y_lmmse_per = [
            lmmse_per_test_data[snr][scenario][value_key] for snr in snrs
        ]

        ax.plot(snrs, y_bilin, marker="o", label="Bilinear interpolation")
        ax.plot(snrs, y_lmmse_all, marker="s", label="LMMSE (all_test)")
        ax.plot(snrs, y_lmmse_per, marker="^", label="LMMSE (per_test)")

        ax.set_title(scenario.replace("_", " "), fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

        if idx // 3 == 2:
            ax.set_xlabel("SNR [dB]")
        if idx % 3 == 0:
            ax.set_ylabel(ylabel)

        # Per-subplot legend for clarity
        ax.legend(loc="lower left", fontsize=7, frameon=False)

    fig.suptitle(
        f"TDL-{delay_profile}, pilot symbols={pilot_suffix}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.94])

    if output_path is None:
        output_path = (
            results_dir / f"ind_dist_plot_TDL{delay_profile}_pilots{pilot_suffix}_{metric}.png"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot in-distribution NMSE vs SNR for LMMSE and bilinear interpolation "
            "for a given delay profile, pilot config, and LMMSE stats type."
        )
    )
    parser.add_argument("--delay_profile", type=str, required=True,
                        help="Delay profile letter, e.g. A, B, C, D, E.")
    parser.add_argument(
        "--pilot_type",
        type=str,
        required=True,
        help="Pilot type suffix used in filenames: '2', '23', or '2711'.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Base results directory (default: results).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["db", "linear"],
        default="db",
        help="Plot NMSE in dB (nmse_mean_db) or linear (nmse_mean_linear). Default: db.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional explicit path for the output PNG.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_ind_dist_results(
        delay_profile=args.delay_profile,
        pilot_type=args.pilot_type,
        results_dir=Path(args.results_dir),
        output_path=Path(args.output_path) if args.output_path is not None else None,
        metric=args.metric,
    )

