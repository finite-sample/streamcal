"""
Generate figures and tables for the paper.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "ms" / "figures"
TABLES_DIR = Path(__file__).parent.parent / "ms" / "tables"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

METHOD_LABELS = {
    "none": "No calibration",
    "ema": "Per-bucket EMA",
    "sgd": "Online SGD",
    "mwu": "MWU",
    "temperature": "Temperature",
    "isotonic": "Isotonic",
    "platt_1": "Platt (k=1)",
    "platt_5": "Platt (k=5)",
    "platt_10": "Platt (k=10)",
    "mwu_diminishing": "MWU (dim. LR)",
    "sgd_diminishing": "SGD (dim. LR)",
    "mwu_constant": "MWU (const. LR)",
}

METHOD_COLORS = {
    "none": "gray",
    "ema": "C4",
    "sgd": "C1",
    "mwu": "C2",
    "temperature": "C5",
    "isotonic": "C3",
    "platt_1": "C0",
    "platt_5": "C6",
    "platt_10": "C7",
    "mwu_diminishing": "C2",
    "sgd_diminishing": "C1",
    "mwu_constant": "C8",
}


def load_results(filename: str) -> dict:
    """Load results from JSON file."""
    filepath = RESULTS_DIR / filename
    with open(filepath, "r") as f:
        return json.load(f)


def aggregate_results(results_list: list, method: str, metric: str) -> tuple:
    """Aggregate results across seeds."""
    values = []
    for seed_result in results_list:
        if isinstance(seed_result, dict) and "results" in seed_result:
            method_results = seed_result["results"][method]
        else:
            method_results = seed_result[method]
        metric_values = [r[metric] for r in method_results]
        values.append(metric_values)

    values = np.array(values)
    mean = np.mean(values, axis=0)
    ci = 1.96 * np.std(values, axis=0, ddof=1) / np.sqrt(len(values))
    return mean, ci


def get_final_metrics(results_list: list, method: str) -> dict:
    """Get final batch metrics across seeds."""
    final_brier = []
    final_ece = []
    mean_cpu = []

    for seed_result in results_list:
        if isinstance(seed_result, dict) and "results" in seed_result:
            method_results = seed_result["results"][method]
        else:
            method_results = seed_result[method]

        final_brier.append(method_results[-1]["brier"])
        final_ece.append(method_results[-1]["ece"])
        mean_cpu.append(np.mean([r["cpu_ms"] for r in method_results]))

    n = len(final_brier)
    return {
        "brier_mean": np.mean(final_brier),
        "brier_ci": 1.96 * np.std(final_brier) / np.sqrt(n),
        "ece_mean": np.mean(final_ece),
        "ece_ci": 1.96 * np.std(final_ece) / np.sqrt(n),
        "cpu_ms_mean": np.mean(mean_cpu),
        "cpu_ms_ci": 1.96 * np.std(mean_cpu) / np.sqrt(n),
    }


def generate_figure1_convergence():
    """
    Generate Figure 1: Log-log convergence plot.
    X: batch number (log scale)
    Y: per-bucket MSE (log scale)
    """
    print("Generating Figure 1 (convergence)...")

    results = load_results("convergence_results.json")

    fig, ax = plt.subplots(figsize=(6, 4))

    for method in ["mwu_diminishing", "sgd_diminishing"]:
        mean, ci = aggregate_results(results, method, "bucket_mse")
        batches = np.arange(1, len(mean) + 1)
        ax.loglog(
            batches, mean,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "black"),
            linewidth=2,
        )
        ax.fill_between(
            batches, mean - ci, mean + ci,
            alpha=0.2, color=METHOD_COLORS.get(method, "black")
        )

    batches = np.arange(1, len(mean) + 1)
    theoretical = 0.05 / batches
    ax.loglog(
        batches, theoretical, "k--", linewidth=1.5, label=r"$O(1/t)$ theoretical"
    )

    ax.set_xlabel("Batch number $t$", fontsize=11)
    ax.set_ylabel("Per-bucket MSE", fontsize=11)
    ax.set_title("Convergence under stationarity", fontsize=12)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    filepath = FIGURES_DIR / "convergence_loglog.pdf"
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {filepath}")


def generate_figure2_tracking_eta():
    """
    Generate Figure 2: Tracking error vs step size.
    X: eta (log scale)
    Y: time-averaged calibration error
    """
    print("Generating Figure 2 (tracking vs eta)...")

    results = load_results("eta_sensitivity_results.json")

    fig, ax = plt.subplots(figsize=(6, 4))

    etas = []
    mean_errors = []
    ci_errors = []

    for eta_str, eta_results in results.items():
        eta = float(eta_str)
        etas.append(eta)

        all_avg_errors = []
        for seed_result in eta_results:
            method_results = seed_result["mwu"]
            avg_error = np.mean([r["bucket_mse"] for r in method_results])
            all_avg_errors.append(avg_error)

        mean_errors.append(np.mean(all_avg_errors))
        ci_errors.append(1.96 * np.std(all_avg_errors) / np.sqrt(len(all_avg_errors)))

    etas = np.array(etas)
    mean_errors = np.array(mean_errors)
    ci_errors = np.array(ci_errors)

    sort_idx = np.argsort(etas)
    etas = etas[sort_idx]
    mean_errors = mean_errors[sort_idx]
    ci_errors = ci_errors[sort_idx]

    ax.errorbar(
        etas, mean_errors, yerr=ci_errors,
        marker="o", color="C2", linewidth=2, markersize=6,
        capsize=4, label="MWU empirical"
    )

    min_idx = np.argmin(mean_errors)
    min_eta = etas[min_idx]
    sigma2 = 0.01
    V_T = 0.7
    D = np.sqrt(100) * np.log(100)
    mu = 0.1
    T = 40

    eta_theory = np.logspace(-3, 0.5, 100)
    theoretical_bound = (eta_theory * sigma2 / mu + D * V_T / (eta_theory * mu * T))
    theoretical_bound = theoretical_bound / theoretical_bound.min() * mean_errors.min()

    ax.loglog(eta_theory, theoretical_bound, "k--", linewidth=1.5, label="Theoretical U-shape")

    ax.set_xlabel(r"Step size $\eta$", fontsize=11)
    ax.set_ylabel("Time-averaged per-bucket MSE", fontsize=11)
    ax.set_title("Tracking error vs step size (linear drift)", fontsize=12)
    ax.legend(frameon=True, fontsize=9)
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    filepath = FIGURES_DIR / "tracking_eta.pdf"
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {filepath}")


def generate_figure3_recovery():
    """
    Generate Figure 3: ECE time series with sudden shift.
    X: batch number
    Y: ECE
    Vertical line at T/2 (shift point)
    """
    print("Generating Figure 3 (recovery)...")

    results = load_results("synthetic_results.json")
    sudden_shift_results = results["sudden_shift"]

    fig, ax = plt.subplots(figsize=(7, 4))

    methods_to_plot = ["ema", "sgd", "mwu", "isotonic", "platt_1", "platt_10"]

    for method in methods_to_plot:
        mean, ci = aggregate_results(sudden_shift_results, method, "ece")
        batches = np.arange(1, len(mean) + 1)
        ax.plot(
            batches, mean,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "black"),
            linewidth=1.5,
        )
        ax.fill_between(
            batches, mean - ci, mean + ci,
            alpha=0.15, color=METHOD_COLORS.get(method, "black")
        )

    n_batches = len(mean)
    ax.axvline(x=n_batches / 2, linestyle="--", color="gray", linewidth=2, label="Shift point")

    ax.set_xlabel("Batch number", fontsize=11)
    ax.set_ylabel("ECE", fontsize=11)
    ax.set_title("ECE over time with sudden shift at $t=T/2$", fontsize=12)
    ax.legend(frameon=True, fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = FIGURES_DIR / "recovery.pdf"
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {filepath}")


def generate_figure4_scaling():
    """
    Generate Figure 4: CPU scaling with data volume.
    X: cumulative data volume
    Y: CPU time per batch
    """
    print("Generating Figure 4 (scaling)...")

    results = load_results("production_results.json")

    fig, ax = plt.subplots(figsize=(6, 4))

    methods_to_plot = ["mwu", "ema", "sgd", "isotonic", "platt_1"]

    for method in methods_to_plot:
        all_cpu = []
        for seed_result in results:
            method_results = seed_result[method]
            cpu_times = [r["cpu_ms"] for r in method_results]
            all_cpu.append(cpu_times)

        all_cpu = np.array(all_cpu)
        mean = np.mean(all_cpu, axis=0)
        n_batches = len(mean)
        batch_size = 10_000
        cumulative_volume = np.arange(1, n_batches + 1) * batch_size / 1e6

        ax.plot(
            cumulative_volume, mean,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "black"),
            linewidth=1.5,
        )

    ax.set_xlabel("Cumulative data volume (millions)", fontsize=11)
    ax.set_ylabel("CPU time per batch (ms)", fontsize=11)
    ax.set_title("CPU scaling with data volume", fontsize=12)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = FIGURES_DIR / "scaling.pdf"
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {filepath}")


def generate_table1_main_comparison():
    """
    Generate Table 1: Main method comparison.
    Columns: Method, Brier, ECE, CPU ms/batch, Streaming?
    Data: Semi-synthetic, linear drift, B=50
    """
    print("Generating Table 1 (main comparison)...")

    results = load_results("semi_synthetic_results.json")
    combined_results = []
    for domain in ["ad_ctr", "income", "loan"]:
        combined_results.extend(results[domain])

    methods = [
        "none", "ema", "sgd", "mwu", "temperature",
        "platt_1", "platt_5", "platt_10", "isotonic"
    ]
    streaming = {
        "none": "---",
        "ema": "Yes",
        "sgd": "Yes",
        "mwu": "Yes",
        "temperature": "No",
        "isotonic": "No",
        "platt_1": "No",
        "platt_5": "No",
        "platt_10": "No",
    }

    rows = []
    for method in methods:
        metrics = get_final_metrics(combined_results, method)
        rows.append({
            "method": METHOD_LABELS.get(method, method),
            "brier": f"{metrics['brier_mean']:.4f}",
            "ece": f"{metrics['ece_mean']:.4f}",
            "cpu_ms": f"{metrics['cpu_ms_mean']:.2f}",
            "streaming": streaming[method],
        })

    latex = r"""\begin{table}[H]
\centering
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & Brier & ECE & CPU ms/batch & Streaming?\\
\midrule
"""
    for row in rows:
        latex += f"{row['method']} & {row['brier']} & {row['ece']} & {row['cpu_ms']} & {row['streaming']}\\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\caption{Main comparison on semi-synthetic streams (LightGBM base model, linear drift, $B=50$). Results averaged across three domains (ad CTR, income, loan default) and 20 seeds.}
\label{tab:main_comparison}
\end{table}
"""

    filepath = TABLES_DIR / "main_comparison.tex"
    with open(filepath, "w") as f:
        f.write(latex)
    print(f"  Saved to {filepath}")


def generate_table2_drift_scenarios():
    """
    Generate Table 2: Drift scenario comparison.
    Columns: Scenario, Platt, Isotonic, MWU (+ others)
    Rows: 6 drift patterns
    """
    print("Generating Table 2 (drift scenarios)...")

    results = load_results("synthetic_results.json")

    scenarios = ["stationary", "linear_up", "linear_down", "sinusoidal", "sudden_shift", "random_walk"]
    methods = ["platt_1", "isotonic", "mwu", "ema", "sgd"]

    latex = r"""\begin{table}[H]
\centering
\begin{tabular}{@{}lccccc@{}}
\toprule
Scenario & Platt & Isotonic & MWU & EMA & SGD\\
\midrule
"""

    for scenario in scenarios:
        scenario_results = results[scenario]
        row_parts = [scenario.replace("_", " ").title()]

        for method in methods:
            metrics = get_final_metrics(scenario_results, method)
            row_parts.append(f"{metrics['ece_mean']:.4f}")

        latex += " & ".join(row_parts) + "\\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\caption{Final ECE by drift scenario (synthetic stream, $B=100$, 20 seeds). Lower is better.}
\label{tab:drift_scenarios}
\end{table}
"""

    filepath = TABLES_DIR / "drift_scenarios.tex"
    with open(filepath, "w") as f:
        f.write(latex)
    print(f"  Saved to {filepath}")


def generate_all():
    """Generate all figures and tables."""
    print("=" * 60)
    print("Generating Figures and Tables")
    print("=" * 60)

    generate_figure1_convergence()
    generate_figure2_tracking_eta()
    generate_figure3_recovery()
    generate_figure4_scaling()
    generate_table1_main_comparison()
    generate_table2_drift_scenarios()

    print("\n" + "=" * 60)
    print("All figures and tables generated!")
    print("=" * 60)


if __name__ == "__main__":
    generate_all()
