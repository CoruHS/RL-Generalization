#!/usr/bin/env python3
"""
Comprehensive Results Analysis for RL Generalization Research

Generates:
- Summary tables (GRS and Reward)
- Statistical comparisons
- Publication-ready plots
- LaTeX tables for paper

Usage:
    python analyze_results.py                    # Full analysis
    python analyze_results.py --latex            # Generate LaTeX tables
    python analyze_results.py --plots-only       # Only generate plots
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from scipy import stats

# Set up paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

# Environments to include in analysis (exclude incomplete experiments)
INCLUDE_ENVS = ["cartpole", "minigrid", "space_invaders"]


def load_all_results(results_dir: str = RESULTS_DIR, include_envs: List[str] = None) -> List[Dict]:
    """Load all experiment result JSON files."""
    if include_envs is None:
        include_envs = INCLUDE_ENVS

    results = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".json") and not filename.startswith("summary"):
            filepath = os.path.join(results_dir, filename)
            # Skip directories and non-experiment files
            if os.path.isfile(filepath) and "_seed" in filename:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        # Ensure required fields exist and environment is included
                        if "algorithm" in data and "technique" in data and "environment" in data:
                            if data["environment"] in include_envs:
                                results.append(data)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not load {filename}: {e}")
    return results


def aggregate_results(results: List[Dict]) -> Dict[str, Dict]:
    """
    Aggregate results by (algorithm, technique, environment).

    Returns dict with keys like "dqn_baseline_cartpole" containing:
    - grs_values: list of GRS scores across seeds
    - reward_values: list of eval rewards across seeds
    - train_perf_values: list of training performances
    """
    aggregated = defaultdict(lambda: {
        "grs_values": [],
        "reward_values": [],
        "train_perf_values": [],
        "train_times": [],
    })

    for r in results:
        key = f"{r['algorithm']}_{r['technique']}_{r['environment']}"
        aggregated[key]["grs_values"].append(r.get("grs", 0))
        aggregated[key]["reward_values"].append(r.get("eval_reward", 0))
        aggregated[key]["train_perf_values"].append(r.get("train_performance", 0))
        aggregated[key]["train_times"].append(r.get("train_time", 0))
        aggregated[key]["algorithm"] = r["algorithm"]
        aggregated[key]["technique"] = r["technique"]
        aggregated[key]["environment"] = r["environment"]

    return dict(aggregated)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, min, max for a list of values."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "n": 0}
    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "n": len(values),
    }


def print_summary_table(aggregated: Dict, metric: str = "grs"):
    """Print a summary table for a given metric."""
    # Get all unique values
    algos = sorted(set(v["algorithm"] for v in aggregated.values()))
    techniques = sorted(set(v["technique"] for v in aggregated.values()))
    envs = sorted(set(v["environment"] for v in aggregated.values()))

    metric_key = f"{metric}_values"
    metric_name = "GRS" if metric == "grs" else "Eval Reward"

    print(f"\n{'='*80}")
    print(f"{metric_name} SUMMARY (Mean ± Std)")
    print(f"{'='*80}")

    for env in envs:
        print(f"\n{'-'*80}")
        print(f"Environment: {env.upper()}")
        print(f"{'-'*80}")

        # Header
        header = f"{'Algorithm':<10}"
        for tech in techniques:
            header += f" {tech:>15}"
        print(header)
        print("-" * (10 + 16 * len(techniques)))

        for algo in algos:
            row = f"{algo.upper():<10}"
            for tech in techniques:
                key = f"{algo}_{tech}_{env}"
                if key in aggregated:
                    values = aggregated[key][metric_key]
                    stats_dict = compute_statistics(values)
                    row += f" {stats_dict['mean']:>6.2f}±{stats_dict['std']:.2f}"
                else:
                    row += f" {'N/A':>15}"
            print(row)

    print()


def compare_gar_effect(aggregated: Dict) -> Dict[str, Dict]:
    """
    Compare GAR vs baseline for each algorithm/environment.

    Returns improvement statistics.
    """
    comparisons = {}

    algos = sorted(set(v["algorithm"] for v in aggregated.values()))
    envs = sorted(set(v["environment"] for v in aggregated.values()))

    for env in envs:
        for algo in algos:
            baseline_key = f"{algo}_baseline_{env}"
            gar_key = f"{algo}_gar_{env}"

            if baseline_key in aggregated and gar_key in aggregated:
                baseline_grs = aggregated[baseline_key]["grs_values"]
                gar_grs = aggregated[gar_key]["grs_values"]
                baseline_reward = aggregated[baseline_key]["reward_values"]
                gar_reward = aggregated[gar_key]["reward_values"]

                # Statistical test (if enough samples)
                grs_pvalue = None
                reward_pvalue = None
                if len(baseline_grs) >= 2 and len(gar_grs) >= 2:
                    try:
                        _, grs_pvalue = stats.ttest_ind(gar_grs, baseline_grs)
                        _, reward_pvalue = stats.ttest_ind(gar_reward, baseline_reward)
                    except Exception:
                        pass

                comparisons[f"{algo}_{env}"] = {
                    "algo": algo,
                    "env": env,
                    "baseline_grs_mean": np.mean(baseline_grs),
                    "gar_grs_mean": np.mean(gar_grs),
                    "grs_improvement": np.mean(gar_grs) - np.mean(baseline_grs),
                    "grs_pvalue": grs_pvalue,
                    "baseline_reward_mean": np.mean(baseline_reward),
                    "gar_reward_mean": np.mean(gar_reward),
                    "reward_improvement": np.mean(gar_reward) - np.mean(baseline_reward),
                    "reward_pvalue": reward_pvalue,
                }

    return comparisons


def print_gar_comparison(comparisons: Dict):
    """Print GAR vs Baseline comparison table."""
    print(f"\n{'='*100}")
    print("GAR vs BASELINE COMPARISON")
    print(f"{'='*100}")
    print(f"{'Config':<20} {'Baseline GRS':>12} {'GAR GRS':>12} {'GRS Δ':>10} {'Baseline Rew':>12} {'GAR Rew':>12} {'Rew Δ':>10}")
    print("-" * 100)

    for key, data in sorted(comparisons.items()):
        config = f"{data['algo'].upper()}_{data['env']}"
        grs_delta = data['grs_improvement']
        rew_delta = data['reward_improvement']

        # Add significance marker
        grs_sig = "*" if data['grs_pvalue'] and data['grs_pvalue'] < 0.05 else ""
        rew_sig = "*" if data['reward_pvalue'] and data['reward_pvalue'] < 0.05 else ""

        print(f"{config:<20} {data['baseline_grs_mean']:>12.3f} {data['gar_grs_mean']:>12.3f} "
              f"{grs_delta:>+9.3f}{grs_sig} {data['baseline_reward_mean']:>12.2f} "
              f"{data['gar_reward_mean']:>12.2f} {rew_delta:>+9.2f}{rew_sig}")

    print("\n* = statistically significant (p < 0.05)")


def plot_grs_comparison(aggregated: Dict, save_path: str = None):
    """Create bar plot comparing GRS across techniques."""
    algos = sorted(set(v["algorithm"] for v in aggregated.values()))
    techniques = sorted(set(v["technique"] for v in aggregated.values()))
    envs = sorted(set(v["environment"] for v in aggregated.values()))

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    colors = {'baseline': '#1f77b4', 'reg': '#ff7f0e', 'gar': '#2ca02c', 'gar+reg': '#d62728'}

    for ax, env in zip(axes, envs):
        x = np.arange(len(algos))
        width = 0.2

        for i, tech in enumerate(techniques):
            means = []
            stds = []
            for algo in algos:
                key = f"{algo}_{tech}_{env}"
                if key in aggregated:
                    stats_dict = compute_statistics(aggregated[key]["grs_values"])
                    means.append(stats_dict["mean"])
                    stds.append(stats_dict["std"])
                else:
                    means.append(0)
                    stds.append(0)

            ax.bar(x + i * width, means, width, label=tech, yerr=stds,
                   color=colors.get(tech, f'C{i}'), capsize=3, alpha=0.8)

        ax.set_xlabel('Algorithm')
        ax.set_ylabel('GRS Score')
        ax.set_title(f'{env.upper()}')
        ax.set_xticks(x + width * (len(techniques) - 1) / 2)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Generalization Robustness Score (GRS) by Technique', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_reward_comparison(aggregated: Dict, save_path: str = None):
    """Create bar plot comparing eval rewards across techniques."""
    algos = sorted(set(v["algorithm"] for v in aggregated.values()))
    techniques = sorted(set(v["technique"] for v in aggregated.values()))
    envs = sorted(set(v["environment"] for v in aggregated.values()))

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    colors = {'baseline': '#1f77b4', 'reg': '#ff7f0e', 'gar': '#2ca02c', 'gar+reg': '#d62728'}

    for ax, env in zip(axes, envs):
        x = np.arange(len(algos))
        width = 0.2

        for i, tech in enumerate(techniques):
            means = []
            stds = []
            for algo in algos:
                key = f"{algo}_{tech}_{env}"
                if key in aggregated:
                    stats_dict = compute_statistics(aggregated[key]["reward_values"])
                    means.append(stats_dict["mean"])
                    stds.append(stats_dict["std"])
                else:
                    means.append(0)
                    stds.append(0)

            ax.bar(x + i * width, means, width, label=tech, yerr=stds,
                   color=colors.get(tech, f'C{i}'), capsize=3, alpha=0.8)

        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Evaluation Reward')
        ax.set_title(f'{env.upper()}')
        ax.set_xticks(x + width * (len(techniques) - 1) / 2)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Evaluation Reward by Technique', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_grs_curves(results: List[Dict], save_path: str = None):
    """Plot GRS degradation curves for all experiments."""
    envs = sorted(set(r["environment"] for r in results))
    algos = sorted(set(r["algorithm"] for r in results))
    techniques = sorted(set(r["technique"] for r in results))

    fig, axes = plt.subplots(len(envs), len(algos), figsize=(5 * len(algos), 4 * len(envs)))
    if len(envs) == 1:
        axes = [axes]
    if len(algos) == 1:
        axes = [[ax] for ax in axes]

    colors = {'baseline': '#1f77b4', 'reg': '#ff7f0e', 'gar': '#2ca02c', 'gar+reg': '#d62728'}

    for i, env in enumerate(envs):
        for j, algo in enumerate(algos):
            ax = axes[i][j] if len(envs) > 1 else axes[j]

            for tech in techniques:
                matching = [r for r in results
                           if r["environment"] == env and r["algorithm"] == algo and r["technique"] == tech]

                if matching and "grs_curve" in matching[0]:
                    # Average across seeds
                    shift_levels = matching[0]["grs_curve"]["shift_levels"]
                    all_perfs = [r["grs_curve"]["normalized"] for r in matching if "grs_curve" in r]

                    if all_perfs:
                        mean_perf = np.mean(all_perfs, axis=0)
                        std_perf = np.std(all_perfs, axis=0)

                        ax.plot(shift_levels, mean_perf, label=tech, color=colors.get(tech, 'gray'),
                               linewidth=2, marker='o', markersize=4)
                        ax.fill_between(shift_levels, mean_perf - std_perf, mean_perf + std_perf,
                                       color=colors.get(tech, 'gray'), alpha=0.2)

            ax.set_xlabel('Shift Level')
            ax.set_ylabel('Normalized Performance')
            ax.set_title(f'{algo.upper()} on {env}')
            ax.legend(loc='lower left', fontsize=8)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Performance Degradation Under Distribution Shift', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_technique_improvement(comparisons: Dict, save_path: str = None):
    """Plot improvement from GAR over baseline."""
    if not comparisons:
        print("No comparison data available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    configs = list(comparisons.keys())
    grs_improvements = [comparisons[k]["grs_improvement"] for k in configs]
    reward_improvements = [comparisons[k]["reward_improvement"] for k in configs]

    # GRS improvement
    colors1 = ['green' if x > 0 else 'red' for x in grs_improvements]
    bars1 = ax1.barh(configs, grs_improvements, color=colors1, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('GRS Improvement (GAR - Baseline)')
    ax1.set_title('GRS Improvement from GAR')
    ax1.grid(True, alpha=0.3, axis='x')

    # Reward improvement
    colors2 = ['green' if x > 0 else 'red' for x in reward_improvements]
    bars2 = ax2.barh(configs, reward_improvements, color=colors2, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Reward Improvement (GAR - Baseline)')
    ax2.set_title('Reward Improvement from GAR')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Effect of Gradient Agreement Regularization (GAR)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def generate_latex_table(aggregated: Dict, metric: str = "grs") -> str:
    """Generate LaTeX table for the paper."""
    algos = sorted(set(v["algorithm"] for v in aggregated.values()))
    techniques = sorted(set(v["technique"] for v in aggregated.values()))
    envs = sorted(set(v["environment"] for v in aggregated.values()))

    metric_key = f"{metric}_values"

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{" + ("GRS Scores" if metric == "grs" else "Evaluation Rewards") + " (Mean $\\pm$ Std)}")

    # Column format
    col_format = "l" + "c" * len(techniques)
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append("\\toprule")

    for env in envs:
        lines.append(f"\\multicolumn{{{len(techniques)+1}}}{{c}}{{\\textbf{{{env.upper()}}}}} \\\\")
        lines.append("\\midrule")

        # Header
        header = "Algorithm & " + " & ".join(techniques) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        for algo in algos:
            row_parts = [algo.upper()]
            best_val = -np.inf
            best_idx = -1

            # Find best technique for this algo/env
            for i, tech in enumerate(techniques):
                key = f"{algo}_{tech}_{env}"
                if key in aggregated:
                    val = np.mean(aggregated[key][metric_key])
                    if val > best_val:
                        best_val = val
                        best_idx = i

            for i, tech in enumerate(techniques):
                key = f"{algo}_{tech}_{env}"
                if key in aggregated:
                    stats_dict = compute_statistics(aggregated[key][metric_key])
                    val_str = f"{stats_dict['mean']:.3f} $\\pm$ {stats_dict['std']:.2f}"
                    if i == best_idx:
                        val_str = f"\\textbf{{{val_str}}}"
                    row_parts.append(val_str)
                else:
                    row_parts.append("N/A")

            lines.append(" & ".join(row_parts) + " \\\\")

        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def create_full_report(results: List[Dict], aggregated: Dict, comparisons: Dict, output_dir: str):
    """Create a full analysis report."""
    report_path = os.path.join(output_dir, "analysis_report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RL GENERALIZATION RESEARCH - ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total experiments: {len(results)}\n")
        f.write(f"Algorithms: {sorted(set(r['algorithm'] for r in results))}\n")
        f.write(f"Techniques: {sorted(set(r['technique'] for r in results))}\n")
        f.write(f"Environments: {sorted(set(r['environment'] for r in results))}\n\n")

        # GRS Summary
        f.write("=" * 80 + "\n")
        f.write("GRS SCORES (Mean ± Std)\n")
        f.write("=" * 80 + "\n\n")

        for env in sorted(set(r["environment"] for r in results)):
            f.write(f"\n{env.upper()}\n")
            f.write("-" * 60 + "\n")

            for algo in sorted(set(r["algorithm"] for r in results)):
                for tech in sorted(set(r["technique"] for r in results)):
                    key = f"{algo}_{tech}_{env}"
                    if key in aggregated:
                        stats_dict = compute_statistics(aggregated[key]["grs_values"])
                        f.write(f"  {algo.upper():>5} + {tech:<10}: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}\n")

        # Reward Summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("EVALUATION REWARDS (Mean ± Std)\n")
        f.write("=" * 80 + "\n\n")

        for env in sorted(set(r["environment"] for r in results)):
            f.write(f"\n{env.upper()}\n")
            f.write("-" * 60 + "\n")

            for algo in sorted(set(r["algorithm"] for r in results)):
                for tech in sorted(set(r["technique"] for r in results)):
                    key = f"{algo}_{tech}_{env}"
                    if key in aggregated:
                        stats_dict = compute_statistics(aggregated[key]["reward_values"])
                        f.write(f"  {algo.upper():>5} + {tech:<10}: {stats_dict['mean']:.2f} ± {stats_dict['std']:.2f}\n")

        # GAR Comparison
        if comparisons:
            f.write("\n" + "=" * 80 + "\n")
            f.write("GAR vs BASELINE COMPARISON\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"{'Config':<20} {'GRS Δ':>12} {'Reward Δ':>12} {'GRS p-val':>12} {'Rew p-val':>12}\n")
            f.write("-" * 70 + "\n")

            for key, data in sorted(comparisons.items()):
                config = f"{data['algo'].upper()}_{data['env']}"
                grs_p = f"{data['grs_pvalue']:.4f}" if data['grs_pvalue'] else "N/A"
                rew_p = f"{data['reward_pvalue']:.4f}" if data['reward_pvalue'] else "N/A"
                f.write(f"{config:<20} {data['grs_improvement']:>+12.3f} {data['reward_improvement']:>+12.2f} "
                       f"{grs_p:>12} {rew_p:>12}\n")

        # Key Findings
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")

        # Best technique per environment
        for env in sorted(set(r["environment"] for r in results)):
            env_results = [(k, v) for k, v in aggregated.items() if v["environment"] == env]
            if env_results:
                best_key = max(env_results, key=lambda x: np.mean(x[1]["grs_values"]))
                best_grs = np.mean(best_key[1]["grs_values"])
                f.write(f"  {env.upper()}: Best config = {best_key[0]} (GRS={best_grs:.3f})\n")

        # GAR effectiveness
        if comparisons:
            gar_helps_grs = sum(1 for c in comparisons.values() if c["grs_improvement"] > 0)
            gar_helps_reward = sum(1 for c in comparisons.values() if c["reward_improvement"] > 0)
            total = len(comparisons)

            f.write(f"\n  GAR improves GRS in {gar_helps_grs}/{total} cases ({100*gar_helps_grs/total:.1f}%)\n")
            f.write(f"  GAR improves Reward in {gar_helps_reward}/{total} cases ({100*gar_helps_reward/total:.1f}%)\n")

    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze RL Generalization Experiment Results")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR, help="Results directory")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    parser.add_argument("--plots-only", action="store_true", help="Only generate plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    # Create output directories
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # Load results
    print("Loading results...")
    results = load_all_results(args.results_dir)
    print(f"Loaded {len(results)} experiments")

    if not results:
        print("No results found!")
        return

    # Aggregate
    aggregated = aggregate_results(results)
    comparisons = compare_gar_effect(aggregated)

    if not args.plots_only:
        # Print tables
        print_summary_table(aggregated, metric="grs")
        print_summary_table(aggregated, metric="reward")
        print_gar_comparison(comparisons)

        # Generate report
        create_full_report(results, aggregated, comparisons, ANALYSIS_DIR)

        # LaTeX tables
        if args.latex:
            grs_latex = generate_latex_table(aggregated, metric="grs")
            reward_latex = generate_latex_table(aggregated, metric="reward")

            latex_path = os.path.join(ANALYSIS_DIR, "latex_tables.tex")
            with open(latex_path, 'w') as f:
                f.write("% GRS Table\n")
                f.write(grs_latex)
                f.write("\n\n% Reward Table\n")
                f.write(reward_latex)
            print(f"\nSaved LaTeX tables: {latex_path}")

    if not args.no_plots:
        # Generate plots
        print("\nGenerating plots...")
        plot_grs_comparison(aggregated, os.path.join(PLOTS_DIR, "grs_comparison.png"))
        plot_reward_comparison(aggregated, os.path.join(PLOTS_DIR, "reward_comparison.png"))
        plot_grs_curves(results, os.path.join(PLOTS_DIR, "grs_curves.png"))
        plot_technique_improvement(comparisons, os.path.join(PLOTS_DIR, "gar_improvement.png"))

    print("\nAnalysis complete!")
    print(f"  Report: {ANALYSIS_DIR}/analysis_report.txt")
    print(f"  Plots: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
