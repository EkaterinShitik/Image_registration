"""Builds a single summary figure comparing all registration methods across all metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_benchmark(df: pd.DataFrame, output_path: str) -> None:
    """Render one figure with a bar-chart subplot per metric, all methods side by side."""
    methods = df["method"].tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(
        "Non-rigid registration benchmark: 108.jpeg \u2192 109.jpeg (adjacent Romanowsky-Giemsa sections)",
        fontsize=15,
        fontweight="bold",
    )

    def bar(ax, col, title, ylabel, higher_is_better, fmt="{:.3f}", yerr=None):
        values = df[col].to_numpy()
        bars = ax.bar(methods, values, color=colors, yerr=yerr, capsize=4)
        arrow = "\u2191 better" if higher_is_better else "\u2193 better"
        ax.set_title(f"{title}\n({arrow})", fontsize=11)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        for b, v in zip(bars, values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    bar(axes[0, 0], "nmi", "Normalized Mutual Information", "NMI", True)
    bar(
        axes[0, 1],
        "tre_mean",
        "Target Registration Error",
        "TRE, px (mean \u00b1 std)",
        False,
        yerr=df["tre_std"].to_numpy(),
    )
    bar(axes[0, 2], "dice", "Dice Similarity Coefficient (tissue mask)", "DSC", True)
    bar(axes[1, 0], "hd95", "Hausdorff Distance 95th pct (tissue contour)", "HD95, px", False)
    bar(
        axes[1, 1],
        "jac_negative_pct",
        "Jacobian determinant of deformation field",
        "% pixels with det(J) \u2264 0 (folding)",
        False,
    )
    bar(axes[1, 2], "runtime_sec", "Runtime", "seconds", False, fmt="{:.1f}")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
