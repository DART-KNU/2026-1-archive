"""Chart generation: saves 8 PNG files to the output/ directory.

Charts produced:
  01_cumulative_returns.png   — Portfolio vs benchmark cumulative return
  02_excess_returns_annual.png— Annual excess return bar chart
  03_rolling_tracking_error.png— 252-day rolling TE
  04_drawdown.png              — Underwater chart (portfolio vs benchmark)
  05_sector_allocation.png     — Sector weights: portfolio vs benchmark
  06_top20_weights.png         — Top-20 stock weights (last rebalancing)
  07_te_comparison.png         — Simple mktcap TE vs modified TE
  08_factor_risk_decomp.png    — Factor vs specific risk decomposition
"""

import logging
import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# Korean font setup (macOS: AppleGothic, Linux: NanumGothic fallback)
def _set_korean_font() -> None:
    candidates = ["AppleGothic", "NanumGothic", "Malgun Gothic", "UnDotum"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    # No Korean font found — suppress missing glyph warnings
    import warnings
    warnings.filterwarnings("ignore", message="Glyph.*missing from font")

_set_korean_font()

logger = logging.getLogger(__name__)

DPI = 150
FIGSIZE_WIDE = (12, 5)
FIGSIZE_SQUARE = (8, 6)
PALETTE = {
    "portfolio": "#1f77b4",
    "benchmark": "#ff7f0e",
    "excess": "#2ca02c",
    "negative": "#d62728",
    "factor": "#9467bd",
    "specific": "#8c564b",
}


def _save(fig: plt.Figure, name: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_cumulative_returns(
    nav: pd.Series,
    benchmark: pd.Series,
    output_dir: str,
) -> None:
    """01: Cumulative return comparison."""
    port_ret = nav / nav.iloc[0] - 1.0
    bench_ret = benchmark.reindex(nav.index).ffill() / benchmark.reindex(nav.index).ffill().iloc[0] - 1.0

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(port_ret.index, port_ret * 100, label="Portfolio", color=PALETTE["portfolio"], linewidth=1.5)
    ax.plot(bench_ret.index, bench_ret * 100, label="KOSPI Benchmark", color=PALETTE["benchmark"], linewidth=1.5, linestyle="--")
    ax.set_title("Cumulative Returns: Portfolio vs KOSPI (2019–2024)", fontsize=13)
    ax.set_ylabel("Cumulative Return (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "01_cumulative_returns.png", output_dir)


def plot_annual_excess_returns(
    annual_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """02: Annual excess return bar chart."""
    fig, ax = plt.subplots(figsize=(9, 5))
    years = annual_df.index.tolist()
    excess = annual_df["excess_return"] * 100

    colors = [PALETTE["excess"] if v >= 0 else PALETTE["negative"] for v in excess]
    bars = ax.bar(years, excess, color=colors, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)

    for bar, val in zip(bars, excess):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.02 if val >= 0 else -0.12),
            f"{val:+.2f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
        )

    ax.set_title("Annual Excess Return vs KOSPI", fontsize=13)
    ax.set_ylabel("Excess Return (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "02_excess_returns_annual.png", output_dir)


def plot_rolling_tracking_error(
    nav: pd.Series,
    benchmark: pd.Series,
    output_dir: str,
    window: int = 252,
) -> None:
    """03: 252-day rolling tracking error."""
    port_ret = nav.pct_change()
    bench_ret = benchmark.reindex(nav.index).ffill().pct_change()
    excess = port_ret - bench_ret
    rolling_te = excess.rolling(window).std() * np.sqrt(252) * 100

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(rolling_te.index, rolling_te, color=PALETTE["portfolio"], linewidth=1.5)
    ax.axhline(2.0, color=PALETTE["negative"], linestyle="--", linewidth=1.0, label="±2% institutional threshold")
    ax.set_title(f"Rolling {window}-day Annualised Tracking Error (%)", fontsize=13)
    ax.set_ylabel("Tracking Error (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "03_rolling_tracking_error.png", output_dir)


def plot_drawdown(
    nav: pd.Series,
    benchmark: pd.Series,
    output_dir: str,
) -> None:
    """04: Underwater chart."""
    bench_aligned = benchmark.reindex(nav.index).ffill()

    port_dd = nav / nav.cummax() - 1.0
    bench_dd = bench_aligned / bench_aligned.cummax() - 1.0

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.fill_between(port_dd.index, port_dd * 100, 0, alpha=0.6, color=PALETTE["portfolio"], label="Portfolio")
    ax.fill_between(bench_dd.index, bench_dd * 100, 0, alpha=0.3, color=PALETTE["benchmark"], label="KOSPI")
    ax.set_title("Drawdown: Portfolio vs KOSPI", fontsize=13)
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "04_drawdown.png", output_dir)


def plot_sector_allocation(
    port_sector_weights: pd.Series,
    bench_sector_weights: pd.Series,
    output_dir: str,
) -> None:
    """05: Sector allocation comparison (horizontal bar)."""
    all_sectors = sorted(set(port_sector_weights.index) | set(bench_sector_weights.index))
    p = port_sector_weights.reindex(all_sectors).fillna(0.0) * 100
    b = bench_sector_weights.reindex(all_sectors).fillna(0.0) * 100

    y = np.arange(len(all_sectors))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(5, len(all_sectors) * 0.6)))
    ax.barh(y - height / 2, p, height, label="Portfolio", color=PALETTE["portfolio"])
    ax.barh(y + height / 2, b, height, label="KOSPI Benchmark", color=PALETTE["benchmark"], alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(all_sectors, fontsize=10)
    ax.set_xlabel("Weight (%)")
    ax.set_title("Sector Allocation: Portfolio vs KOSPI Benchmark", fontsize=13)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "05_sector_allocation.png", output_dir)


def plot_top20_weights(
    weights: pd.Series,
    output_dir: str,
) -> None:
    """06: Top-20 holdings by weight (horizontal bar)."""
    top20 = weights.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top20)), top20 * 100, color=PALETTE["portfolio"])
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index.tolist(), fontsize=9)
    ax.axvline(15.0, color=PALETTE["negative"], linestyle="--", linewidth=1.0, label="15% cap")
    ax.set_xlabel("Weight (%)")
    ax.set_title("Top 20 Holdings (Last Rebalancing)", fontsize=13)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "06_top20_weights.png", output_dir)


def plot_te_comparison(
    modified_te: float,
    simple_te: float,
    output_dir: str,
) -> None:
    """07: TE comparison — simple market-cap vs modified strategy."""
    labels = ["Simple Market-Cap\n(no cap, no filter)", "Modified Strategy\n(15% cap + filter)"]
    values = [simple_te * 100, modified_te * 100]
    colors = [PALETTE["negative"], PALETTE["portfolio"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_title("Annualised Tracking Error Comparison", fontsize=13)
    ax.set_ylabel("Tracking Error (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "07_te_comparison.png", output_dir)


def plot_factor_risk_decomp(
    factor_pct: float,
    specific_pct: float,
    output_dir: str,
) -> None:
    """08: Factor vs specific risk decomposition pie chart."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sizes = [factor_pct * 100, specific_pct * 100]
    labels = [
        f"Common Factor Risk\n({factor_pct*100:.1f}%)",
        f"Specific Risk\n({specific_pct*100:.1f}%)",
    ]
    colors = [PALETTE["factor"], PALETTE["specific"]]
    ax.pie(sizes, labels=labels, colors=colors, startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Portfolio Risk Decomposition (Barra-style)", fontsize=13)
    fig.tight_layout()
    _save(fig, "08_factor_risk_decomp.png", output_dir)


def generate_all_charts(
    nav: pd.Series,
    benchmark: pd.Series,
    annual_df: pd.DataFrame,
    last_port_sector: pd.Series,
    last_bench_sector: pd.Series,
    last_weights: pd.Series,
    modified_te: float,
    simple_te: float,
    factor_pct: float,
    specific_pct: float,
    output_dir: str,
) -> None:
    """Generate and save all 8 charts."""
    logger.info("Generating charts → %s", output_dir)
    plot_cumulative_returns(nav, benchmark, output_dir)
    plot_annual_excess_returns(annual_df, output_dir)
    plot_rolling_tracking_error(nav, benchmark, output_dir)
    plot_drawdown(nav, benchmark, output_dir)
    plot_sector_allocation(last_port_sector, last_bench_sector, output_dir)
    plot_top20_weights(last_weights, output_dir)
    plot_te_comparison(modified_te, simple_te, output_dir)
    plot_factor_risk_decomp(factor_pct, specific_pct, output_dir)
    logger.info("All 8 charts saved to %s", output_dir)
