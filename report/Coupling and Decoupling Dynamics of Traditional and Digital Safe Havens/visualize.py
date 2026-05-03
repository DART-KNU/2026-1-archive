"""
Rolling correlation visualization script.
Input : merged_data.csv
Output: fig1 ~ fig7 PNG files
"""

from __future__ import annotations

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

# Global style: thinner/smoother lines and balanced visuals
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#d8dee9",
        "axes.linewidth": 0.8,
        "grid.color": "#cfd8dc",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": True,
        "savefig.facecolor": "white",
        "savefig.dpi": 220,
    }
)

INPUT_FILE = "merged_data.csv"
WINDOWS = [30, 60, 90]

WINDOW_COLORS = {
    30: "#1f78b4",  # blue
    60: "#e31a1c",  # red
    90: "#111111",  # black
}

PAIRS = {
    "Gold_BTC": ("Gold_Return", "BTC_Return", "Gold-Bitcoin"),
    "Silver_BTC": ("Silver_Return", "BTC_Return", "Silver-Bitcoin"),
    "Gold_Silver": ("Gold_Return", "Silver_Return", "Gold-Silver"),
    "Silver_Cu": ("Silver_Return", "Copper_Return", "Silver-Copper"),
}

EVENTS = {
    "2020-03-11": ("COVID-19", "#6a1b9a"),      # deep purple
    "2022-02-24": ("Ukraine War", "#ef6c00"),   # deep orange
    "2022-03-16": ("Fed Hike Start", "#c62828"),# deep red
    "2023-07-26": ("Fed Hike End", "#8e24aa"),  # bold violet
    "2024-01-10": ("BTC ETF", "#2e7d32"),       # deep green
    "2024-11-06": ("Trump Elected", "#1565c0"), # deep blue
}


def add_events(ax, data_index):
    ylim = ax.get_ylim()
    text_y = ylim[1] - (ylim[1] - ylim[0]) * 0.03
    for date_str, (label, color) in EVENTS.items():
        dt = pd.to_datetime(date_str)
        if data_index.min() <= dt <= data_index.max():
            ax.axvline(dt, color=color, linestyle="--", linewidth=1.0, alpha=0.72, zorder=2)
            ax.text(
                dt,
                text_y,
                label,
                rotation=90,
                va="top",
                ha="right",
                fontsize=7.0,
                alpha=0.9,
                color=color,
            )


def fmt_xaxis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def base_line(ax):
    ax.axhline(0, color="#7d8597", linewidth=0.75, alpha=0.38)


def plot_corr_line(ax, x, y, label, color, linewidth=1.35, alpha=0.95):
    ax.plot(
        x,
        y,
        label=label,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        antialiased=True,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


print("=" * 70)
print("Rolling Correlation Visualization")
print("=" * 70)
print(f"\n[1/8] Load {INPUT_FILE} and calculate rolling correlations...")

df = None
for enc in ("utf-8", "cp949", "euc-kr"):
    try:
        df = pd.read_csv(INPUT_FILE, parse_dates=["Date"], index_col="Date", encoding=enc)
        break
    except UnicodeDecodeError:
        continue

if df is None:
    raise UnicodeDecodeError("read_csv", b"", 0, 1, f"Failed to decode {INPUT_FILE}")

corr = {}
for pair_key, (col_a, col_b, _) in PAIRS.items():
    for w in WINDOWS:
        col_name = f"Corr_{pair_key}_{w}d"
        if col_name in df.columns:
            corr[(pair_key, w)] = df[col_name]
        elif col_a in df.columns and col_b in df.columns:
            corr[(pair_key, w)] = df[col_a].rolling(w).corr(df[col_b])

print(f"  OK {len(df)} rows ({df.index[0].date()} ~ {df.index[-1].date()})")

# Fig 1
print("\n[2/8] Fig1: Gold-Bitcoin")
fig, ax = plt.subplots(figsize=(16, 6))
for w in WINDOWS:
    s = corr.get(("Gold_BTC", w))
    if s is not None:
        plot_corr_line(ax, s.index, s, f"{w}-Day", WINDOW_COLORS[w])
base_line(ax)
add_events(ax, df.index)
fmt_xaxis(ax)
ax.set_title("Gold-Bitcoin Rolling Correlation | 30 / 60 / 90-Day", fontsize=15, fontweight="bold", pad=14)
ax.set_ylabel("Correlation Coefficient", fontsize=11)
ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
ax.grid(True)
plt.tight_layout()
plt.savefig("fig1_gold_btc.png", bbox_inches="tight")
print("  OK fig1_gold_btc.png")
plt.close()

# Fig 2
print("[3/8] Fig2: Silver-Bitcoin")
fig, ax = plt.subplots(figsize=(16, 6))
for w in WINDOWS:
    s = corr.get(("Silver_BTC", w))
    if s is not None:
        plot_corr_line(ax, s.index, s, f"{w}-Day", WINDOW_COLORS[w])
base_line(ax)
add_events(ax, df.index)
fmt_xaxis(ax)
ax.set_title("Silver-Bitcoin Rolling Correlation | 30 / 60 / 90-Day", fontsize=15, fontweight="bold", pad=14)
ax.set_ylabel("Correlation Coefficient", fontsize=11)
ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
ax.grid(True)
plt.tight_layout()
plt.savefig("fig2_silver_btc.png", bbox_inches="tight")
print("  OK fig2_silver_btc.png")
plt.close()

# Fig 3
print("[4/8] Fig3: Gold-Silver")
fig, ax = plt.subplots(figsize=(16, 6))
for w in WINDOWS:
    s = corr.get(("Gold_Silver", w))
    if s is not None:
        plot_corr_line(ax, s.index, s, f"{w}-Day", WINDOW_COLORS[w])
base_line(ax)
add_events(ax, df.index)
fmt_xaxis(ax)
ax.set_title(
    "Gold-Silver Rolling Correlation | 30 / 60 / 90-Day\n(Indicator of Silver's Precious Metal Regime)",
    fontsize=14,
    fontweight="bold",
    pad=14,
)
ax.set_ylabel("Correlation Coefficient", fontsize=11)
ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
ax.grid(True)
plt.tight_layout()
plt.savefig("fig3_gold_silver.png", bbox_inches="tight")
print("  OK fig3_gold_silver.png")
plt.close()

# Fig 4
print("[5/8] Fig4: Silver-Copper")
fig, ax = plt.subplots(figsize=(16, 6))
for w in WINDOWS:
    s = corr.get(("Silver_Cu", w))
    if s is not None:
        plot_corr_line(ax, s.index, s, f"{w}-Day", WINDOW_COLORS[w])
base_line(ax)
add_events(ax, df.index)
fmt_xaxis(ax)
ax.set_title(
    "Silver-Copper Rolling Correlation | 30 / 60 / 90-Day\n(Indicator of Silver's Industrial Metal Regime)",
    fontsize=14,
    fontweight="bold",
    pad=14,
)
ax.set_ylabel("Correlation Coefficient", fontsize=11)
ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
ax.grid(True)
plt.tight_layout()
plt.savefig("fig4_silver_copper.png", bbox_inches="tight")
print("  OK fig4_silver_copper.png")
plt.close()

# Fig 3-1 (60d overlap)
print("[5-1/8] Fig3-1: Silver-Gold + Silver-Copper (60d overlap)")
fig, ax = plt.subplots(figsize=(16, 6))
s_gold = corr.get(("Gold_Silver", 60))
s_copper = corr.get(("Silver_Cu", 60))
if s_gold is not None:
    plot_corr_line(ax, s_gold.index, s_gold, "Silver-Gold (60-Day)", "#e31a1c", linewidth=1.7, alpha=0.98)
if s_copper is not None:
    plot_corr_line(ax, s_copper.index, s_copper, "Silver-Copper (60-Day)", "#1f78b4", linewidth=1.7, alpha=0.98)
base_line(ax)
fmt_xaxis(ax)
ax.set_title("Silver-Gold vs Silver-Copper Rolling Correlation | 60-Day", fontsize=15, fontweight="bold", pad=14)
ax.set_ylabel("Correlation Coefficient", fontsize=11)
ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
ax.grid(True)
plt.tight_layout()
plt.savefig("fig3-1_gold_silver_60d.png", bbox_inches="tight")
print("  OK fig3-1_gold_silver_60d.png")
plt.close()

# Fig 4-1 (same overlap as fig3-1)
print("[5-2/8] Fig4-1: Silver-Gold + Silver-Copper (60d overlap)")
fig, ax = plt.subplots(figsize=(16, 6))
if s_gold is not None:
    plot_corr_line(ax, s_gold.index, s_gold, "Silver-Gold (60-Day)", "#e31a1c", linewidth=1.7, alpha=0.98)
if s_copper is not None:
    plot_corr_line(ax, s_copper.index, s_copper, "Silver-Copper (60-Day)", "#1f78b4", linewidth=1.7, alpha=0.98)
base_line(ax)
fmt_xaxis(ax)
ax.set_title("Silver-Gold vs Silver-Copper Rolling Correlation | 60-Day", fontsize=15, fontweight="bold", pad=14)
ax.set_ylabel("Correlation Coefficient", fontsize=11)
ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
ax.grid(True)
plt.tight_layout()
plt.savefig("fig4-1_silver_copper_60d.png", bbox_inches="tight")
print("  OK fig4-1_silver_copper_60d.png")
plt.close()

# Fig 5
print("[6/8] Fig5: 2x2 grid")
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.flatten()

for idx, (pair_key, (_, _, label)) in enumerate(PAIRS.items()):
    ax = axes[idx]
    for w in WINDOWS:
        s = corr.get((pair_key, w))
        if s is not None:
            plot_corr_line(ax, s.index, s, f"{w}d", WINDOW_COLORS[w], linewidth=1.2, alpha=0.92)
    base_line(ax)
    for date_str, (_, color) in EVENTS.items():
        dt = pd.to_datetime(date_str)
        if df.index.min() <= dt <= df.index.max():
            ax.axvline(dt, color=color, linestyle="--", linewidth=0.9, alpha=0.4)
    fmt_xaxis(ax)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_ylabel("Correlation", fontsize=10)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(True)

plt.suptitle("Rolling Correlation by Asset Pair | 30 / 60 / 90-Day", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("fig5_grid_by_pair.png", bbox_inches="tight")
print("  OK fig5_grid_by_pair.png")
plt.close()

# Fig 6
print("[7/8] Fig6: boxplot")
fig, axes = plt.subplots(1, 4, figsize=(22, 6))

for idx, (pair_key, (_, _, label)) in enumerate(PAIRS.items()):
    ax = axes[idx]
    plot_data, tick_labels = [], []
    for w in WINDOWS:
        s = corr.get((pair_key, w))
        if s is not None:
            plot_data.append(s.dropna().values)
            tick_labels.append(f"{w}d")

    bp = ax.boxplot(
        plot_data,
        labels=tick_labels,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="white", linewidth=1.4),
    )
    for patch, w in zip(bp["boxes"], WINDOWS):
        patch.set_facecolor(WINDOW_COLORS[w])
        patch.set_alpha(0.62)

    ax.axhline(0, color="#2c3e50", linestyle="--", linewidth=0.9, alpha=0.45)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Window", fontsize=10)
    ax.set_ylabel("Correlation Coefficient", fontsize=10)
    ax.grid(True, axis="y")

plt.suptitle("Correlation Distribution by Window Size", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("fig6_boxplot.png", bbox_inches="tight")
print("  OK fig6_boxplot.png")
plt.close()

# Fig 7
print("[8/8] Fig7: correlation volatility")
fig, axes = plt.subplots(1, 4, figsize=(22, 6))

for idx, (pair_key, (_, _, label)) in enumerate(PAIRS.items()):
    ax = axes[idx]
    for w in WINDOWS:
        s = corr.get((pair_key, w))
        if s is not None:
            rolling_std = s.rolling(30).std()
            plot_corr_line(ax, rolling_std.index, rolling_std, f"{w}d", WINDOW_COLORS[w], linewidth=1.15, alpha=0.9)
    fmt_xaxis(ax)
    ax.set_title(f"{label}\n(30d Rolling Std)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rolling Std", fontsize=10)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True)

plt.suptitle("Correlation Volatility | 30-Day Rolling Std", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("fig7_volatility.png", bbox_inches="tight")
print("  OK fig7_volatility.png")
plt.close()

print("\n" + "=" * 70)
print("Done. Generated files:")
files = [
    ("fig1_gold_btc.png", "Gold-Bitcoin Rolling Correlation"),
    ("fig2_silver_btc.png", "Silver-Bitcoin Rolling Correlation"),
    ("fig3_gold_silver.png", "Gold-Silver (Precious Regime)"),
    ("fig4_silver_copper.png", "Silver-Copper (Industrial Regime)"),
    ("fig3-1_gold_silver_60d.png", "Silver-Gold + Silver-Copper (60-Day overlap)"),
    ("fig4-1_silver_copper_60d.png", "Silver-Gold + Silver-Copper (60-Day overlap)"),
    ("fig5_grid_by_pair.png", "2x2 Grid (all pairs, all windows)"),
    ("fig6_boxplot.png", "Boxplot by window"),
    ("fig7_volatility.png", "Volatility of correlation (30d std)"),
]
for i, (fname, desc) in enumerate(files, 1):
    print(f"  {i}. {fname:<35} {desc}")
print("=" * 70)
