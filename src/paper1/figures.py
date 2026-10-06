"""Figures 2–4 and S1–S3 from the single audited seven-model result set."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from .config import LABELS, NEW

MODELS = list(LABELS)
COLORS = ["#4C78A8", "#E68422", "#7654A3", "#888888", "#438979", "#B28B68", "#555555"]
SHORT = [
    "Name-based readout",
    "Coordinate-based readout",
    "Coordinate–time\npolynomial ridge",
    "Hourly mean",
    "Region–hour categorical",
    "Marginal product",
    "Log-linear gravity",
]
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]
LINES = ["-", "-", "-", "--", "-.", ":", "--"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
QA = []


def read(root, name):
    return pd.read_csv(root / "results" / name)


def save(root, fig, folder, name):
    for ax in fig.axes:
        for artist in ax.collections:
            artist.set_rasterized(False)
        if ax.get_label() == "<colorbar>":
            for artist in ax.collections:
                artist.set_edgecolor("face")
    fig.canvas.draw()
    fig.savefig(root / "figures" / folder / (name + ".pdf"), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(
        root / "figures" / folder / (name + ".png"), dpi=600, bbox_inches="tight", pad_inches=0.12
    )
    QA.append(
        dict(figure=name, axes=len(fig.axes), png_dpi=600, pdf="vector paths and embedded fonts")
    )
    plt.close(fig)


def data(root, df, name):
    df.to_csv(root / "figure_data" / (name + ".csv"), index=False)


def table(root, df, name):
    df.to_csv(root / "tables" / (name + ".csv"), index=False)

    def fmt(v):
        if pd.isna(v):
            return "N/A"
        if isinstance(v, (float, np.floating)):
            if "S1_" in name:
                return f"{v:g}"
            if "S4_" in name or "S6_" in name:
                return f"{v:.6f}"
            return f"{v:.3f}"
        return str(v)

    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    lines += [
        "| " + " | ".join((fmt(v) for v in row)) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    (root / "tables" / (name + ".md")).write_text("\n".join(lines) + "\n")


def workflow(root):
    from matplotlib import font_manager
    import os

    font_path = os.environ.get("PAPER1_ARIAL_FONT", "")
    if Path(font_path).is_file():
        font_manager.fontManager.addfont(font_path)
    font_manager.findfont("Arial", fallback_to_default=False)
    with plt.rc_context({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(180 / 25.4, 252 / 25.4))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 180)
        ax.set_ylim(252, 0)
        ax.axis("off")

        def box(x, y, w, h, txt, color="#F3F4F5", size=8, lw=0.8):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle="round,pad=0.35,rounding_size=1.2",
                    facecolor=color,
                    edgecolor="#626970",
                    linewidth=lw,
                )
            )
            return ax.text(
                x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=size, linespacing=1.35
            )

        def arrow(points, color="#58616B"):
            for a, b in zip(points[:-2], points[1:-1]):
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.85)
            ax.annotate(
                "",
                xy=points[-1],
                xytext=points[-2],
                arrowprops=dict(arrowstyle="->", color=color, lw=0.85, shrinkA=0, shrinkB=1),
            )

        def heading(y, s):
            ax.text(5, y, s, fontsize=9.2, weight="bold", va="center")

        heading(5, "1  Prepare inputs without fitting to mobility targets")
        box(
            5,
            11,
            96,
            29,
            "MAIN CONDITIONS\nName-based / Coordinate-based readout\nDistrict names or coordinates + arrival hour\nFrozen Llama 3 → 4,096 features\nNo flow targets; no LLM fitting",
            "#EAF0FA",
            8.1,
            1.2,
        )
        box(
            107,
            11,
            68,
            29,
            "DIRECT-INPUT COMPARISON\nOrigin / destination latitude and longitude\nsin(2πt/24), cos(2πt/24)\nt = arrival hour; 6 basic features",
            "#F0EAF5",
            8,
        )
        arrow([(53, 40.5), (53, 45)])
        arrow([(141, 40.5), (141, 45)])
        box(
            5,
            45,
            170,
            15,
            "2  Reuse saved outer folds: 300 unordered district pairs\nBoth directions and all 24 hours stay together",
            "#E9EDF0",
            8.4,
        )
        arrow([(68, 60.5), (68, 66)])
        arrow([(158, 60.5), (158, 66)])
        box(5, 66, 127, 12, "Outer training: 240 pairs", "#EAF0FA", 8.5)
        box(140, 66, 35, 12, "Outer evaluation\n60 pairs", "#F4F4F4", 8)
        arrow([(68, 78.5), (68, 84)])
        arrow([(132.5, 72), (136, 72), (136, 106), (140, 106)])
        ax.add_patch(
            FancyBboxPatch(
                (5, 84),
                127,
                77,
                boxstyle="round,pad=0.35,rounding_size=1.2",
                facecolor="#FAFBFC",
                edgecolor="#626970",
                linewidth=0.85,
            )
        )
        ax.text(68, 90, "3  Inner penalty selection", ha="center", fontsize=9, weight="bold")
        ax.text(
            68,
            98,
            "Two LLM readouts + Coordinate–time polynomial ridge\n+ Region–hour categorical",
            ha="center",
            va="center",
            fontsize=8,
            linespacing=1.25,
        )
        box(
            9,
            107,
            59,
            29,
            "Inner training: 210 pairs\nFit preprocessing and ridge\nPolynomial: scale 6 → expand 27\n→ scale 27 → ridge\nFit log-target mean and coefficients",
            "#EAF0FA",
            7.7,
        )
        box(
            77,
            107,
            51,
            29,
            "Validation: 30 pairs\nApply fitted transforms\nPredict log1p(flow)\nCalculate log-MSE",
            "#F0F1F3",
            8,
        )
        arrow([(68.5, 120), (76.5, 120)])
        arrow([(102, 136.5), (102, 141), (68, 141), (68, 146)])
        box(9, 146, 119, 11, "Select penalty with minimum validation log-MSE", "#EDE9F3", 8)
        box(
            140,
            90,
            35,
            46,
            "OTHER BENCHMARKS\n\nHourly mean\nMarginal product\nLog-linear gravity\n\nNo ridge tuning",
            "#F4F4F4",
            7.7,
        )
        arrow([(68, 161.5), (68, 170)])
        arrow([(158, 136.5), (158, 170)])
        box(
            5,
            170,
            127,
            27,
            "4  Refit on all 240 outer-training pairs\nRelearn preprocessing and log-target mean; fit ridge\nPolynomial: scale 6 → expand 27 → scale 27 → ridge\nBoth scalers are fitted on outer training only",
            "#EAF0FA",
            8,
        )
        box(
            140,
            170,
            35,
            27,
            "Fit / compute\nbenchmark statistics\non all 240\nouter-training pairs",
            "#F4F4F4",
            7.7,
        )
        arrow([(68, 197.5), (68, 207)])
        arrow([(158, 197.5), (158, 207)])
        arrow([(175.5, 72), (178, 72), (178, 216), (175.5, 216)])
        box(
            5,
            207,
            170,
            18,
            "5  Predict the 60 held-out evaluation pairs\nApply outer-training preprocessing and fitted models; no refitting",
            "#E9EDF0",
            8.4,
        )
        arrow([(90, 225.5), (90, 234)])
        box(
            5,
            234,
            170,
            14,
            "6  Combine held-out predictions from five outer folds\n14,400 OD–hours; exactly one OOF prediction per observation",
            "#E9EDF0",
            8.4,
        )
        fig.canvas.draw()
        assert all((t.get_fontfamily() == ["Arial"] for t in ax.texts))
        fig.savefig(root / "figures/main/Figure_2_workflow.pdf")
        fig.savefig(root / "figures/main/Figure_2_workflow.png", dpi=600)
        plt.close(fig)


def profiles(root, g):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    fig.subplots_adjust(bottom=0.23, hspace=0.42, wspace=0.32)
    for idx, (typ, metric, title, ylabel) in enumerate(
        [
            ("flow", "total_bias_pct", "(a) Flow-scale bias", "Total-flow bias (%)"),
            ("flow", "smape", "(b) Flow-scale relative error", "sMAPE"),
            (
                "distance",
                "mae",
                "(c) Distance-specific absolute error",
                "MAE (movements per OD–hour)",
            ),
            ("distance", "smape", "(d) Distance-specific relative error", "sMAPE"),
        ]
    ):
        ax = axes.flat[idx]
        order = [f"D{i}" for i in range(1, 11)] if typ == "flow" else [f"Q{i}" for i in range(1, 6)]
        part = g[g.group_type == typ]
        data(root, part[["condition", "group", "n_rows", metric]], f"Figure_3_panel_{'abcd'[idx]}")
        for m, c, marker, ls in zip(MODELS, COLORS, MARKERS, LINES):
            p = part[part.condition == m].set_index("group").loc[order]
            ax.plot(
                range(len(order)),
                p[metric],
                label=LABELS[m],
                color=c,
                marker=marker,
                ls=ls,
                lw=1.8 if m in MODELS[:3] else 1,
                ms=4 if m in MODELS[:3] else 3,
                alpha=1 if m in MODELS[:3] else 0.75,
            )
        ax.set_xticks(range(len(order)), order)
        ax.set_xlabel(
            "Mobility-data flow decile" if typ == "flow" else "District-pair distance quintile"
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "total_bias_pct":
            ax.axhline(0, c="black", lw=0.6)
        else:
            ax.set_ylim(bottom=0)
    inset = axes[0, 0].inset_axes([0.44, 0.4, 0.53, 0.53])
    for m, c, ls in zip(MODELS, COLORS, LINES):
        p = (
            g[(g.group_type == "flow") & (g.condition == m)]
            .set_index("group")
            .loc[[f"D{i}" for i in range(7, 11)]]
        )
        inset.plot(range(7, 11), p.total_bias_pct, color=c, ls=ls, lw=1.1)
    inset.set_xticks(range(7, 11), ["D7", "D8", "D9", "D10"])
    inset.tick_params(labelsize=7)
    inset.axhline(0, c="black", lw=0.5)
    inset.set_title("D7–D10", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    save(root, fig, "main", "Figure_3_error_profiles")


def districts(root, d):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    fig.subplots_adjust(left=0.29, bottom=0.19, wspace=0.2)
    for k, (metric, title, xlabel) in enumerate(
        [
            ("abs_total_bias_pct", "(a) District totals", "Absolute district-total bias (%)"),
            ("tv", "(b) Counterpart allocation", "Allocation discrepancy (TV)"),
        ]
    ):
        for j, (m, c) in enumerate(zip(MODELS, COLORS)):
            for side, off, marker in [("origin", -0.16, "o"), ("destination", 0.16, "^")]:
                p = d[(d.condition == m) & (d.side == side)].sort_values("district_code")
                v = p[metric].to_numpy()
                q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
                ax = axes[k]
                ax.scatter(
                    v, j + off + np.linspace(-0.04, 0.04, 25), s=15, alpha=0.4, c=c, marker=marker
                )
                ax.plot([q1, q3], [j + off] * 2, c=c, lw=2)
                ax.scatter([med], [j + off], c=c, s=35, marker=marker, edgecolors="black", lw=0.5)
        axes[k].set_title(title, loc="left")
        axes[k].set_xlabel(xlabel)
        axes[k].set_xlim(left=0)
        axes[k].grid(axis="x", alpha=0.2)
        data(root, d[["condition", "side", "district_code", metric]], f"Figure_4_panel_{'ab'[k]}")
    axes[0].set_yticks(range(7), SHORT)
    axes[0].invert_yaxis()
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color="#555555", label="Outflow"),
            Line2D([], [], marker="^", ls="", color="#555555", label="Inflow"),
            Line2D(
                [], [], marker="o", ls="-", color="#555555", label="Median and interquartile range"
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    save(root, fig, "main", "Figure_4_district_totals_allocation")


def supplements(root, g, d):
    h = g[g.group_type == "distance_hour"]
    order = [f"Q{i}" for i in range(1, 6)]
    pairs = [
        ("name_llm", "coordinate_llm"),
        ("corrected_loglinear_gravity", "coordinate_llm"),
        (NEW, "coordinate_llm"),
        (NEW, "name_llm"),
    ]
    for metric, num, label in [
        ("mae", 1, "Mean AE difference (movements per OD–hour)"),
        ("smape", 2, "Mean sAE difference"),
    ]:
        fig, axes = plt.subplots(4, 1, figsize=(9, 10))
        fig.subplots_adjust(left=0.1, right=0.84, hspace=0.62, bottom=0.07, top=0.96)
        mats = []
        for a, b in pairs:
            p = h[h.condition == a].merge(
                h[h.condition == b], on=["group", "hour"], suffixes=("_a", "_b")
            )
            p["difference"] = p[metric + "_a"] - p[metric + "_b"]
            mats.append(
                p.pivot(index="group", columns="hour", values="difference").loc[order, range(24)]
            )
        vmax = max((np.abs(m.to_numpy()).max() for m in mats))
        for j, (ax, mat, (a, b)) in enumerate(zip(axes, mats, pairs)):
            im = ax.pcolormesh(
                np.arange(25) - 0.5, np.arange(6) - 0.5, mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax
            )
            ax.invert_yaxis()
            ax.set_xticks(range(24))
            ax.set_yticks(range(5), order)
            ax.set_ylabel("Distance group")
            ax.set_title(f"({'abcd'[j]}) {LABELS[a]} minus {LABELS[b]}", loc="left", fontsize=9)
            data(
                root,
                mat.rename_axis("distance_bin").reset_index(),
                f"Figure_S{num}_panel_{'abcd'[j]}",
            )
        axes[-1].set_xlabel("Arrival hour")
        fig.colorbar(im, cax=fig.add_axes([0.89, 0.13, 0.025, 0.73])).set_label(label)
        save(root, fig, "supplementary", f"Figure_S{num}_distance_hour")


def district_heatmap_publication(root):
    """Layout-only S3 export at 180 mm width; never rewrite source CSVs."""
    d = read(root, "district_diagnostics.csv")
    names = pd.read_csv(root / "inputs/district_names.csv").set_index("district_code")
    codes = sorted(d.district_code.unique())
    assert len(codes) == 25 and set(codes) == set(names.index)
    fig = plt.figure(figsize=(180 / 25.4, 260 / 25.4))
    vmax = d.total_bias_pct.abs().max()
    for r, side in enumerate(["origin", "destination"]):
        for c, metric in enumerate(["total_bias_pct", "tv"]):
            left = 32 if c == 0 else 126
            bottom = 154 if r == 0 else 35
            ax = fig.add_axes([left / 180, bottom / 260, 51 / 180, 85 / 260])
            mat = (
                d[d.side == side]
                .pivot(index="district_code", columns="condition", values=metric)
                .loc[codes, MODELS]
            )
            data(root, mat.reset_index(), f"Figure_S3_panel_{chr(97 + r * 2 + c)}")
            im = ax.pcolormesh(
                np.arange(8) - 0.5,
                np.arange(26) - 0.5,
                mat,
                cmap="RdBu_r" if c == 0 else "viridis",
                vmin=-vmax if c == 0 else 0,
                vmax=vmax if c == 0 else 1,
                rasterized=False,
            )
            ax.invert_yaxis()
            ax.set_yticks(range(25), names.loc[codes, "english_name"], fontsize=7)
            ax.set_xticks(range(7), SHORT, rotation=60, ha="right", fontsize=7)
            ax.tick_params(axis="both", pad=2, length=2)
            ax.set_title(
                f"({'abcd'[r * 2 + c]}) "
                + ("Outflow" if r == 0 else "Inflow")
                + (" total bias (%)" if c == 0 else " allocation TV"),
                loc="left",
                fontsize=7.5,
                pad=5,
            )
            if r == 0:
                cb = fig.colorbar(
                    im,
                    cax=fig.add_axes([left / 180, 250 / 260, 51 / 180, 2.5 / 260]),
                    orientation="horizontal",
                )
                cb.ax.tick_params(labelsize=6.5, length=2, pad=2)
                cb.solids.set_rasterized(False)
                cb.solids.set_edgecolor("face")
    fig.savefig(root / "figures/supplementary/Figure_S3_districts.pdf")
    fig.savefig(root / "figures/supplementary/Figure_S3_districts.png", dpi=600)
    plt.close(fig)


def main(root):
    g = read(root, "group_metrics.csv")
    d = read(root, "district_diagnostics.csv")
    o = read(root, "metrics_overall.csv")
    f = read(root, "metrics_by_fold.csv")
    table(root, o[["model_label", "mae", "rmse", "smape", "cpc"]], "Table_3_overall")
    a = o[
        [
            "model_label",
            "distance_flow_beta",
            "delta_beta",
            "gini",
            "delta_gini",
            "rho_origin",
            "rho_destination",
            "total_bias_pct",
        ]
    ].copy()
    a.loc[o.condition.eq("time_mean"), ["rho_origin", "rho_destination"]] = np.nan
    table(root, a, "Table_4_aggregate")
    table(root, read(root, "selected_penalties.csv"), "Table_S1_poly2_penalties")
    old = pd.read_csv(root / "inputs/audit/readout_selection_audited.csv")
    selected = old[["condition", "outer_fold", "audited_selected_alpha"]].rename(
        columns={"audited_selected_alpha": "selected_alpha"}
    )
    new = read(root, "selected_penalties.csv").rename(columns={"alpha": "selected_alpha"})
    new["condition"] = NEW
    ps = (
        pd.concat([selected, new[["condition", "outer_fold", "selected_alpha"]]])
        .pivot(index="outer_fold", columns="condition", values="selected_alpha")
        .rename(columns=LABELS)
        .reset_index()
    )
    ps["outer_fold"] += 1
    table(root, ps.rename(columns={"outer_fold": "Fold"}), "Table_S1_all_ridge_penalties")
    ft = f[["model_label", "outer_fold", "mae", "rmse", "smape", "cpc"]].copy()
    ft["outer_fold"] += 1
    table(root, ft.rename(columns={"outer_fold": "Fold"}), "Table_S2_by_fold")
    for metric in ["total_bias_pct", "mae", "smape"]:
        table(
            root,
            g[g.group_type == "flow"]
            .pivot(index="group", columns="condition", values=metric)
            .reindex([f"D{i}" for i in range(1, 11)])[MODELS]
            .rename(columns=LABELS)
            .reset_index(),
            "Table_S5_" + metric,
        )
    dt = g[g.group_type == "distance"][["group", "condition", "mae", "smape"]].copy()
    dt["condition"] = dt.condition.map(LABELS)
    table(root, dt, "Table_S7_distance")
    bins = pd.read_csv(root / "inputs/bins/flow_bin_definitions.csv")
    share = g[(g.group_type == "flow") & (g.condition == NEW)][
        ["group", "y_sum", "observed_flow_share"]
    ]
    fb = bins.merge(share, left_on="flow_bin", right_on="group").drop(columns=["group", "y_sum"])
    fb["Flow share (%)"] = 100 * fb.pop("observed_flow_share")
    table(root, fb, "Table_S4_flow_bins")
    table(
        root,
        pd.read_csv(root / "inputs/bins/distance_bin_definitions.csv"),
        "Table_S6_distance_bins",
    )
    dg = g[g.group_type == "distance"]
    a = dg[dg.condition == "name_llm"]
    b = dg[dg.condition == "coordinate_llm"]
    diff = a[["group", "n_rows", "mae"]].merge(
        b[["group", "mae"]], on="group", suffixes=("_name", "_coordinate")
    )
    diff["weighted_mae_difference"] = diff.n_rows * (diff.mae_name - diff.mae_coordinate)
    diff["contribution_pct"] = (
        100 * diff.weighted_mae_difference / diff.weighted_mae_difference.sum()
    )
    table(root, diff, "Table_S8_mae_decomposition")
    ds = read(root, "district_summary.csv")
    ds["condition"] = ds.condition.map(LABELS)
    table(
        root,
        ds[
            [
                "condition",
                "side",
                "n_districts",
                "abs_total_bias_pct_median",
                "abs_total_bias_pct_q1",
                "abs_total_bias_pct_q3",
                "tv_median",
                "tv_q1",
                "tv_q3",
            ]
        ],
        "Table_S9_district",
    )
    comparisons = read(root, "llm_vs_poly2_comparisons.csv")
    counts = (
        comparisons.groupby(["level", "group_type", "llm", "metric"], dropna=False)
        .agg(n=("llm_better", "size"), llm_better=("llm_better", "sum"))
        .reset_index()
    )
    table(root, counts, "comparison_counts")
    hg = comparisons[(comparisons.level == "groups") & (comparisons.group_type == "distance_hour")]
    table(
        root,
        hg.groupby(["llm", "metric"])
        .agg(n=("llm_better", "size"), llm_better=("llm_better", "sum"))
        .reset_index(),
        "Table_S10_new_comparisons",
    )
    rows = []
    h = g[g.group_type == "distance_hour"]
    for a, b in [
        ("coordinate_llm", "name_llm"),
        ("coordinate_llm", "corrected_loglinear_gravity"),
        ("coordinate_llm", NEW),
        ("name_llm", NEW),
    ]:
        j = h[h.condition == a].merge(
            h[h.condition == b], on=["group", "hour"], suffixes=("_a", "_b")
        )
        for scope, sub in [
            ("All distance–hour groups", j),
            ("Longest-distance group (Q5)", j[j.group == "Q5"]),
        ]:
            for metric in ["mae", "smape"]:
                rows.append(
                    dict(
                        first_method=LABELS[a],
                        second_method=LABELS[b],
                        scope=scope,
                        metric=metric,
                        n=len(sub),
                        first_lower=int((sub[metric + "_a"] < sub[metric + "_b"]).sum()),
                        ties=int((sub[metric + "_a"] == sub[metric + "_b"]).sum()),
                    )
                )
    table(root, pd.DataFrame(rows), "Table_S10_all_comparisons")
    workflow(root)
    profiles(root, g)
    districts(root, d)
    supplements(root, g, d)
    district_heatmap_publication(root)
    (root / "results/figure_export_checks.json").write_text(json.dumps(QA, indent=2))
    print("Exported six PDF/600-dpi PNG figure pairs and tables")
