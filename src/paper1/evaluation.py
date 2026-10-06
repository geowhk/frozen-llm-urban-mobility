"""Evaluation separated from training; fixed OOF and saved group boundaries."""

from pathlib import Path
import json, sys, platform
import numpy as np
import pandas as pd
import scipy, sklearn
from . import core
from . import diagnostics as ld
from .config import NEW, LABELS, MODELS
from .validation import record_check
from functools import partial


def evaluate_predictions(root):
    """Results: reproduce all seven conditions from a fixed OOF file; never fit models."""
    root = Path(root)
    R = root / "results"
    I = root / "inputs"
    df = pd.read_parquet(R / "oof_predictions_all_models.parquet")
    CHECKS = []
    check = partial(record_check, CHECKS)
    overall = []
    byfold = []
    for model, col in MODELS.items():
        for f, part in [(None, df), *list(df.groupby("outer_fold"))]:
            frame = part.copy()
            frame["y_hat"] = frame[col]
            m = core.metrics(frame)
            m.update(
                condition=model,
                model_label=LABELS[model],
                total_bias_pct=100 * (part[col].sum() / part.y_gt.sum() - 1),
                distance_flow_beta=-float(
                    np.polyfit(np.log(part.dist_km), np.log1p(part[col]), 1)[0]
                ),
                gini=core.gini(part[col].to_numpy()),
            )
            if f is None:
                overall.append(m)
            else:
                byfold.append(dict(outer_fold=int(f), **m))
    ov = pd.DataFrame(overall)
    ov.to_csv(R / "metrics_overall.csv", index=False)
    pd.DataFrame(byfold).to_csv(R / "metrics_by_fold.csv", index=False)
    ref = pd.read_csv(I / "audit/metrics_overall_audited.csv").set_index("condition")
    for model in ref.index:
        for metric in [
            "mae",
            "rmse",
            "smape",
            "cpc",
            "delta_beta",
            "delta_gini",
            "rho_origin",
            "rho_destination",
        ]:
            if model == "time_mean" and metric.startswith("rho"):
                continue
            value = ov.set_index("condition").loc[model, metric]
            check(
                "audit_all_metrics_" + model + "_" + metric,
                np.isclose(value, ref.loc[model, metric], rtol=1e-10, atol=1e-10),
            )
    pd.DataFrame(
        [
            dict(
                distance_flow_beta=-np.polyfit(np.log(df.dist_km), np.log1p(df.y_gt), 1)[0],
                gini=core.gini(df.y_gt),
                total_flow=df.y_gt.sum(),
            )
        ]
    ).to_csv(R / "mobility_reference_metrics.csv", index=False)
    flow = pd.read_csv(I / "bins/flow_bin_definitions.csv")
    dist = pd.read_csv(I / "bins/distance_bin_definitions.csv")
    for tag, defs, lower, upper, value, key in [
        ("flow", flow, "lower_edge", "upper_edge", "y_gt", "flow_bin"),
        ("distance", dist, "lower_edge_km", "upper_edge_km", "dist_km", "distance_bin"),
    ]:
        edges = np.r_[defs[lower].iloc[0], defs[upper].to_numpy()]
        df[tag + "_bin"] = pd.cut(
            df[value], edges, labels=defs[key].tolist(), include_lowest=True, right=True
        ).astype(str)
        check(tag + "_bins_complete", not df[tag + "_bin"].eq("nan").any())
        counts = df.groupby(tag + "_bin").size().reindex(defs[key]).to_numpy()
        check(tag + "_counts_preserved", np.array_equal(counts, defs.n_rows))
    groups = []
    for typ, cols in [
        ("flow", ["flow_bin"]),
        ("distance", ["distance_bin"]),
        ("distance_hour", ["distance_bin", "hour"]),
    ]:
        for k, part in df.groupby(cols, sort=True):
            kval = k if isinstance(k, tuple) else (k,)
            for model, col in MODELS.items():
                groups.append(
                    dict(
                        group_type=typ,
                        group=kval[0],
                        hour=int(kval[1]) if len(kval) > 1 else -1,
                        condition=model,
                        n_rows=len(part),
                        n_dyads=part.dyad_id.nunique(),
                        y_sum=part.y_gt.sum(),
                        pred_sum=part[col].sum(),
                        observed_flow_share=part.y_gt.sum() / df.y_gt.sum(),
                        total_bias_pct=100 * (part[col].sum() / part.y_gt.sum() - 1),
                        **ld.metrics(part.y_gt.to_numpy(), part[col].to_numpy()),
                    )
                )
    gm = pd.DataFrame(groups)
    gm.to_csv(R / "group_metrics.csv", index=False)
    ld.analysis_c(df, R)
    dd = pd.read_csv(R / "district_allocation_diagnostics.csv")
    dd.to_csv(R / "district_diagnostics.csv", index=False)
    oldd = pd.read_csv(I / "prior_diagnostics/district_allocation_diagnostics.csv")
    md = oldd.merge(
        dd,
        on=["condition", "side", "district_code"],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    check("old_district_rows_preserved", len(md) == 300)
    for metric in [
        "reference_total",
        "predicted_total",
        "total_bias_pct",
        "abs_total_bias_pct",
        "tv",
    ]:
        check(
            "old_district_" + metric,
            np.allclose(md[metric + "_old"], md[metric + "_new"], rtol=1e-10, atol=1e-08),
        )
    summary = []
    for (m, s), p in dd.groupby(["condition", "side"]):
        row = dict(condition=m, side=s, n_districts=len(p))
        for metric in ["abs_total_bias_pct", "total_bias_pct", "tv"]:
            for q, v in zip(["q1", "median", "q3"], p[metric].quantile([0.25, 0.5, 0.75])):
                row[metric + "_" + q] = v
        summary.append(row)
    pd.DataFrame(summary).to_csv(R / "district_summary.csv", index=False)
    for oldfile, typ, key in [
        ("flow_scale_diagnostics.csv", "flow", "flow_decile"),
        ("distance_bin_metrics.csv", "distance", "distance_bin"),
        ("distance_hour_metrics.csv", "distance_hour", "distance_bin"),
    ]:
        old = pd.read_csv(I / "prior_diagnostics" / oldfile).rename(columns={key: "group"})
        join = ["condition", "group"] + (["hour"] if typ == "distance_hour" else [])
        merged = old.merge(
            gm[gm.group_type == typ], on=join, suffixes=("_old", "_new"), validate="one_to_one"
        )
        check("old_" + typ + "_row_count", len(merged) == len(old))
        for metric in ["mae", "rmse", "smape", "cpc"]:
            check(
                "old_" + typ + "_" + metric,
                np.allclose(
                    merged[metric + "_old"], merged[metric + "_new"], rtol=1e-10, atol=1e-08
                ),
            )
    comparisons = []
    frames = [
        ("overall", ov, ["condition"]),
        ("fold", pd.DataFrame(byfold), ["condition", "outer_fold"]),
        ("groups", gm, ["condition", "group_type", "group", "hour"]),
        ("district", dd, ["condition", "side", "district_code"]),
    ]
    for level, frame, keys_ in frames:
        keys_ = [k for k in keys_ if k != "condition"]
        base = frame[frame.condition == NEW]
        for llm in ["name_llm", "coordinate_llm"]:
            left = frame[frame.condition == llm]
            j = (
                left.merge(base, on=keys_, suffixes=("_llm", "_poly"))
                if keys_
                else left.assign(_k=1).merge(base.assign(_k=1), on="_k", suffixes=("_llm", "_poly"))
            )
            metrics = (
                ["mae", "rmse", "smape", "cpc"]
                if level != "district"
                else ["abs_total_bias_pct", "tv"]
            )
            for _, r in j.iterrows():
                for metric in metrics:
                    a, b = (float(r[metric + "_llm"]), float(r[metric + "_poly"]))
                    comparisons.append(
                        dict(
                            level=level,
                            llm=llm,
                            metric=metric,
                            **{k: r[k] for k in keys_},
                            llm_value=a,
                            poly2_value=b,
                            llm_minus_poly2=a - b,
                            llm_better=a > b if metric == "cpc" else a < b,
                            relative_error_reduction_pct=(
                                100 * (b - a) / b if metric != "cpc" and b else np.nan
                            ),
                        )
                    )
    pd.DataFrame(comparisons).to_csv(R / "llm_vs_poly2_comparisons.csv", index=False)
    df[["query_id", "dyad_id", "outer_fold", "flow_bin", "distance_bin"]].to_csv(
        R / "bin_assignments.csv", index=False
    )
    check("all_seven_predictions_finite", np.isfinite(df[list(MODELS.values())]).all().all())
    check("district_tv_finite", np.isfinite(dd.tv).all())
    pd.DataFrame(CHECKS).to_csv(R / "validation_checks.csv", index=False)
    (root / "environment.txt").write_text(
        "\n".join(
            [
                sys.version,
                platform.platform(),
                f"numpy=={np.__version__}",
                f"pandas=={pd.__version__}",
                f"scipy=={scipy.__version__}",
                f"scikit-learn=={sklearn.__version__}",
            ]
        )
    )
    print(ov[["condition", "mae", "rmse", "smape", "cpc"]].to_string(index=False))
