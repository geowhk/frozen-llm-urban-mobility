"""Audited observation and district diagnostics; no model fitting."""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .config import MODELS, LABELS as MODEL_LABELS, BASE_MODELS

RTOL = 1e-07
ATOL = 1e-08


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = p - y
    smape_terms = scaled_absolute_error(y, p)
    total = float(np.sum(y) + np.sum(p) + 1e-09)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "smape": float(np.mean(smape_terms)),
        "cpc": float(2.0 * np.minimum(y, p).sum() / total),
    }


def scaled_absolute_error(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Observation-level symmetric absolute error specified for this analysis."""
    return 2.0 * np.abs(p - y) / (p + y + 1e-09)


def require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def compare_metrics(reproduced: pd.DataFrame, reference: pd.DataFrame, keys: list[str]):
    metric_cols = ["mae", "rmse", "smape", "cpc"]
    merged = reproduced.merge(
        reference, on=keys, how="outer", suffixes=("", "_audit"), indicator=True
    )
    require((merged["_merge"] == "both").all(), f"Metric key mismatch on {keys}")
    passes = []
    for col in metric_cols:
        merged[f"{col}_diff"] = merged[col] - merged[f"{col}_audit"]
        merged[f"{col}_pass"] = np.isclose(
            merged[col], merged[f"{col}_audit"], rtol=RTOL, atol=ATOL, equal_nan=True
        )
        passes.append(merged[f"{col}_pass"])
    merged["all_metrics_pass"] = np.logical_and.reduce(passes)
    require(bool(merged["all_metrics_pass"].all()), f"Metric reproduction failed on {keys}")
    return merged.drop(columns="_merge")


def validate_inputs(
    df: pd.DataFrame, gt: pd.DataFrame, audit_dir: Path, tables: Path, models=BASE_MODELS
):
    required = ["query_id", "orig", "dest", "dyad_id", "hour", "outer_fold", "y_gt", "dist_km"]
    required += list(models.values())
    missing = sorted(set(required) - set(df.columns))
    require(not missing, f"Missing columns: {missing}")
    checks = {}
    checks["rows_14400"] = len(df) == 14400
    checks["query_id_unique"] = df["query_id"].nunique() == 14400
    checks["od_hour_unique"] = not df.duplicated(["orig", "dest", "hour"]).any()
    checks["districts_25"] = len(set(df["orig"]) | set(df["dest"])) == 25
    checks["directed_od_600"] = df[["orig", "dest"]].drop_duplicates().shape[0] == 600
    checks["each_directed_od_has_24_hours"] = df.groupby(["orig", "dest"]).size().eq(24).all()
    checks["dyads_300"] = df["dyad_id"].nunique() == 300
    checks["hours_24"] = set(df["hour"].astype(int)) == set(range(24))
    checks["no_self_flows"] = bool((df["orig"] != df["dest"]).all())
    checks["folds_5"] = set(df["outer_fold"].astype(int)) == set(range(5))
    checks["fold_rows_2880"] = df.groupby("outer_fold").size().eq(2880).all()
    checks["fold_dyads_60"] = df.groupby("outer_fold")["dyad_id"].nunique().eq(60).all()
    dyad = df.groupby("dyad_id").agg(
        rows=("query_id", "size"),
        directions=("query_id", lambda _: 0),
        hours=("hour", "nunique"),
        folds=("outer_fold", "nunique"),
        distances=("dist_km", "nunique"),
    )
    direction_counts = df.groupby("dyad_id").apply(
        lambda x: x[["orig", "dest"]].drop_duplicates().shape[0], include_groups=False
    )
    dyad["directions"] = direction_counts
    checks["dyad_48_rows"] = dyad["rows"].eq(48).all()
    checks["dyad_2_directions"] = dyad["directions"].eq(2).all()
    checks["dyad_24_hours"] = dyad["hours"].eq(24).all()
    checks["dyad_single_fold"] = dyad["folds"].eq(1).all()
    checks["dyad_single_distance"] = dyad["distances"].eq(1).all()
    numeric = ["y_gt", "dist_km"] + list(models.values())
    checks["all_numeric_finite"] = bool(np.isfinite(df[numeric].to_numpy(float)).all())
    checks["one_prediction_per_model_per_row"] = bool(df[list(models.values())].notna().all().all())
    checks["all_flows_nonnegative"] = bool((df[["y_gt"] + list(models.values())] >= 0).all().all())
    checks["distance_positive"] = bool((df["dist_km"] > 0).all())
    if {"observed_y", "distance_km"}.issubset(df.columns):
        checks["duplicate_y_columns_equal"] = bool(
            np.allclose(df["y_gt"], df["observed_y"], rtol=0, atol=0)
        )
        checks["duplicate_distance_columns_equal"] = bool(
            np.allclose(df["dist_km"], df["distance_km"], rtol=0, atol=0)
        )
    gt_keyed = gt.rename(columns={"arrival_hour": "hour", "flow": "flow_gt_source"})
    joined = df.merge(
        gt_keyed[["orig", "dest", "hour", "flow_gt_source", "dist_km"]],
        on=["orig", "dest", "hour"],
        how="left",
        suffixes=("", "_gt_source"),
        validate="one_to_one",
    )
    checks["ground_truth_join_complete"] = bool(joined["flow_gt_source"].notna().all())
    checks["ground_truth_flow_equal"] = bool(
        np.allclose(joined["y_gt"], joined["flow_gt_source"], rtol=0, atol=1e-12)
    )
    checks["ground_truth_distance_equal"] = bool(
        np.allclose(joined["dist_km"], joined["dist_km_gt_source"], rtol=0, atol=1e-12)
    )
    require(
        all(checks.values()), f"Input validation failed: {[k for k, v in checks.items() if not v]}"
    )
    overall_rows = []
    fold_rows = []
    y = df["y_gt"].to_numpy(float)
    for condition, col in models.items():
        overall_rows.append({"condition": condition, **metrics(y, df[col].to_numpy(float))})
        for fold, part in df.groupby("outer_fold", sort=True):
            fold_rows.append(
                {
                    "condition": condition,
                    "outer_fold": int(fold),
                    **metrics(part["y_gt"].to_numpy(float), part[col].to_numpy(float)),
                }
            )
    overall = pd.DataFrame(overall_rows)
    by_fold = pd.DataFrame(fold_rows)
    overall_ref = pd.read_csv(audit_dir / "metrics_overall_audited.csv")
    fold_ref = pd.read_csv(audit_dir / "metrics_by_fold_cell_audited.csv")
    overall_check = compare_metrics(overall, overall_ref, ["condition"])
    fold_check = compare_metrics(by_fold, fold_ref, ["condition", "outer_fold"])
    overall_check.to_csv(tables / "reproduced_overall_metrics.csv", index=False)
    fold_check.to_csv(tables / "reproduced_fold_metrics.csv", index=False)
    checks["overall_metrics_reproduced"] = bool(overall_check["all_metrics_pass"].all())
    checks["fold_metrics_reproduced"] = bool(fold_check["all_metrics_pass"].all())
    return checks


def total_variation(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() <= 0 or p.sum() <= 0:
        return np.nan
    return float(0.5 * np.abs(y / y.sum() - p / p.sum()).sum())


def analysis_c(df: pd.DataFrame, tables: Path, models=MODELS):
    agg_cols = {"y_gt": "sum", **{col: "sum" for col in models.values()}}
    daily = df.groupby(["orig", "dest"], as_index=False).agg(agg_cols)
    require(len(daily) == 600, "Daily OD aggregation did not produce 600 directed pairs")
    require(
        np.isclose(total_variation(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 1.0),
        "TV test failed",
    )
    require(
        np.isclose(total_variation(np.array([1.0, 1.0]), np.array([2.0, 2.0])), 0.0),
        "TV test failed",
    )
    require(
        np.isclose(total_variation(np.array([1.0, 3.0]), np.array([2.0, 6.0])), 0.0),
        "TV scale test failed",
    )
    rows = []
    for side, key in [("origin", "orig"), ("destination", "dest")]:
        for district, part in daily.groupby(key, sort=True):
            y = part["y_gt"].to_numpy(float)
            ref_total = float(y.sum())
            for condition, col in models.items():
                p = part[col].to_numpy(float)
                pred_total = float(p.sum())
                bias_pct = 100.0 * (pred_total - ref_total) / ref_total if ref_total > 0 else np.nan
                tv_value = total_variation(y, p)
                if np.isfinite(tv_value):
                    require(-ATOL <= tv_value <= 1 + ATOL, "TV outside [0,1]")
                    require(
                        np.isclose((y / y.sum()).sum(), 1.0, rtol=RTOL, atol=ATOL),
                        "Observed shares do not sum to one",
                    )
                    require(
                        np.isclose((p / p.sum()).sum(), 1.0, rtol=RTOL, atol=ATOL),
                        "Predicted shares do not sum to one",
                    )
                rows.append(
                    {
                        "side": side,
                        "district_code": str(district),
                        "condition": condition,
                        "model_label": MODEL_LABELS[condition],
                        "n_counterparts": len(part),
                        "reference_total": ref_total,
                        "predicted_total": pred_total,
                        "total_bias_pct": bias_pct,
                        "abs_total_bias_pct": abs(bias_pct) if np.isfinite(bias_pct) else np.nan,
                        "tv": tv_value,
                        "tv_na_reason": (
                            ""
                            if np.isfinite(tv_value)
                            else (
                                "mobility total is zero"
                                if ref_total <= 0
                                else "predicted total is zero"
                            )
                        ),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(tables / "district_allocation_diagnostics.csv", index=False)
    summary_rows = []
    for (side, condition), part in out.groupby(["side", "condition"], sort=True):
        valid = part["tv"].notna()
        weights = part.loc[valid, "reference_total"]
        weighted_tv = (
            float(np.average(part.loc[valid, "tv"], weights=weights)) if valid.any() else np.nan
        )
        summary_rows.append(
            {
                "side": side,
                "condition": condition,
                "model_label": MODEL_LABELS[condition],
                "n_districts": len(part),
                "n_tv_valid": int(valid.sum()),
                "n_tv_na": int((~valid).sum()),
                "tv_median": float(part["tv"].median()),
                "tv_q1": float(part["tv"].quantile(0.25)),
                "tv_q3": float(part["tv"].quantile(0.75)),
                "tv_mean_unweighted": float(part["tv"].mean()),
                "tv_mean_flow_weighted": weighted_tv,
                "median_abs_total_bias_pct": float(part["abs_total_bias_pct"].median()),
                "n_underpredicted": int((part["total_bias_pct"] < -1e-12).sum()),
                "n_overpredicted": int((part["total_bias_pct"] > 1e-12).sum()),
                "n_equal": int((part["total_bias_pct"].abs() <= 1e-12).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables / "district_allocation_summary.csv", index=False)
    overall_bias = {
        condition: 100.0 * (df[col].sum() - df["y_gt"].sum()) / df["y_gt"].sum()
        for condition, col in models.items()
    }
    check_rows = []
    for (side, condition), part in out.groupby(["side", "condition"], sort=True):
        weighted = (
            100.0
            * (part["predicted_total"].sum() - part["reference_total"].sum())
            / part["reference_total"].sum()
        )
        passed = bool(np.isclose(weighted, overall_bias[condition], rtol=RTOL, atol=ATOL))
        check_rows.append(
            {
                "side": side,
                "condition": condition,
                "overall_total_bias_pct": overall_bias[condition],
                "district_aggregated_total_bias_pct": weighted,
                "difference": weighted - overall_bias[condition],
                "pass": passed,
            }
        )
    checks = pd.DataFrame(check_rows)
    require(bool(checks["pass"].all()), "District totals do not reproduce overall total bias")
    checks.to_csv(tables / "district_total_bias_consistency_checks.csv", index=False)
    return summary
