from __future__ import annotations
import argparse
import hashlib
import inspect
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow
import scipy
from scipy.sparse.linalg import lsqr
from .core import (
    EXPECTED_CACHE_SHA256,
    EXPECTED_GT_SHA256,
    canonical_data,
    fit_standardizer,
    gini,
    metrics,
    sha256_file,
    transform_standardized,
)
from .splits import (
    add_dyads,
    categorical_role_hour_matrix,
    metric_components,
    require,
    validate_strict_data,
    validate_strict_splits,
)

CONDITIONS = [
    "name_llm",
    "coordinate_llm",
    "time_mean",
    "categorical_role_hour",
    "corrected_marginal_product",
    "corrected_loglinear_gravity",
]
RIDGE_CONDITIONS = ["name_llm", "coordinate_llm", "categorical_role_hour"]
ORIGINAL_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
CATEGORICAL_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
EXTRA_CATEGORICAL = [0.0001, 0.001, 10000.0, 100000.0]
CELL_METRICS = ["mae", "rmse", "smape", "cpc"]
ALL_METRICS = [
    "mae",
    "rmse",
    "smape",
    "cpc",
    "delta_beta",
    "rho_origin",
    "rho_destination",
    "delta_gini",
]
RTOL = 1e-07
ATOL = 1e-08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Urban mobility 최소 수치 감사")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--archived-final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_ids(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def rel_difference(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(a), 1e-30)
    return float(np.max(np.abs(a - b) / denom))


def git_info(root: Path) -> dict[str, object]:

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                list(args), cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "not recorded"

    status = run("git", "status", "--short")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(status and status != "not recorded"),
        "status_short": status.splitlines(),
    }


def environment_info() -> dict[str, object]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_config()
    signature = inspect.signature(lsqr)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "lsqr_signature": str(signature),
        "lsqr_default_conlim": signature.parameters["conlim"].default,
        "blas_configuration": buf.getvalue(),
        "thread_environment": {
            key: os.environ.get(key, "unset")
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def fit_lsqr(
    z: np.ndarray,
    centered_target: np.ndarray,
    alpha: float,
    atol: float,
    btol: float,
    iter_lim: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    started = time.monotonic()
    result = lsqr(
        z,
        centered_target,
        damp=float(alpha) ** 0.5,
        atol=atol,
        btol=btol,
        iter_lim=iter_lim,
        show=False,
    )
    coef = result[0]
    residual = z @ coef - centered_target
    gradient = z.T @ residual + float(alpha) * coef
    denominator = (
        np.linalg.norm(z.T @ centered_target) + float(alpha) * np.linalg.norm(coef) + 1e-30
    )
    diag: dict[str, float | int] = {
        "alpha": float(alpha),
        "damp": float(alpha) ** 0.5,
        "atol": atol,
        "btol": btol,
        "conlim": float(inspect.signature(lsqr).parameters["conlim"].default),
        "iter_lim": iter_lim,
        "istop": int(result[1]),
        "itn": int(result[2]),
        "r1norm": float(result[3]),
        "r2norm": float(result[4]),
        "anorm": float(result[5]),
        "acond": float(result[6]),
        "arnorm": float(result[7]),
        "xnorm": float(result[8]),
        "objective": float(residual @ residual + float(alpha) * (coef @ coef)),
        "gradient_l2": float(np.linalg.norm(gradient)),
        "stationarity_rel": float(np.linalg.norm(gradient) / denominator),
        "coef_all_finite": bool(np.isfinite(coef).all()),
        "elapsed_seconds": time.monotonic() - started,
    }
    return (coef, diag)


def needs_strict(diag: dict[str, float | int]) -> bool:
    return (
        int(diag["istop"]) in {3, 6, 7}
        or not bool(diag["coef_all_finite"])
        or float(diag["stationarity_rel"]) > 1e-08
    )


def fit_with_diagnostics(
    z: np.ndarray,
    centered_target: np.ndarray,
    alpha: float,
    common: dict[str, object],
    diagnostics: list[dict[str, object]],
) -> tuple[np.ndarray, str]:
    primary_id = (
        f"{common['condition']}-f{common['outer_fold']}-{common['stage']}-a{alpha:g}-primary"
    )
    coef, diag = fit_lsqr(z, centered_target, alpha, 1e-08, 1e-08, 2000)
    row = {**common, "diagnostic_id": primary_id, "setting": "primary", **diag}
    diagnostics.append(row)
    if not needs_strict(diag):
        return (coef, primary_id)
    strict_id = f"{common['condition']}-f{common['outer_fold']}-{common['stage']}-a{alpha:g}-strict"
    strict_coef, strict_diag = fit_lsqr(z, centered_target, alpha, 1e-10, 1e-10, 10000)
    diagnostics.append(
        {**common, "diagnostic_id": strict_id, "setting": "strict_recheck", **strict_diag}
    )
    if needs_strict(strict_diag):
        raise RuntimeError(f"NUMERICAL_UNRESOLVED: {strict_id}")
    return (strict_coef, strict_id)


def prepare_xy(
    features: np.ndarray, train_mask: np.ndarray, validation_mask: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    mean, scale = fit_standardizer(features[train_mask])
    train_z = transform_standardized(features[train_mask], mean, scale)
    validation_z = transform_standardized(features[validation_mask], mean, scale)
    train_y_log = np.log1p(target[train_mask])
    validation_y_log = np.log1p(target[validation_mask])
    zero_count = int(
        np.count_nonzero(np.asarray(features[train_mask], dtype=np.float64).std(axis=0) == 0.0)
    )
    return (
        train_z,
        validation_z,
        train_y_log - train_y_log.mean(),
        validation_y_log,
        float(train_y_log.mean()),
        zero_count,
    )


def select(candidates: list[dict[str, object]]) -> dict[str, object]:
    return min(candidates, key=lambda row: (float(row["validation_log_mse"]), -float(row["alpha"])))


def load_feature(condition: str, paper_root: Path, categorical: np.ndarray) -> np.ndarray:
    if condition == "name_llm":
        return np.load(
            paper_root / "analysis_cache/original/forward_lasttoken_layer31.npy", mmap_mode="r"
        )
    if condition == "coordinate_llm":
        return np.load(
            paper_root / "analysis_cache/geometry/forward_lasttoken_layer31.npy", mmap_mode="r"
        )
    return categorical


def fit_condition_fold(
    condition: str,
    fold: int,
    features: np.ndarray,
    data: pd.DataFrame,
    inner: pd.DataFrame,
    archived_candidates: pd.DataFrame,
    diagnostics: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]], dict[str, object], pd.DataFrame, pd.DataFrame, dict[str, object]
]:
    fold_inner = inner[inner["outer_fold"].eq(fold)]
    inner_train_ids = set(fold_inner.loc[fold_inner["inner_subset"].eq("train"), "dyad_id"])
    validation_ids = set(fold_inner.loc[fold_inner["inner_subset"].eq("validation"), "dyad_id"])
    inner_train_mask = data["dyad_id"].isin(inner_train_ids).to_numpy()
    validation_mask = data["dyad_id"].isin(validation_ids).to_numpy()
    outer_train_mask = data["outer_fold"].ne(fold).to_numpy()
    test_mask = data["outer_fold"].eq(fold).to_numpy()
    require(
        inner_train_mask.sum() == 10080 and validation_mask.sum() == 1440,
        "inner row count mismatch",
    )
    require(outer_train_mask.sum() == 11520 and test_mask.sum() == 2880, "outer row count mismatch")
    target = data["y_gt"].to_numpy(float)
    train_z, validation_z, centered_y, validation_y_log, intercept, zero_count = prepare_xy(
        features, inner_train_mask, validation_mask, target
    )
    common_base = {
        "run_id": "urban_mobility_numerical_audit_v1",
        "condition": condition,
        "outer_fold": fold,
        "n_features": train_z.shape[1],
        "dtype": str(train_z.dtype),
        "train_row_id_sha256": hash_ids(data.loc[inner_train_mask, "query_id"]),
        "feature_transform": "inner-train mean/std ddof=0; zero scale->1",
        "target_transform": "log1p then inner-train mean centering only",
        "target_mean": intercept,
        "zero_variance_columns": zero_count,
        "n_train": int(inner_train_mask.sum()),
    }
    candidate_rows: list[dict[str, object]] = []
    coefficient_by_alpha: dict[float, np.ndarray] = {}
    original_rows: list[dict[str, object]] = []
    for alpha in ORIGINAL_GRID:
        common = {**common_base, "stage": "inner_candidate", "grid_id": "original"}
        coef, diagnostic_id = fit_with_diagnostics(train_z, centered_y, alpha, common, diagnostics)
        coefficient_by_alpha[alpha] = coef
        pred_log = validation_z @ coef + intercept
        loss = float(np.mean((pred_log - validation_y_log) ** 2))
        archived_match = archived_candidates[
            archived_candidates["condition"].eq(condition)
            & archived_candidates["outer_fold"].eq(fold)
            & np.isclose(archived_candidates["alpha"], alpha)
        ]
        require(
            len(archived_match) == 1,
            f"archived candidate missing: {condition} fold {fold} alpha {alpha}",
        )
        old_loss = float(archived_match.iloc[0]["validation_log_mse"])
        row = {
            "condition": condition,
            "outer_fold": fold,
            "grid_id": "original",
            "alpha": alpha,
            "validation_log_mse": loss,
            "archived_validation_log_mse": old_loss,
            "loss_abs_diff": abs(loss - old_loss),
            "loss_rel_diff": abs(loss - old_loss) / max(abs(old_loss), 1e-30),
            "diagnostic_id": diagnostic_id,
        }
        original_rows.append(row)
        candidate_rows.append(row)
    original_selected = select(original_rows)
    final_grid = ORIGINAL_GRID
    if condition == "categorical_role_hour":
        for alpha in EXTRA_CATEGORICAL:
            common = {**common_base, "stage": "inner_candidate", "grid_id": "categorical_extended"}
            coef, diagnostic_id = fit_with_diagnostics(
                train_z, centered_y, alpha, common, diagnostics
            )
            coefficient_by_alpha[alpha] = coef
            loss = float(np.mean((validation_z @ coef + intercept - validation_y_log) ** 2))
            candidate_rows.append(
                {
                    "condition": condition,
                    "outer_fold": fold,
                    "grid_id": "categorical_extended",
                    "alpha": alpha,
                    "validation_log_mse": loss,
                    "archived_validation_log_mse": np.nan,
                    "loss_abs_diff": np.nan,
                    "loss_rel_diff": np.nan,
                    "diagnostic_id": diagnostic_id,
                }
            )
        final_grid = CATEGORICAL_GRID
    final_candidates = [row for row in candidate_rows if float(row["alpha"]) in final_grid]
    audited_selected = select(final_candidates)
    mean, scale = fit_standardizer(features[outer_train_mask])
    outer_train_z = transform_standardized(features[outer_train_mask], mean, scale)
    test_z = transform_standardized(features[test_mask], mean, scale)
    outer_y_log = np.log1p(target[outer_train_mask])
    outer_intercept = float(outer_y_log.mean())
    outer_centered = outer_y_log - outer_intercept
    outer_common = {
        "run_id": "urban_mobility_numerical_audit_v1",
        "condition": condition,
        "outer_fold": fold,
        "stage": "outer_final",
        "grid_id": "original",
        "n_train": int(outer_train_mask.sum()),
        "n_features": outer_train_z.shape[1],
        "dtype": str(outer_train_z.dtype),
        "train_row_id_sha256": hash_ids(data.loc[outer_train_mask, "query_id"]),
        "feature_transform": "outer-train mean/std ddof=0; zero scale->1",
        "target_transform": "log1p then outer-train mean centering only",
        "target_mean": outer_intercept,
        "zero_variance_columns": int(
            np.count_nonzero(
                np.asarray(features[outer_train_mask], dtype=np.float64).std(axis=0) == 0.0
            )
        ),
    }
    original_coef, original_final_diag_id = fit_with_diagnostics(
        outer_train_z, outer_centered, float(original_selected["alpha"]), outer_common, diagnostics
    )
    original_prediction_log = test_z @ original_coef + outer_intercept
    original_prediction = np.clip(np.expm1(original_prediction_log), 0.0, None)
    final_coef = original_coef
    final_diag_id = original_final_diag_id
    if float(audited_selected["alpha"]) != float(original_selected["alpha"]):
        final_coef, final_diag_id = fit_with_diagnostics(
            outer_train_z,
            outer_centered,
            float(audited_selected["alpha"]),
            {**outer_common, "grid_id": "categorical_extended"},
            diagnostics,
        )
    prediction_log = test_z @ final_coef + outer_intercept
    prediction = np.clip(np.expm1(prediction_log), 0.0, None)
    require(np.isfinite(prediction).all() and (prediction >= 0).all(), "invalid audited prediction")
    prediction_frame = data.loc[
        test_mask, ["query_id", "orig", "dest", "dyad_id", "hour", "outer_fold", "y_gt", "dist_km"]
    ].copy()
    prediction_frame["y_hat"] = prediction
    prediction_frame["y_hat_log_raw"] = prediction_log
    prediction_frame["source_run_id"] = "urban_mobility_numerical_audit_v1"
    prediction_frame["prediction_version"] = "audited_final"
    solver_prediction_frame = data.loc[
        test_mask, ["query_id", "orig", "dest", "dyad_id", "hour", "outer_fold", "y_gt", "dist_km"]
    ].copy()
    solver_prediction_frame["y_hat"] = original_prediction
    solver_prediction_frame["y_hat_log_raw"] = original_prediction_log
    solver_prediction_frame["source_run_id"] = "urban_mobility_numerical_audit_v1"
    solver_prediction_frame["prediction_version"] = "solver_verified_original_grid"
    original_losses = sorted(original_rows, key=lambda row: float(row["alpha"]))
    old_alpha = float(
        archived_candidates[
            archived_candidates["condition"].eq(condition)
            & archived_candidates["outer_fold"].eq(fold)
            & archived_candidates["selected"].astype(bool)
        ].iloc[0]["alpha"]
    )
    selected_row = {
        "condition": condition,
        "outer_fold": fold,
        "original_selected_alpha_reproduced": float(original_selected["alpha"]),
        "archived_selected_alpha": old_alpha,
        "audited_selected_alpha": float(audited_selected["alpha"]),
        "validation_log_mse": float(audited_selected["validation_log_mse"]),
        "selection_changed_from_archived": float(audited_selected["alpha"]) != old_alpha,
        "selected_at_lower_boundary": float(audited_selected["alpha"]) == min(final_grid),
        "selected_at_upper_boundary": float(audited_selected["alpha"]) == max(final_grid),
        "grid_id": "categorical_extended" if condition == "categorical_role_hour" else "original",
        "final_diagnostic_id": final_diag_id,
        "original_final_diagnostic_id": original_final_diag_id,
        "neighbor_losses": json.dumps(
            [
                {"alpha": float(row["alpha"]), "loss": float(row["validation_log_mse"])}
                for row in sorted(final_candidates, key=lambda row: float(row["alpha"]))
            ],
            ensure_ascii=False,
        ),
    }
    reproduction = {
        "condition": condition,
        "outer_fold": fold,
        "entity": "candidate_losses",
        "n_values": len(original_losses),
        "max_abs_diff": max((float(row["loss_abs_diff"]) for row in original_losses)),
        "max_rel_diff": max((float(row["loss_rel_diff"]) for row in original_losses)),
        "allclose_rtol_1e-7_atol_1e-8": all(
            (
                np.isclose(
                    float(row["validation_log_mse"]),
                    float(row["archived_validation_log_mse"]),
                    rtol=RTOL,
                    atol=ATOL,
                )
                for row in original_losses
            )
        ),
    }
    return (candidate_rows, selected_row, solver_prediction_frame, prediction_frame, reproduction)


def wide_to_long(wide: pd.DataFrame, condition: str) -> pd.DataFrame:
    return wide[
        ["query_id", "orig", "dest", "dyad_id", "hour", "outer_fold", "y_gt", "dist_km"]
    ].assign(y_hat=wide[f"prediction_{condition}"].to_numpy(float))


def calculate_all_metrics(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall, by_fold, components = ([], [], [])
    for condition in CONDITIONS:
        frame = wide_to_long(wide, condition)
        overall.append({"condition": condition, **metrics(frame)})
        components.append({"condition": condition, **metric_components(frame)})
        for fold, block in frame.groupby("outer_fold"):
            result = metrics(block)
            by_fold.append(
                {
                    "condition": condition,
                    "outer_fold": int(fold),
                    **{key: result[key] for key in CELL_METRICS},
                }
            )
    overall_frame = pd.DataFrame(overall)
    overall_frame["marginal_rho_rank_eligible"] = overall_frame["condition"].ne("time_mean")
    return (overall_frame, pd.DataFrame(by_fold), pd.DataFrame(components))


def total_bias(wide: pd.DataFrame) -> pd.DataFrame:
    reference_total = float(wide["y_gt"].sum())
    rows = []
    for condition in CONDITIONS:
        predicted_total = float(wide[f"prediction_{condition}"].sum())
        signed = predicted_total - reference_total
        rows.append(
            {
                "condition": condition,
                "reference_total": reference_total,
                "predicted_total": predicted_total,
                "signed_total_bias": signed,
                "total_bias_pct": 100.0 * signed / reference_total if reference_total else np.nan,
                "total_ratio": predicted_total / reference_total if reference_total else np.nan,
                "origin_sum_matches": bool(
                    np.isclose(
                        predicted_total, wide.groupby("orig")[f"prediction_{condition}"].sum().sum()
                    )
                ),
                "destination_sum_matches": bool(
                    np.isclose(
                        predicted_total, wide.groupby("dest")[f"prediction_{condition}"].sum().sum()
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "NA" if pd.isna(value) else f"{value:.{digits}f}"
        )
    lines = [
        "| " + " | ".join(map(str, display.columns)) + " |",
        "|" + "|".join(["---"] * len(display.columns)) + "|",
    ]
    lines.extend(
        (
            "| " + " | ".join(map(str, row)) + " |"
            for row in display.itertuples(index=False, name=None)
        )
    )
    return "\n".join(lines)


def write_reports(
    output: Path,
    overall: pd.DataFrame,
    by_fold: pd.DataFrame,
    bias: pd.DataFrame,
    selection: pd.DataFrame,
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
    reproduction: pd.DataFrame,
) -> None:
    changed = selection[selection["selection_changed_from_archived"]]
    warnings = diagnostics[
        diagnostics["istop"].isin([3, 6, 7]) | (diagnostics["stationarity_rel"] > 1e-08)
    ]
    rounded_changes = comparison[
        comparison["stage_comparison"].eq("archived_to_audited_final")
        & comparison["rounded_3_changed"]
    ]
    impact = f"# 원고 수치·해석 영향표\n\n## alpha 선택 변화\n\n{markdown_table(selection[['condition', 'outer_fold', 'archived_selected_alpha', 'audited_selected_alpha', 'selection_changed_from_archived', 'selected_at_lower_boundary', 'selected_at_upper_boundary']], 6)}\n\n## 원고 표에서 세 자리 반올림 값이 바뀌는 항목\n\n{(markdown_table(rounded_changes[['condition', 'metric', 'old_value', 'new_value', 'difference']], 6) if len(rounded_changes) else '없음.')}\n\n## 반드시 바꿔야 할 해석\n\n- RQ3와 본문에서 `district totals`라고 넓게 표현한 부분은 실제 지표에 맞춰 `district-total rankings`로 제한해야 한다.\n- 전체 총량은 아래 표처럼 조건별로 과소 또는 과대예측된다. 이 값은 지역별 규모 정확도를 대신하지 않는다.\n- categorical alpha grid는 감사 후 확장 범위를 사용했으므로 세 ridge 조건이 동일한 alpha grid를 사용했다고 쓰면 안 된다.\n- 이 감사는 유의성 검정, bootstrap, repeated CV, smearing 또는 새 기준모형을 수행하지 않았다.\n\n## 그대로 유지 가능한 비교 방향\n\n비교 방향 변화 여부는 `comparison_to_archived.csv`의 `direction_changed`를 기준으로 확인한다. 숫자가 바뀌면 기존 문장에 새 숫자를 반영하되, 통계적 우월성으로 표현하지 않는다.\n"
    (output / "manuscript_impact.md").write_text(impact, encoding="utf-8")
    tables = f"# 집필용 최종 표\n\n## 전체 OOF 성능\n\n{markdown_table(overall[['condition', *ALL_METRICS]], 6)}\n\n## 전체 총량 편향\n\n{markdown_table(bias[['condition', 'reference_total', 'predicted_total', 'signed_total_bias', 'total_bias_pct', 'total_ratio']], 6)}\n\n## fold별 셀 지표\n\nsMAPE는 퍼센트가 아니라 비율이다. 다섯 fold는 독립 반복실험이나 신뢰구간으로 해석하지 않는다.\n\n{markdown_table(by_fold, 6)}\n\n## 감사 후 alpha 선택\n\n{markdown_table(selection[['condition', 'outer_fold', 'audited_selected_alpha', 'validation_log_mse', 'selected_at_lower_boundary', 'selected_at_upper_boundary']], 6)}\n"
    (output / "tables_for_manuscript.md").write_text(tables, encoding="utf-8")
    report = f"# Urban mobility 최소 수치 감사 보고서\n\n## Material Passport\n\n- Material ID: `urban_mobility_numerical_audit_v1`\n- Verification Status: `VERIFIED`\n- Source result: `strict_dyad_corrected_20260913T103227Z/final`\n- Scope: LSQR 진단, categorical alpha 사전지정 확장, 전체 총량 편향, fold별 셀 지표\n- Out of scope: 새 표현 추출, 새 기준모형, bootstrap, repeated CV, smearing, 추가 도시·기간·LLM·prompt\n\n## 완료 상태\n\n- 기존 candidate loss 재현 행: {int((reproduction['entity'] == 'candidate_losses').sum())}개 condition-fold\n- 실행된 LSQR 적합: {len(diagnostics)}회(엄격 재확인 포함)\n- 1차 진단 경고: {len(warnings)}회\n- alpha 선택 변화: {len(changed)}개 condition-fold\n- 원고 세 자리 수치 변화: {len(rounded_changes)}개 condition-metric\n\n## 수렴 판정\n\n종료코드, 반복 수, 잔차, 조건수, 직접 계산한 목적함수와 stationarity를 `solver_diagnostics.csv`에 기록했다. 3/6/7 종료, 비유한 계수 또는 stationarity_rel > 1e-8은 사전 규칙에 따라 엄격 재확인했다. 미해결 적합이 있으면 실행은 완료되지 않도록 구성했다.\n\n## categorical 경계 판정\n\n사전 지정된 `[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]`만 평가했다. 최종 경계 여부는 `readout_selection_audited.csv`에 기록했다. outer 성능은 alpha 선택에 사용하지 않았다.\n\n## 해석 범위\n\n전체 총량 편향은 연구에 포함한 서울 자치구 간 평균 평일 OD-hour 프로필에 관한 값이다. 월간 전체 이동이나 고유 이용자 수가 아니다. fold별 지표는 분할별 기술값이며 추론통계가 아니다.\n"
    (output / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    paper_root = args.data_root.resolve()
    archived = args.archived_final.resolve()
    output = args.output.resolve()
    if output.exists() and (not args.resume):
        raise FileExistsError(f"기존 결과 보호: output already exists: {output}")
    output.mkdir(parents=True, exist_ok=args.resume)
    (output / "code").mkdir(exist_ok=True)
    started = utc_now()
    gt_path = paper_root / "processed/gt_flow_gu_202411_weekday_daily.parquet"
    name_cache = paper_root / "analysis_cache/original"
    coordinate_cache = paper_root / "analysis_cache/geometry"
    outer_path = archived / "fold_assignments_strict_dyad.csv"
    inner_path = archived / "inner_split_assignments_strict_dyad.csv"
    archived_oof_path = archived / "oof_predictions.parquet"
    archived_candidates_path = archived / "readout_candidates.csv"
    required = [
        gt_path,
        outer_path,
        inner_path,
        archived_oof_path,
        archived / "metrics_overall.csv",
        archived / "metrics_by_fold.csv",
        archived_candidates_path,
        archived / "readout_selection.csv",
        archived / "run_manifest.json",
    ]
    for cache in (name_cache, coordinate_cache):
        required.extend(
            [
                cache / "forward_lasttoken_layer31.npy",
                cache / "forward_row_index.parquet",
                cache / "forward_cache_meta.json",
            ]
        )
    for path in required:
        require(path.is_file(), f"required input missing: {path}")
    require(sha256_file(gt_path) == EXPECTED_GT_SHA256, "ground truth SHA-256 mismatch")
    for label, cache in (("original", name_cache), ("geometry", coordinate_cache)):
        for filename, expected in EXPECTED_CACHE_SHA256[label].items():
            require(
                sha256_file(cache / filename) == expected, f"cache SHA mismatch: {label}/{filename}"
            )
    config = {
        "spec_version": "urban_mobility_numerical_audit_v1",
        "created_utc": started,
        "original_grid": ORIGINAL_GRID,
        "categorical_extended_grid": CATEGORICAL_GRID,
        "selection_metric": "inner validation log-scale MSE",
        "tie_rule": "min(validation_log_mse, -alpha)",
        "primary_solver": {"damp": "sqrt(alpha)", "atol": 1e-08, "btol": 1e-08, "iter_lim": 2000},
        "strict_recheck": {
            "trigger": "istop in 3/6/7 or nonfinite or stationarity_rel > 1e-8",
            "atol": 1e-10,
            "btol": 1e-10,
            "iter_lim": 10000,
        },
        "reproduction_tolerance": {"rtol": RTOL, "atol": ATOL},
        "split": "saved strict dyad outer/inner assignments",
        "scope_exclusions": [
            "matched baseline",
            "bootstrap",
            "repeated CV",
            "smearing",
            "new LLM extraction",
            "masking sensitivity",
        ],
    }
    (output / "audit_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    script_path = Path(__file__).resolve()
    code_paths = [
        Path(__file__).with_name("core.py"),
        Path(__file__).with_name("splits.py"),
        Path(__file__).with_name("audited_models.py"),
        script_path,
    ]
    manifest = {
        "created_utc": started,
        "inputs": {
            path.name + "::" + str(index): file_record(path) for index, path in enumerate(required)
        },
        "code": {path.name: file_record(path) for path in code_paths},
        "git": git_info(paper_root),
        "environment": environment_info(),
        "archived_result_id": archived.parent.name,
    }
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(script_path, output / "code" / script_path.name)
    data = add_dyads(canonical_data(gt_path, name_cache / "forward_row_index.parquet"))
    validate_strict_data(data)
    outer = pd.read_csv(outer_path, dtype={"dyad_id": str})
    inner = pd.read_csv(inner_path, dtype={"dyad_id": str})
    validate_strict_splits(data, outer, inner)
    data = data.merge(outer[["dyad_id", "outer_fold"]], on="dyad_id", validate="many_to_one")
    archived_oof = pd.read_parquet(archived_oof_path).sort_values("query_id").reset_index(drop=True)
    require(
        len(archived_oof) == 14400 and archived_oof["query_id"].nunique() == 14400,
        "archived OOF invalid",
    )
    require(
        archived_oof["query_id"].tolist() == data.sort_values("query_id")["query_id"].tolist(),
        "OOF query IDs mismatch",
    )
    archived_candidates = pd.read_csv(archived_candidates_path)
    require(len(archived_candidates) == 90, "archived candidate row count mismatch")
    categorical, _ = categorical_role_hour_matrix(data, sorted(map(str, set(data["orig"]))))
    diagnostics: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    reproduction_rows: list[dict[str, object]] = []
    solver_predictions: dict[str, list[pd.DataFrame]] = {key: [] for key in RIDGE_CONDITIONS}
    new_predictions: dict[str, list[pd.DataFrame]] = {key: [] for key in RIDGE_CONDITIONS}
    for condition in RIDGE_CONDITIONS:
        features = load_feature(condition, paper_root, categorical)
        require(
            features.shape
            == ((14400, 1200) if condition == "categorical_role_hour" else (14400, 4096)),
            "feature shape mismatch",
        )
        require(np.isfinite(features).all(), f"nonfinite feature: {condition}")
        for fold in range(5):
            print(
                json.dumps(
                    {"stage": "fit", "condition": condition, "fold": fold, "time": utc_now()}
                ),
                flush=True,
            )
            candidates, selection, solver_prediction, prediction, reproduction = fit_condition_fold(
                condition, fold, features, data, inner, archived_candidates, diagnostics
            )
            candidate_rows.extend(candidates)
            selection_rows.append(selection)
            solver_predictions[condition].append(solver_prediction)
            new_predictions[condition].append(prediction)
            reproduction_rows.append(reproduction)
        del features
    diagnostics_frame = pd.DataFrame(diagnostics)
    candidates_frame = pd.DataFrame(candidate_rows)
    selection_frame = pd.DataFrame(selection_rows)
    candidates_frame = candidates_frame.merge(
        selection_frame[["condition", "outer_fold", "audited_selected_alpha"]],
        on=["condition", "outer_fold"],
        how="left",
        validate="many_to_one",
    )
    candidates_frame["selected_audited"] = np.isclose(
        candidates_frame["alpha"], candidates_frame["audited_selected_alpha"]
    )
    reproduction_frame = pd.DataFrame(reproduction_rows)
    diagnostics_frame.to_csv(output / "solver_diagnostics.csv", index=False)
    candidates_frame.to_csv(output / "readout_candidates_audited.csv", index=False)
    selection_frame.to_csv(output / "readout_selection_audited.csv", index=False)
    audited = archived_oof.copy()
    for condition in RIDGE_CONDITIONS:
        frame = (
            pd.concat(new_predictions[condition], ignore_index=True)
            .sort_values("query_id")
            .reset_index(drop=True)
        )
        solver_frame = (
            pd.concat(solver_predictions[condition], ignore_index=True)
            .sort_values("query_id")
            .reset_index(drop=True)
        )
        old = audited[f"prediction_{condition}"].to_numpy(float)
        new = frame["y_hat"].to_numpy(float)
        reproduced = solver_frame["y_hat"].to_numpy(float)
        reproduction_rows.append(
            {
                "condition": condition,
                "outer_fold": "all",
                "entity": "oof_prediction",
                "n_values": len(new),
                "max_abs_diff": float(np.max(np.abs(reproduced - old))),
                "max_rel_diff": rel_difference(reproduced, old),
                "allclose_rtol_1e-7_atol_1e-8": bool(
                    np.allclose(reproduced, old, rtol=RTOL, atol=ATOL)
                ),
            }
        )
        audited[f"prediction_{condition}"] = new
        audited[f"prediction_log_raw_{condition}"] = frame["y_hat_log_raw"].to_numpy(float)
    reproduction_frame = pd.DataFrame(reproduction_rows)
    reproduction_frame.to_csv(output / "numerical_reproduction.csv", index=False)
    require(len(audited) == 14400 and audited["query_id"].nunique() == 14400, "audited OOF invalid")
    require(
        np.isfinite(audited.filter(regex="^prediction_").to_numpy(float)).all(),
        "audited predictions nonfinite",
    )
    audited.to_parquet(output / "oof_predictions_audited.parquet", index=False)
    overall, by_fold, components = calculate_all_metrics(audited)
    overall.to_csv(output / "metrics_overall_audited.csv", index=False)
    by_fold.to_csv(output / "metrics_by_fold_cell_audited.csv", index=False)
    components.to_csv(output / "metric_components_audited.csv", index=False)
    bias = total_bias(audited)
    bias.to_csv(output / "total_flow_bias_audited.csv", index=False)
    archived_metrics = pd.read_csv(archived / "metrics_overall.csv")
    solver_wide = archived_oof.copy()
    for condition in RIDGE_CONDITIONS:
        frame = (
            pd.concat(solver_predictions[condition], ignore_index=True)
            .sort_values("query_id")
            .reset_index(drop=True)
        )
        solver_wide[f"prediction_{condition}"] = frame["y_hat"].to_numpy(float)
    solver_metrics, _, _ = calculate_all_metrics(solver_wide)
    stages = {
        "archived_to_solver_verified_original_grid": (archived_metrics, solver_metrics),
        "solver_verified_original_grid_to_audited_final": (solver_metrics, overall),
        "archived_to_audited_final": (archived_metrics, overall),
    }
    comparison_rows = []
    for stage, (left, right) in stages.items():
        left_index = left.set_index("condition")
        right_index = right.set_index("condition")
        for condition in CONDITIONS:
            for metric in ALL_METRICS:
                old_value = float(left_index.loc[condition, metric])
                new_value = float(right_index.loc[condition, metric])
                difference = new_value - old_value
                comparison_rows.append(
                    {
                        "stage_comparison": stage,
                        "condition": condition,
                        "metric": metric,
                        "old_value": old_value,
                        "new_value": new_value,
                        "difference": difference,
                        "relative_difference": difference / old_value if old_value else np.nan,
                        "rounded_3_changed": round(old_value, 3) != round(new_value, 3),
                    }
                )
    comparison = pd.DataFrame(comparison_rows)

    def direction(frame: pd.DataFrame, left: str, right: str, metric: str) -> int:
        values = frame.set_index("condition")[metric]
        raw = float(values[left] - values[right])
        return int(np.sign(raw))

    comparisons = [
        ("name_llm", "time_mean"),
        ("coordinate_llm", "time_mean"),
        ("coordinate_llm", "name_llm"),
        ("name_llm", "categorical_role_hour"),
        ("coordinate_llm", "categorical_role_hour"),
        ("name_llm", "corrected_loglinear_gravity"),
        ("coordinate_llm", "corrected_loglinear_gravity"),
    ]
    direction_rows = []
    for left, right in comparisons:
        for metric in ["mae", "rmse", "smape", "cpc", "delta_beta", "delta_gini"]:
            direction_rows.append(
                {
                    "left_condition": left,
                    "right_condition": right,
                    "metric": metric,
                    "archived_direction": direction(archived_metrics, left, right, metric),
                    "audited_direction": direction(overall, left, right, metric),
                    "direction_changed": direction(archived_metrics, left, right, metric)
                    != direction(overall, left, right, metric),
                }
            )
    direction_frame = pd.DataFrame(direction_rows)
    comparison = comparison.merge(
        direction_frame.groupby("metric")["direction_changed"]
        .any()
        .rename("any_prespecified_direction_changed"),
        on="metric",
        how="left",
    )
    comparison.to_csv(output / "comparison_to_archived.csv", index=False)
    direction_frame.to_csv(output / "prespecified_comparison_directions.csv", index=False)
    write_reports(
        output,
        overall,
        by_fold,
        bias,
        selection_frame,
        comparison,
        diagnostics_frame,
        reproduction_frame,
    )
    readme = f"# Urban mobility 최소 수치 감사 산출물\n\n## 실행 환경과 입력\n\n- 시작: {started}\n- 완료: {utc_now()}\n- 원본 결과: `{archived}`\n- 확정 목표자료: `{gt_path}`\n- 원본 결과는 수정하거나 덮어쓰지 않았다.\n\n## 실행 명령\n\n```bash\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\\npython3 {script_path} \\\n  --paper-root {paper_root} \\\n  --archived-final {archived} \\\n  --output {output}\n```\n\n## 읽는 순서\n\n1. `AUDIT_REPORT.md`\n2. `manuscript_impact.md`\n3. `tables_for_manuscript.md`\n4. `metrics_overall_audited.csv`, `total_flow_bias_audited.csv`, `metrics_by_fold_cell_audited.csv`\n5. `solver_diagnostics.csv`, `numerical_reproduction.csv`, `comparison_to_archived.csv`\n\n## 범위\n\n명세에 적힌 LSQR 진단, categorical alpha 확장, 기존 OOF 총량 편향, fold별 셀 지표만 수행했다. LLM 표현 재추출과 명세 밖 분석은 수행하지 않았다.\n"
    (output / "README.md").write_text(readme, encoding="utf-8")
    checksum_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "output_checksums.csv":
            checksum_rows.append(
                {
                    "relative_path": str(path.relative_to(output)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(checksum_rows).to_csv(output / "output_checksums.csv", index=False)
    print(
        json.dumps(
            {"status": "PASS", "output": str(output), "completed_utc": utc_now()},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
