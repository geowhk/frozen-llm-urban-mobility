from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse.linalg import lsqr

SEED = 202511
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
EXPECTED_GT_SHA256 = "5a34e3867e3f4e7430419cfec67fbd3772503b53267cf71b16d873a34aad910c"
LEGACY_GT_SHA256 = "d791b8edd32442b5b29201cdc849586b66cc418dc0857789b29cc797046d5826"
EXPECTED_CACHE_SHA256 = {
    "original": {
        "forward_row_index.parquet": "f568df4b3944412d41c86a6277f65c24919dcb224d937ac4cd021161b82a1b58",
        "forward_lasttoken_layer31.npy": "db35b8613e0a4293cc77512a8b1daf1a4bd1b41293367c683b431d85a0c7c5e6",
        "forward_cache_meta.json": "ed092959652ba7f5cb7c491d30f76e6a7720eb95a776ca5deb9b25d677a7ee4b",
    },
    "geometry": {
        "forward_row_index.parquet": "f568df4b3944412d41c86a6277f65c24919dcb224d937ac4cd021161b82a1b58",
        "forward_lasttoken_layer31.npy": "a1950aecf745466fefdeb7a1121410a95d69d930991abf2d2199aa43e17fb219",
        "forward_cache_meta.json": "9d2cef931f204b2d0024615ac675d51741d72dbb2e67136c0428cc05794233bb",
    },
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_id(orig: pd.Series, dest: pd.Series) -> pd.Series:
    return orig.astype(str) + "|" + dest.astype(str)


def canonical_data(gt_path: Path, row_index_path: Path) -> pd.DataFrame:
    gt = pd.read_parquet(gt_path).rename(columns={"arrival_hour": "hour", "flow": "y_gt"})
    gt["orig"] = gt["orig"].astype(str)
    gt["dest"] = gt["dest"].astype(str)
    gt["hour"] = pd.to_numeric(gt["hour"], errors="raise").astype(int)
    index = pd.read_parquet(row_index_path)
    index["orig"] = index["orig"].astype(str)
    index["dest"] = index["dest"].astype(str)
    index["hour"] = pd.to_numeric(index["hour"], errors="raise").astype(int)
    keys = ["orig", "dest", "hour"]
    out = index.merge(
        gt[keys + ["flow" if "flow" in gt.columns else "y_gt", "dist_km"]].rename(
            columns={"flow": "y_gt", "dist_km": "gt_dist_km"}
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if out["y_gt"].isna().any():
        raise ValueError("표현 행 인덱스와 원자료가 1:1로 연결되지 않습니다.")
    if not np.allclose(out["dist_km"], out["gt_dist_km"], rtol=0.0, atol=1e-12):
        raise ValueError("표현 행 인덱스와 원자료의 거리가 일치하지 않습니다.")
    out = out.drop(columns="gt_dist_km")
    out["pair_id"] = pair_id(out["orig"], out["dest"])
    return out


def make_folds(pairs: Iterable[str], seed: int = SEED, n_folds: int = 5) -> pd.DataFrame:
    unique = sorted(set(map(str, pairs)))
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"outer|{seed}|{value}".encode()).hexdigest(),
    )
    rows = [(value, rank % n_folds) for rank, value in enumerate(ordered)]
    return pd.DataFrame(rows, columns=["pair_id", "fold"])


def inner_validation_pairs(
    outer_train_pairs: Iterable[str], outer_fold: int, seed: int = SEED
) -> set[str]:
    unique = sorted(set(map(str, outer_train_pairs)))
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"inner|{seed}|{outer_fold}|{value}".encode()).hexdigest(),
    )
    n_validation = max(1, round(len(ordered) / 8))
    return set(ordered[:n_validation])


def validate_folds(data: pd.DataFrame, folds: pd.DataFrame) -> dict[str, object]:
    if folds["pair_id"].duplicated().any():
        raise ValueError("하나의 방향성 OD 관계에 fold가 중복 배정되었습니다.")
    attached = data.merge(folds, on="pair_id", how="left", validate="many_to_one")
    if attached["fold"].isna().any():
        raise ValueError("fold가 배정되지 않은 관측이 있습니다.")
    hours = attached.groupby("pair_id")["hour"].nunique()
    if not (hours == 24).all():
        raise ValueError("모든 방향성 OD 관계가 정확히 24시간을 포함하지 않습니다.")
    fold_counts = folds.groupby("fold").size().sort_index()
    if fold_counts.max() - fold_counts.min() > 1:
        raise ValueError("fold 사이의 OD 관계 수 차이가 1을 초과합니다.")
    return {
        "n_rows": int(len(attached)),
        "n_pairs": int(folds["pair_id"].nunique()),
        "fold_pair_counts": {str(k): int(v) for k, v in fold_counts.items()},
        "all_rows_assigned_once": bool(len(attached) == len(data)),
        "all_pair_hours_together": True,
    }


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(x, dtype=np.float64).mean(axis=0)
    scale = np.asarray(x, dtype=np.float64).std(axis=0)
    scale[scale == 0.0] = 1.0
    return mean, scale


def transform_standardized(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - mean) / scale


def fit_ridge(x: np.ndarray, y_log: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y_log, dtype=np.float64)
    y_mean = float(y.mean())
    coef = lsqr(
        x,
        y - y_mean,
        damp=float(alpha) ** 0.5,
        atol=1e-8,
        btol=1e-8,
        iter_lim=2000,
        show=False,
    )[0]
    return coef, y_mean


def predict_ridge(x: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return np.clip(np.expm1(np.asarray(x, dtype=np.float64) @ coef + intercept), 0.0, None)


def choose_alpha(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray
) -> dict:
    mean, scale = fit_standardizer(x_train)
    train_z = transform_standardized(x_train, mean, scale)
    val_z = transform_standardized(x_val, mean, scale)
    y_train_log = np.log1p(np.clip(np.asarray(y_train, dtype=np.float64), 0.0, None))
    y_val_log = np.log1p(np.clip(np.asarray(y_val, dtype=np.float64), 0.0, None))
    candidates = []
    for alpha in ALPHAS:
        coef, intercept = fit_ridge(train_z, y_train_log, alpha)
        pred_log = val_z @ coef + intercept
        loss = float(np.mean((pred_log - y_val_log) ** 2))
        candidates.append({"alpha": float(alpha), "validation_log_mse": loss})
    selected = min(candidates, key=lambda row: (row["validation_log_mse"], -row["alpha"]))
    return {**selected, "candidates": candidates}


def fit_corrected_marginals(train: pd.DataFrame, all_pairs: pd.DataFrame) -> dict:
    pair_membership = train[["orig", "dest"]].drop_duplicates()
    all_pair_membership = all_pairs[["orig", "dest"]].drop_duplicates()
    origin_rate = (
        pair_membership.groupby("orig").size() / all_pair_membership.groupby("orig").size()
    ).rename("origin_rate")
    destination_rate = (
        pair_membership.groupby("dest").size() / all_pair_membership.groupby("dest").size()
    ).rename("destination_rate")
    global_rate = len(pair_membership) / len(all_pair_membership)
    origin_hour = train.groupby(["orig", "hour"])["y_gt"].sum().rename("partial_O").reset_index()
    origin_hour = origin_hour.merge(origin_rate.reset_index(), on="orig", how="left")
    origin_hour["corrected_O"] = origin_hour["partial_O"] / origin_hour["origin_rate"]
    destination_hour = (
        train.groupby(["dest", "hour"])["y_gt"].sum().rename("partial_D").reset_index()
    )
    destination_hour = destination_hour.merge(destination_rate.reset_index(), on="dest", how="left")
    destination_hour["corrected_D"] = (
        destination_hour["partial_D"] / destination_hour["destination_rate"]
    )
    hour_total = train.groupby("hour")["y_gt"].sum().rename("partial_T").reset_index()
    hour_total["corrected_T"] = hour_total["partial_T"] / global_rate
    hourly_mean = train.groupby("hour")["y_gt"].mean().rename("hourly_mean").reset_index()
    return {
        "origin_hour": origin_hour,
        "destination_hour": destination_hour,
        "hour_total": hour_total,
        "hourly_mean": hourly_mean,
        "global_mean": float(train["y_gt"].mean()),
        "global_rate": float(global_rate),
    }


def _margin_features(model: dict, rows: pd.DataFrame) -> pd.DataFrame:
    out = rows[["orig", "dest", "hour", "dist_km"]].copy()
    out = out.merge(
        model["origin_hour"][["orig", "hour", "corrected_O"]], on=["orig", "hour"], how="left"
    )
    out = out.merge(
        model["destination_hour"][["dest", "hour", "corrected_D"]], on=["dest", "hour"], how="left"
    )
    out = out.merge(model["hour_total"][["hour", "corrected_T"]], on="hour", how="left")
    out = out.merge(model["hourly_mean"], on="hour", how="left")
    return out


def predict_corrected_marginal_product(model: dict, rows: pd.DataFrame) -> np.ndarray:
    features = _margin_features(model, rows)
    fallback = features["hourly_mean"].fillna(model["global_mean"]).to_numpy(float)
    o = features["corrected_O"].to_numpy(float)
    d = features["corrected_D"].to_numpy(float)
    total = features["corrected_T"].to_numpy(float)
    valid = np.isfinite(o) & np.isfinite(d) & np.isfinite(total) & (total > 0.0)
    return np.clip(np.where(valid, o * d / total, fallback), 0.0, None)


def fit_corrected_gravity(train: pd.DataFrame, all_pairs: pd.DataFrame) -> dict:
    model = fit_corrected_marginals(train, all_pairs)
    features = _margin_features(model, train)
    x = np.column_stack(
        [
            np.ones(len(features)),
            np.log1p(features["corrected_O"].fillna(0.0).to_numpy(float)),
            np.log1p(features["corrected_D"].fillna(0.0).to_numpy(float)),
            np.log1p(features["dist_km"].clip(lower=0.0).to_numpy(float)),
        ]
    )
    model["gravity_coef"] = np.linalg.lstsq(x, np.log1p(train["y_gt"].to_numpy(float)), rcond=None)[
        0
    ]
    return model


def predict_corrected_gravity(model: dict, rows: pd.DataFrame) -> np.ndarray:
    features = _margin_features(model, rows)
    x = np.column_stack(
        [
            np.ones(len(features)),
            np.log1p(features["corrected_O"].fillna(0.0).to_numpy(float)),
            np.log1p(features["corrected_D"].fillna(0.0).to_numpy(float)),
            np.log1p(features["dist_km"].clip(lower=0.0).to_numpy(float)),
        ]
    )
    return np.clip(np.expm1(x @ model["gravity_coef"]), 0.0, None)


def gini(values: np.ndarray) -> float:
    x = np.sort(np.clip(np.nan_to_num(np.asarray(values, dtype=float)), 0.0, None))
    if len(x) == 0 or x.sum() <= 0.0:
        return 0.0
    index = np.arange(1, len(x) + 1, dtype=float)
    return float(2.0 * np.sum(index * x) / (len(x) * x.sum()) - (len(x) + 1.0) / len(x))


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    gt = frame["y_gt"].to_numpy(float)
    pred = frame["y_hat"].to_numpy(float)
    error = pred - gt
    beta_gt = -float(np.polyfit(np.log(frame["dist_km"].to_numpy(float)), np.log1p(gt), 1)[0])
    beta_pred = -float(np.polyfit(np.log(frame["dist_km"].to_numpy(float)), np.log1p(pred), 1)[0])
    origin = frame.groupby("orig")[["y_gt", "y_hat"]].sum()
    destination = frame.groupby("dest")[["y_gt", "y_hat"]].sum()
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "smape": float(np.mean(2.0 * np.abs(error) / (np.abs(pred) + np.abs(gt) + 1e-9))),
        "cpc": float(2.0 * np.minimum(gt, pred).sum() / (gt.sum() + pred.sum() + 1e-9)),
        "delta_beta": float(abs(beta_pred - beta_gt)),
        "rho_origin": float(origin["y_gt"].corr(origin["y_hat"], method="spearman")),
        "rho_destination": float(destination["y_gt"].corr(destination["y_hat"], method="spearman")),
        "delta_gini": float(abs(gini(pred) - gini(gt))),
    }


def stored_metrics(path: Path) -> dict[str, float]:
    block = json.loads(path.read_text(encoding="utf-8"))["gu"]
    return {**block["accuracy"], **block["patterns"]}
