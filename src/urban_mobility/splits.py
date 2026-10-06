from __future__ import annotations
import hashlib
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
from .core import SEED, gini

N_FOLDS = 5
N_DISTRICTS = 25
N_DIRECTED_OD = 600
N_DYADS = 300
N_HOURS = 24
N_ROWS = 14400
OUTER_TEST_DYADS = 60
INNER_VALIDATION_DYADS = 30


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def canonical_dyad_id(orig: pd.Series, dest: pd.Series) -> pd.Series:
    left = orig.astype(str).to_numpy()
    right = dest.astype(str).to_numpy()
    left_first = left <= right
    endpoint_1 = np.where(left_first, left, right)
    endpoint_2 = np.where(left_first, right, left)
    return pd.Series(endpoint_1 + "|" + endpoint_2, index=orig.index, name="dyad_id")


def add_dyads(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["dyad_id"] = canonical_dyad_id(out["orig"], out["dest"])
    return out


def _hash(prefix: str, seed: int, value: str, outer_fold: int | None = None) -> str:
    payload = (
        f"{prefix}|{seed}|{value}"
        if outer_fold is None
        else f"{prefix}|{seed}|{outer_fold}|{value}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def make_strict_outer_folds(
    dyads: Iterable[str], seed: int = SEED, n_folds: int = N_FOLDS
) -> pd.DataFrame:
    unique = sorted(set(map(str, dyads)))
    rows = [(dyad, _hash("outer", seed, dyad)) for dyad in unique]
    rows.sort(key=lambda row: row[1])
    return pd.DataFrame(
        [
            {
                "dyad_id": dyad,
                "hash_sha256": digest,
                "hash_rank": rank,
                "outer_fold": rank % n_folds,
            }
            for rank, (dyad, digest) in enumerate(rows)
        ]
    )


def make_strict_inner_assignments(
    outer_folds: pd.DataFrame, seed: int = SEED, n_validation: int = INNER_VALIDATION_DYADS
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outer_fold in range(N_FOLDS):
        train_dyads = outer_folds.loc[outer_folds["outer_fold"].ne(outer_fold), "dyad_id"].astype(
            str
        )
        hashed = [(dyad, _hash("inner", seed, dyad, outer_fold)) for dyad in train_dyads]
        hashed.sort(key=lambda row: row[1])
        require(len(hashed) == N_DYADS - OUTER_TEST_DYADS, "outer train dyad 수가 240이 아닙니다.")
        for rank, (dyad, digest) in enumerate(hashed):
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "dyad_id": dyad,
                    "inner_subset": "validation" if rank < n_validation else "train",
                    "hash_sha256": digest,
                    "hash_rank": rank,
                }
            )
    return pd.DataFrame(rows)


def validate_strict_data(data: pd.DataFrame) -> dict[str, object]:
    required = {"query_id", "orig", "dest", "hour", "y_gt", "dist_km", "pair_id", "dyad_id"}
    require(
        required.issubset(data.columns), f"필수 열 누락: {sorted(required - set(data.columns))}"
    )
    require(len(data) == N_ROWS, f"전체 행 수가 {N_ROWS:,}이 아닙니다: {len(data):,}")
    require(data["query_id"].nunique() == N_ROWS, "query_id가 14,400개 고유하지 않습니다.")
    require(data["orig"].nunique() == N_DISTRICTS, "출발 자치구가 25개가 아닙니다.")
    require(data["dest"].nunique() == N_DISTRICTS, "도착 자치구가 25개가 아닙니다.")
    require(set(data["orig"]) == set(data["dest"]), "출발·도착 자치구 집합이 다릅니다.")
    require(
        not data["orig"].astype(str).eq(data["dest"].astype(str)).any(), "자기이동 행이 있습니다."
    )
    require(data["pair_id"].nunique() == N_DIRECTED_OD, "directed OD가 600개가 아닙니다.")
    require(data["dyad_id"].nunique() == N_DYADS, "dyad가 300개가 아닙니다.")
    require(
        data["hour"].nunique() == N_HOURS and set(data["hour"]) == set(range(N_HOURS)),
        "시간이 0~23시가 아닙니다.",
    )
    require((data.groupby("pair_id").size() == N_HOURS).all(), "directed OD마다 24행이 아닙니다.")
    require(
        (data.groupby("pair_id")["hour"].nunique() == N_HOURS).all(),
        "directed OD의 24시간이 완전하지 않습니다.",
    )
    require(
        (data.groupby("dyad_id")["pair_id"].nunique() == 2).all(), "dyad마다 두 방향 OD가 아닙니다."
    )
    require((data.groupby("dyad_id").size() == 2 * N_HOURS).all(), "dyad마다 48행이 아닙니다.")
    require(
        np.isfinite(data[["y_gt", "dist_km"]].to_numpy(float)).all(),
        "목표값 또는 거리에 NaN/Inf가 있습니다.",
    )
    require(
        (data["y_gt"] >= 0).all() and (data["dist_km"] > 0).all(),
        "목표값 또는 거리가 허용 범위를 벗어납니다.",
    )
    return {
        "districts": N_DISTRICTS,
        "directed_od": N_DIRECTED_OD,
        "dyads": N_DYADS,
        "hours": N_HOURS,
        "rows": N_ROWS,
        "directed_od_hours_complete": True,
        "dyad_directions_complete": True,
    }


def validate_strict_splits(
    data: pd.DataFrame, outer_folds: pd.DataFrame, inner: pd.DataFrame
) -> dict[str, object]:
    require(outer_folds["dyad_id"].nunique() == N_DYADS, "outer fold dyad가 300개가 아닙니다.")
    require(not outer_folds["dyad_id"].duplicated().any(), "outer fold에 dyad가 중복되었습니다.")
    require(
        set(outer_folds["outer_fold"]) == set(range(N_FOLDS)), "outer fold ID가 0~4가 아닙니다."
    )
    require(
        (outer_folds.groupby("outer_fold").size() == OUTER_TEST_DYADS).all(),
        "outer fold마다 60 dyad가 아닙니다.",
    )
    attached = data.merge(
        outer_folds[["dyad_id", "outer_fold"]], on="dyad_id", how="left", validate="many_to_one"
    )
    require(not attached["outer_fold"].isna().any(), "outer fold가 없는 행이 있습니다.")
    require(
        (attached.groupby("outer_fold").size() == 2880).all(), "outer fold마다 2,880행이 아닙니다."
    )
    require(
        (attached.groupby(["dyad_id"])["outer_fold"].nunique() == 1).all(),
        "한 dyad의 두 방향이 다른 outer fold에 있습니다.",
    )
    for fold in range(N_FOLDS):
        fold_inner = inner[inner["outer_fold"].eq(fold)]
        test_dyads = set(outer_folds.loc[outer_folds["outer_fold"].eq(fold), "dyad_id"])
        train_dyads = set(outer_folds.loc[outer_folds["outer_fold"].ne(fold), "dyad_id"])
        require(
            set(fold_inner["dyad_id"]) == train_dyads,
            f"outer fold {fold}: inner split이 outer train과 다릅니다.",
        )
        require(
            not test_dyads & set(fold_inner["dyad_id"]),
            f"outer fold {fold}: test dyad가 inner split에 있습니다.",
        )
        require(
            fold_inner["dyad_id"].nunique() == 240,
            f"outer fold {fold}: inner dyad가 240개가 아닙니다.",
        )
        require(
            not fold_inner["dyad_id"].duplicated().any(),
            f"outer fold {fold}: inner dyad가 중복되었습니다.",
        )
        counts = fold_inner.groupby("inner_subset").size().to_dict()
        require(
            counts == {"train": 210, "validation": 30},
            f"outer fold {fold}: inner 210/30 분할이 아닙니다: {counts}",
        )
    return {
        "outer_fold_dyads": {
            str(k): int(v) for k, v in outer_folds.groupby("outer_fold").size().items()
        },
        "outer_fold_rows": {
            str(k): int(v) for k, v in attached.groupby("outer_fold").size().items()
        },
        "reverse_direction_leakage_count": 0,
        "inner_train_dyads_per_fold": 210,
        "inner_validation_dyads_per_fold": 30,
    }


def dyad_split_audit(data: pd.DataFrame, outer_folds: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        data[["dyad_id", "orig", "dest", "pair_id"]]
        .drop_duplicates()
        .sort_values(["dyad_id", "pair_id"])
    )
    grouped = pairs.groupby("dyad_id")
    audit = grouped.agg(
        endpoint_1=("orig", lambda s: min(map(str, s))),
        endpoint_2=("orig", lambda s: max(map(str, s))),
        directed_OD_1=("pair_id", "first"),
        directed_OD_2=("pair_id", "last"),
    ).reset_index()
    row_counts = data.groupby("dyad_id").size().rename("row_count").reset_index()
    return (
        audit.merge(row_counts, on="dyad_id", validate="one_to_one")
        .merge(outer_folds, on="dyad_id", validate="one_to_one")
        .sort_values("hash_rank")
        .reset_index(drop=True)
    )


def fold_membership_audit(data: pd.DataFrame, outer_folds: pd.DataFrame) -> pd.DataFrame:
    relations = data[["orig", "dest", "dyad_id"]].drop_duplicates()
    districts = sorted(set(relations["orig"]) | set(relations["dest"]))
    rows: list[dict[str, object]] = []
    for fold in range(N_FOLDS):
        test_dyads = set(outer_folds.loc[outer_folds["outer_fold"].eq(fold), "dyad_id"])
        test = relations[relations["dyad_id"].isin(test_dyads)]
        train = relations[~relations["dyad_id"].isin(test_dyads)]
        for district in districts:
            outgoing = int(train["orig"].eq(district).sum())
            incoming = int(train["dest"].eq(district).sum())
            require(
                outgoing > 0 and incoming > 0,
                f"outer fold {fold}: {district}의 학습 outgoing/incoming 관계가 0입니다.",
            )
            rows.append(
                {
                    "outer_fold": fold,
                    "district_code": district,
                    "test_incident_dyads": int(
                        ((test["orig"] == district) | (test["dest"] == district)).sum() / 2
                    ),
                    "train_outgoing_relations": outgoing,
                    "train_incoming_relations": incoming,
                    "outgoing_inclusion_rate": outgoing / 24,
                    "incoming_inclusion_rate": incoming / 24,
                }
            )
    return pd.DataFrame(rows)


def categorical_schema(district_codes: Sequence[str]) -> pd.DataFrame:
    codes = list(map(str, district_codes))
    require(
        len(codes) == N_DISTRICTS and len(set(codes)) == N_DISTRICTS,
        "범주형 schema의 자치구가 25개 고유하지 않습니다.",
    )
    rows = []
    for role, offset in (("origin", 0), ("destination", 600)):
        for district_index, district in enumerate(codes):
            for hour in range(N_HOURS):
                rows.append(
                    {
                        "feature_index": offset + district_index * N_HOURS + hour,
                        "role": role,
                        "district_code": district,
                        "hour": hour,
                    }
                )
    return pd.DataFrame(rows)


def categorical_role_hour_matrix(
    data: pd.DataFrame, district_codes: Sequence[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    schema = categorical_schema(district_codes)
    code_index = {code: index for index, code in enumerate(map(str, district_codes))}
    orig_index = data["orig"].astype(str).map(code_index)
    dest_index = data["dest"].astype(str).map(code_index)
    require(
        not orig_index.isna().any() and (not dest_index.isna().any()),
        "범주형 schema에 없는 자치구 코드가 있습니다.",
    )
    hours = data["hour"].to_numpy(int)
    require(
        np.isin(hours, np.arange(N_HOURS)).all(), "범주형 특징의 시간이 0~23 범위를 벗어납니다."
    )
    matrix = np.zeros((len(data), 1200), dtype=np.float32)
    row_index = np.arange(len(data))
    matrix[row_index, orig_index.to_numpy(int) * N_HOURS + hours] = 1.0
    matrix[row_index, 600 + dest_index.to_numpy(int) * N_HOURS + hours] = 1.0
    require(
        np.array_equal(matrix.sum(axis=1), np.full(len(data), 2.0)),
        "범주형 특징행마다 1이 정확히 두 개가 아닙니다.",
    )
    return (matrix, schema)


def zero_variance_feature_indices(matrix: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(matrix, dtype=np.float64).std(axis=0, ddof=0) == 0.0)


def metric_components(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["y_gt"].to_numpy(float)
    predicted = frame["y_hat"].to_numpy(float)
    log_distance = np.log(frame["dist_km"].to_numpy(float))
    beta_observed = -float(np.polyfit(log_distance, np.log1p(observed), 1)[0])
    beta_predicted = -float(np.polyfit(log_distance, np.log1p(predicted), 1)[0])
    gini_observed = gini(observed)
    gini_predicted = gini(predicted)
    return {
        "beta_observed": beta_observed,
        "beta_predicted": beta_predicted,
        "abs_delta_beta": abs(beta_predicted - beta_observed),
        "gini_observed": gini_observed,
        "gini_predicted": gini_predicted,
        "abs_delta_gini": abs(gini_predicted - gini_observed),
    }
