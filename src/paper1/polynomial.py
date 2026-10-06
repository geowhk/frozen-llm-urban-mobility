"""Methods: six direct inputs, 27 polynomial terms, training-only SVD ridge."""

from pathlib import Path
import json, sys, platform
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from . import core
from . import diagnostics as ld
from .config import NEW, LABELS, MODELS
from .validation import record_check
from functools import partial


def transform_fit(x):
    a = StandardScaler().fit(x)
    p = PolynomialFeatures(degree=2, include_bias=False)
    q = p.fit_transform(a.transform(x))
    b = StandardScaler().fit(q)
    return (b.transform(q), (a, p, b))


def transform(x, obj):
    a, p, b = obj
    return b.transform(p.transform(a.transform(x)))


def fit(x, y, alpha):
    return Ridge(alpha=float(alpha), solver="svd", fit_intercept=False).fit(x, y - y.mean())


def fit_benchmark(root):
    """Fit only the prespecified polynomial benchmark using saved inner/outer splits."""
    root = Path(root)
    CHECKS = []
    check = partial(record_check, CHECKS)
    cfg = json.loads((root / "config.json").read_text())
    R = root / "results"
    I = root / "inputs"
    before = ld.sha256(I / "oof_audited.parquet")
    original = pd.read_parquet(I / "oof_audited.parquet")
    gt = pd.read_parquet(I / "mobility.parquet")
    checks = ld.validate_inputs(original, gt, I / "audit", R)
    for k, v in checks.items():
        check(k, v)
    folds = pd.read_csv(I / "folds.csv")
    inner = pd.read_csv(I / "inner_splits.csv")
    check(
        "outer_manifest_exact",
        original.merge(
            folds[["dyad_id", "outer_fold"]],
            on="dyad_id",
            validate="many_to_one",
            suffixes=("", "_saved"),
        )
        .eval("outer_fold == outer_fold_saved")
        .all(),
    )
    coords = ["orig_lat", "orig_lon", "dest_lat", "dest_lon"]
    keys = ["orig", "dest", "hour"]
    historic = pd.read_parquet(I / "prompt_source.parquet").rename(columns={"arrival_hour": "hour"})
    current = gt.rename(columns={"arrival_hour": "hour"})
    cmp = historic.merge(
        current, on=keys, validate="one_to_one", suffixes=("_historical", "_current")
    )
    for c in coords:
        check(
            "prompt_coordinate_round3_" + c,
            np.array_equal(cmp[c + "_historical"].round(3), cmp[c + "_current"].round(3)),
        )
    df = original.merge(historic[keys + coords], on=keys, validate="one_to_one", sort=False)
    check("oof_order_preserved", np.array_equal(df.query_id, original.query_id))
    df[coords] = df[coords].astype(float).round(3)
    df["sin_hour"] = np.sin(2 * np.pi * df.hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df.hour / 24)
    x = df[cfg["base_features"]].to_numpy(float)
    z = np.log1p(df.y_gt.to_numpy(float))
    pred = np.full(len(df), np.nan)
    raw = pred.copy()
    coverage = np.zeros(len(df), int)
    candidates = []
    selected = []
    states = []
    for f in range(5):
        test = df.outer_fold.eq(f).to_numpy()
        train = ~test
        s = inner[inner.outer_fold == f]
        trids = set(s.loc[s.inner_subset == "train", "dyad_id"])
        vids = set(s.loc[s.inner_subset == "validation", "dyad_id"])
        tids = set(df.loc[test, "dyad_id"])
        alltrain = set(df.loc[train, "dyad_id"])
        check(f"fold{f}_split_sizes", len(trids) == 210 and len(vids) == 30 and (len(tids) == 60))
        check(
            f"fold{f}_no_dyad_overlap",
            not (trids & vids or tids & trids or tids & vids) and trids | vids == alltrain,
        )
        a = df.dyad_id.isin(trids).to_numpy()
        b = df.dyad_id.isin(vids).to_numpy()
        xx, st = transform_fit(x[a])
        xv = transform(x[b], st)
        check(f"fold{f}_27_features", xx.shape[1] == 27)
        vals = []
        for alpha in cfg["alphas"]:
            m = fit(xx, z[a], alpha)
            v = m.predict(xv) + z[a].mean()
            loss = float(np.mean((v - z[b]) ** 2))
            row = dict(outer_fold=f, alpha=alpha, validation_log_mse=loss)
            vals.append(row)
            candidates.append(row)
        best = min(vals, key=lambda q: (q["validation_log_mse"], -q["alpha"]))
        xt, state = transform_fit(x[train])
        xe = transform(x[test], state)
        m = fit(xt, z[train], best["alpha"])
        loghat = m.predict(xe) + z[train].mean()
        raw[test] = loghat
        pred[test] = np.maximum(0, np.expm1(loghat))
        coverage[test] += 1
        w = np.linalg.lstsq(
            np.vstack([xt, np.sqrt(best["alpha"]) * np.eye(27)]),
            np.r_[z[train] - z[train].mean(), np.zeros(27)],
            rcond=None,
        )[0]
        diff = float(np.max(np.abs(xe @ w + z[train].mean() - loghat)))
        check(f"fold{f}_independent_solver", diff < 1e-07, str(diff))
        check(
            f"fold{f}_training_scaler_means",
            np.allclose(state[0].mean_, x[train].mean(0), rtol=0, atol=1e-12),
        )
        selected.append(
            {
                **best,
                "boundary": best["alpha"] in [cfg["alphas"][0], cfg["alphas"][-1]],
                "solver_max_abs_log_difference": diff,
            }
        )
        for scope, mask, objs in [("inner_train", a, st), ("outer_train", train, state)]:
            states.append(
                dict(
                    outer_fold=f,
                    scope=scope,
                    n_rows=int(mask.sum()),
                    target_mean=float(z[mask].mean()),
                    base_mean=objs[0].mean_.tolist(),
                    base_scale=objs[0].scale_.tolist(),
                    expanded_mean=objs[2].mean_.tolist(),
                    expanded_scale=objs[2].scale_.tolist(),
                )
            )
        np.savez(
            R / f"fold_{f}_fit.npz",
            coef=m.coef_,
            target_mean=z[train].mean(),
            base_mean=state[0].mean_,
            base_scale=state[0].scale_,
            expanded_mean=state[2].mean_,
            expanded_scale=state[2].scale_,
        )
        print(
            "fold",
            f,
            "alpha",
            best["alpha"],
            "validation MSE",
            best["validation_log_mse"],
            flush=True,
        )
    check("one_new_prediction_per_row", np.all(coverage == 1))
    check("new_predictions_finite", np.isfinite(pred).all())
    df[MODELS[NEW]] = pred
    df["prediction_log_raw_" + NEW] = raw
    for col in original.columns:
        check("old_column_unchanged_" + col, original[col].equals(df[col]))
    check("source_oof_hash_unchanged", before == ld.sha256(I / "oof_audited.parquet"))
    df.to_parquet(R / "oof_predictions_all_models.parquet", index=False)
    pd.DataFrame(candidates).to_csv(R / "penalty_candidates.csv", index=False)
    pd.DataFrame(selected).to_csv(R / "selected_penalties.csv", index=False)
    (R / "preprocessing_states.json").write_text(json.dumps(states, indent=2))
    pd.DataFrame({"feature": state[1].get_feature_names_out(cfg["base_features"])}).to_csv(
        R / "feature_names.csv", index=False
    )
    pd.DataFrame(CHECKS).to_csv(R / "training_checks.csv", index=False)
