"""Read-only checks of accepted artifacts and refactor equivalence."""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .config import MODELS, BASE_MODELS
from .core import metrics


def record_check(rows, name, value, detail=""):
    rows.append({"check": name, "passed": bool(value), "detail": detail})
    if not value:
        raise AssertionError(f"{name}: {detail}")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_package(package):
    """Check stored hashes, keys, predictions, saved folds and overall metrics."""
    package = Path(package).resolve()
    checks = []
    ledger = package / "CHECKSUMS.sha256"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            expected, rel = line.split("  ", 1)
            path = (package / rel).resolve()
            if not path.is_relative_to(package):
                raise ValueError(f"Unsafe checksum path: {rel}")
            record_check(checks, "sha256:" + rel, sha256(path) == expected)
    frame = pd.read_parquet(package / "results/oof_predictions_all_models.parquet")
    record_check(
        checks,
        "unique_14400_od_hours",
        len(frame) == 14400 and not frame.duplicated(["orig", "dest", "hour"]).any(),
    )
    record_check(
        checks,
        "seven_finite_nonnegative_predictions",
        np.isfinite(frame[list(MODELS.values())]).all().all()
        and frame[list(MODELS.values())].ge(0).all().all(),
    )
    old = pd.read_parquet(package / "inputs/oof_audited.parquet")
    for condition, column in BASE_MODELS.items():
        record_check(checks, "unchanged:" + condition, old[column].equals(frame[column]))
    folds = pd.read_csv(package / "inputs/folds.csv")
    joined = frame.merge(folds, on="dyad_id", suffixes=("", "_saved"), validate="many_to_one")
    record_check(checks, "saved_folds", joined.outer_fold.eq(joined.outer_fold_saved).all())
    record_check(
        checks,
        "dyads_48_rows_single_fold",
        frame.groupby("dyad_id").size().eq(48).all()
        and frame.groupby("dyad_id").outer_fold.nunique().eq(1).all(),
    )
    reference = pd.read_csv(package / "results/metrics_overall.csv").set_index("condition")
    for condition, column in MODELS.items():
        evaluated = metrics(frame.assign(y_hat=frame[column]))
        for metric, value in evaluated.items():
            record_check(
                checks,
                f"metric:{condition}:{metric}",
                np.isclose(value, reference.loc[condition, metric], rtol=1e-10, atol=1e-8),
            )
    return checks


def compare_outputs(reference, output):
    """Compare derived numeric CSVs; validation-log counts may intentionally differ."""
    reference, output = Path(reference), Path(output)
    checked = []
    for directory in ["results", "tables", "figure_data"]:
        for path in sorted((output / directory).glob("*.csv")):
            if "check" in path.name:
                continue
            source = reference / directory / path.name
            if not source.exists():
                continue
            pd.testing.assert_frame_equal(
                pd.read_csv(source),
                pd.read_csv(path),
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-8,
            )
            checked.append(str(path.relative_to(output)))
    return checked
