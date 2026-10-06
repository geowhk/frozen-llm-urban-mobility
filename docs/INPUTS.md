# Inputs and reproduction boundaries

## Data-free checks

Run `PYTHONPATH=src python -m unittest discover -s tests -v` from the repository root.
These synthetic tests check mathematical and split contracts, not published results.

## Fixed-prediction reproduction

The `--package` argument must point to the accepted artifact tree, not this repository.
Required layout:

```
config.json
CHECKSUMS.sha256                  # checked when supplied
inputs/                          # accepted source inputs and saved assignments
results/oof_predictions_all_models.parquet
results/metrics_overall.csv
```

`inputs/` includes `oof_audited.parquet`, `folds.csv`, historical prompt coordinates,
mobility estimates, inner splits, district names, bin definitions, and diagnostic
reference files. Preserve the complete accepted tree: individual commands need
additional files referenced by the evaluation and plotting modules. The existing
CLI validates the reference package before evaluation and refuses existing outputs.
It is an accepted-artifact replay workflow, not an arbitrary-new-dataset interface.

The accepted tree is currently held privately. No public download is claimed.
Before release with reproducible results, supply a cleared companion artifact with
an immutable version, checksum, and download instructions. If redistribution is
restricted, document how to obtain each input and which results remain unavailable.
Do not publish the whole private workspace as a substitute.

## Raw preprocessing and historical coordinates

Raw mobility data: Seoul Open Data Plaza, November 2024, 24 arrival-hour CSV files,
CP949 encoding, with the columns checked in `scripts/01_preprocess_flow.R`.
Boundary source: SGIS, 2019 district polygons, 25 Seoul districts and `SGG1_CD`.
The R script computes weekday-average flows and EPSG:5179 centroid distances.
Its transformed lon/lat fields are not the historical prompt coordinates.
Published representations and polynomial features use preserved `prompt_source.parquet`.
Never substitute the new geometry's lon/lat into that historical input or cached vectors.

Original-model refitting additionally needs cached representations and the archived
strict-dyad audit inputs listed by `audited_models.py`. Those are not bundled here.
R package versions and the historical extraction environment are not fully locked.
