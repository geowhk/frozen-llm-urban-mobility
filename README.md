# Urban mobility: reproducible analysis code

This is the active code tree. Historical scripts are retained separately, not used as competing entry points. No data, model weights, embeddings, manuscripts, or credentials are included here.

## Read in paper order

| Stage | Implementation | Purpose |
|---|---|---|
| 1. Mobility preprocessing | `scripts/01_preprocess_flow.R` | Weekday daily-average OD-hour mobility estimates; district geometry |
| 2. Frozen representations | `src/urban_mobility/representation/` | Preserved prompt/representation/cache primitives; see limitations below |
| 3. Evaluation design | `src/urban_mobility/splits.py` | Reciprocal-dyad outer/inner splits |
| 4. Audited original models | `src/urban_mobility/audited_models.py`, `core.py` | Final numerical-audit solver and six original conditions |
| 5. Direct-input comparator | `src/urban_mobility/polynomial.py` | Coordinate–time degree-2 ridge; six inputs, 27 expanded features |
| 6. Fixed-prediction evaluation | `src/urban_mobility/evaluation.py`, `diagnostics.py` | Overall, fold, flow, distance and district summaries |
| 7. Publication outputs | `src/urban_mobility/figures.py` | Tables and Figures 2–4/S1–S3 |
| Checks | `validation.py`, `tests/` | Stored hashes, fixed predictions, numeric equivalence and synthetic contracts |

## Install and run

Use Python 3.12 or later and a separate virtual environment. Release preparation was tested on Python 3.13/macOS arm64; Python 3.12 was not retested in this round. The pinned requirements are not a guarantee of availability on every operating system.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_urban_mobility.py verify --package /path/to/reference_artifacts
python run_urban_mobility.py reproduce --package /path/to/reference_artifacts --output /path/to/NEW_run
PYTHONPATH=src python -m unittest discover -s tests -v
```

`verify` is read-only. `evaluate` recalculates summaries; `figures` exports figures/tables; `reproduce` does both. These commands reuse stored OOF predictions and do not fit models. Output directories must not already exist. Figures require Arial installed locally (or `URBAN_MOBILITY_ARIAL_FONT` set to an existing Arial font file); no font file is distributed.

Training is deliberately separate and opt-in:

```sh
python run_urban_mobility.py fit-poly2 --package /path/to/reference_artifacts --output /path/to/NEW_poly2_fit
PYTHONPATH=src python -m urban_mobility.audited_models --data-root /path/to/private_data --archived-final /path/to/prior_strict_final --output /path/to/NEW_audit
Rscript scripts/01_preprocess_flow.R /path/to/raw_csv /path/to/districts.shp /path/to/NEW_mobility.parquet
```

R preprocessing requires `arrow`, `sf`, and `tidyverse`. R package versions were not pinned by this cleanup. Full preprocessing/model training was not rerun during reorganization. See [reproducibility levels](docs/REPRODUCIBILITY.md) before interpreting these commands.

## Publication readiness

The author's original code and accompanying documentation are licensed under the [MIT License](LICENSE). This grant does not cover third-party software, source datasets, model weights, embeddings, or fonts. Raw data are not redistributed: follow the [download and preprocessing guide](docs/DATA_DOWNLOAD.md). No GitHub publication has been performed as part of local release preparation. Do not publish the surrounding private workspace or accepted input bundle automatically.

## Release scope

The source tree supports synthetic tests without research data. Reproducing paper
results additionally requires the accepted input/prediction bundle; that bundle is
not distributed in this code-only candidate. See [input contract](docs/INPUTS.md).
This is not an end-to-end, raw-data-to-LLM reproduction claim. Representation
modules are preserved implementation evidence, not a supported extraction CLI.
PyTorch and Transformers are therefore not dependencies of the supported CPU
evaluation workflow; a future extraction environment needs separate validation.

After `pip install .`, `urban_mobility` and `python -m urban_mobility` are also entry points.
The source launcher is named `run_urban_mobility.py` to avoid shadowing the package.
The frozen requirements must be installed in a fresh environment before release;
historical runtime provenance is distinct from a newly validated environment.

No manuscript, private audit logs, model weights, prompts, mobility data, or OOF
predictions are included in the code-only archive. Do not treat omitted data as
synthetic. Redistribution of derived artifacts remains separate from the MIT
code license; see [release checklist](docs/RELEASE_CHECKLIST.md).
