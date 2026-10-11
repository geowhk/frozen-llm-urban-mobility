# Frozen language-model representations for urban mobility

Analysis code for **Evaluating frozen language-model representations for urban mobility prediction: A study of interdistrict flows in Seoul**

**Woohyung Kim** — Department of Geography Education, Seoul National University  
[ORCID](https://orcid.org/0009-0002-6347-6447)

**Manuscript status:** submitted to *Transactions in GIS* on October 7, 2026. This is a submitted manuscript, not an accepted or published article.

## Research questions

Can representations from a frozen language model support supervised prediction of flows between urban areas relative to non-LLM benchmarks? How does predictive usefulness change when locations are encoded as district names or coordinates? How well do the predicted flows reproduce individual flows, district totals, and allocation?

## Approach

The study uses average weekday hourly mobility estimates for November 2024 among Seoul's 25 districts: 600 directed district pairs and 24 arrival hours, yielding 14,400 OD-hour observations. Representations from a frozen Llama 3 model are supplied to separate ridge readouts for the name and coordinate conditions; the LLM does not directly generate mobility volumes.

Five-fold out-of-fold evaluation holds out both directions and all hours of each selected district pair. The districts themselves remain represented in training. Comparators include coordinate–time polynomial ridge and temporal, categorical, and spatial-interaction benchmarks.

## Main findings

The submitted manuscript reports that:

- The coordinate-based readout outperformed the name-based readout and the polynomial benchmark using coordinates and time directly, while gravity achieved the lowest overall errors.
- The name-based readout's higher district-total rank correlations than the polynomial benchmark did not extend to better allocation.
- Predictive usefulness depended on location encoding and the aspect of mobility being evaluated. These findings concern held-out relationships among known districts in this study, not demonstrated transfer to new cities or unseen districts.

## Research work represented here

This single-author project brings together mobility-data preprocessing, geographic input preparation, representation-based prediction, benchmark comparison, and spatial diagnostics. The code map below identifies the implementations: training-subset preprocessing and penalty selection, reciprocal-pair evaluation, coordinate–time polynomial features, flow/distance/district summaries, and numerical validation.

The pretrained language model and third-party libraries are external components. The public repository preserves analysis implementations; it does not provide every input needed to regenerate the reported results.

## Related master's-thesis repository

The earlier master's-thesis research is preserved in [llm-human-mobility](https://github.com/geowhk/llm-human-mobility). It includes output-level evaluation, internal-representation diagnostics, multiple spatial scales, and principle-prompt comparisons.

Use this repository for the subsequent journal-article analysis. Its preprocessing, evaluation design, and comparator set differ from the thesis workflow; the two repositories' results should not be treated as outputs from the same experiment.

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

## Start here: tests without research data

Use Python 3.12 or later and a separate virtual environment. Release preparation was tested on Python 3.13/macOS arm64; Python 3.12 was not retested in this round. The pinned requirements are not a guarantee of availability on every operating system.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
```

These synthetic tests check implementation contracts; they do not reproduce the manuscript's empirical results.

### Recalculate results with the reference artifact bundle

The following commands require the separately held input/prediction bundle described in [INPUTS.md](docs/INPUTS.md). It is not publicly downloadable from this repository, and raw data downloads alone do not supply it.

```sh
python run_urban_mobility.py verify --package /path/to/reference_artifacts
python run_urban_mobility.py reproduce --package /path/to/reference_artifacts --output /path/to/NEW_run
```

`verify` is read-only. `evaluate` recalculates summaries; `figures` exports figures/tables; `reproduce` does both. These commands reuse stored OOF predictions and do not fit models. Output directories must not already exist. Figures require Arial installed locally (or `URBAN_MOBILITY_ARIAL_FONT` set to an existing Arial font file); no font file is distributed.

### Preprocessing and model fitting

These commands require the additional source data, cached representations, or archived inputs specified in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md):

```sh
python run_urban_mobility.py fit-poly2 --package /path/to/reference_artifacts --output /path/to/NEW_poly2_fit
PYTHONPATH=src python -m urban_mobility.audited_models --data-root /path/to/private_data --archived-final /path/to/prior_strict_final --output /path/to/NEW_audit
Rscript scripts/01_preprocess_flow.R /path/to/raw_csv /path/to/districts.shp /path/to/NEW_mobility.parquet
```

R preprocessing requires `arrow`, `sf`, and `tidyverse`. R package versions were not pinned by this cleanup. Full preprocessing/model training was not rerun during reorganization. See [reproducibility levels](docs/REPRODUCIBILITY.md) before interpreting these commands.

## License and data access

The author's original code and accompanying documentation are licensed under the [MIT License](LICENSE). This grant does not cover third-party software, source datasets, model weights, embeddings, or fonts. Raw data are not redistributed: follow the [download and preprocessing guide](docs/DATA_DOWNLOAD.md). The private workspace and reference input/prediction bundle are not included in this public repository.

## Release scope

The source tree supports synthetic tests without research data. Reproducing paper
results additionally requires the accepted input/prediction bundle; that bundle is
not distributed in this code-only repository. See [input contract](docs/INPUTS.md).
This is not an end-to-end, raw-data-to-LLM reproduction claim. Representation
modules are preserved implementation evidence, not a supported extraction CLI.
PyTorch and Transformers are therefore not dependencies of the supported CPU
evaluation workflow; a future extraction environment needs separate validation.

After `pip install .`, `urban_mobility` and `python -m urban_mobility` are also entry points.
The source launcher is named `run_urban_mobility.py` to avoid shadowing the package.
Use a fresh environment when installing the pinned requirements. Historical runtime
provenance and the environment tested during release preparation are documented
separately in [reproducibility levels](docs/REPRODUCIBILITY.md).

No manuscript, private audit logs, model weights, prompts, mobility data, or OOF
predictions are included in the code-only repository. The omitted research inputs are not synthetic data. Redistribution of derived artifacts remains separate from the MIT
code license; see [release checklist](docs/RELEASE_CHECKLIST.md).
