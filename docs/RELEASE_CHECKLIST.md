# Public release scope and validation record

## Publicly available

- Original analysis code and accompanying documentation under the MIT License, approved by the author on October 6, 2026.
- Source-data retrieval instructions in [DATA_DOWNLOAD.md](DATA_DOWNLOAD.md); raw data are excluded.
- The manuscript identifies [this repository](https://github.com/geowhk/frozen-llm-urban-mobility) as its analysis-code source.

## Documented checks during release preparation

The release-preparation record reports a fresh Python 3.13/macOS arm64 environment: pinned dependencies installed, pip check passed, ten synthetic/entry-point tests passed, and the package wheel built and installed.

Fixed-prediction replay passed 228 input checks and 49 CSV comparisons at rtol=1e-10 and atol=1e-8. Some fit-only CSVs are preserved copies, not independently refitted results. Six PDF/600 dpi PNG pairs were exported; five PNGs exactly matched the reference snapshot, while Figure 2 changed only the arrival-hour notation.

These checks describe earlier release preparation, not new experiment runs during documentation updates. The initial cleanup tests and later environment checks are separate stages; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Remaining reproduction limits

- The reference input/prediction bundle is held separately and is not distributed here. Public code alone does not reproduce the reported empirical results.
- Redistribution of companion data and derived artifacts requires separate clearance. No public artifact release or download is claimed.
- Full model inference and end-to-end raw-data reproduction are not covered by the checks above.
- The earlier checklist's exported-archive review and release-identifier tasks are not certified as completed by this documentation update.

Model weights, credentials, raw workspace history, manuscripts, and private logs are excluded. MIT covers the author's original code and documentation, not third-party software, datasets, model weights, or fonts. Figure 2 requires a locally supplied Arial font. Figure 1 is a manuscript asset outside this analysis pipeline.
