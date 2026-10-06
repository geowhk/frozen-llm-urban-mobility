# Release gates

- [x] Author approved MIT for original code on 2026-10-06; LICENSE included.
- [x] Raw data excluded; official retrieval routes and required inputs documented in DATA_DOWNLOAD.md.
- [ ] Author clears redistribution of companion data/derived artifacts separately.
- [ ] Publish cleared companion artifacts and replace the private-input limitation with actual retrieval instructions.
- [x] Fresh Python 3.13/macOS arm64 environment: pinned dependencies installed, pip check passed, ten synthetic/entry-point tests passed, package wheel built and installed.
- [ ] Review the exact exported archive, not the surrounding manuscript workspace.
- [ ] Record repository URL and release identifier in the manuscript after publication.

Do not distribute model weights, access tokens, SSH keys, raw workspace history,
manuscripts, or private logs. A code license does not automatically license data,
third-party software, fonts, or model weights. MIT covers the author's original code and accompanying documentation only.

Figure 2 uses Arial supplied by the user; no proprietary font is included.
Figure 1 is a manuscript asset outside this analysis pipeline.

Fixed-prediction replay passed 228 input checks and 49 CSV comparisons at
rtol=1e-10, atol=1e-8. Some fit-only CSVs are preserved copies, not independently
refitted results. Six PDF/600 dpi PNG pairs were exported; five PNGs exactly match
the accepted snapshot, while Figure 2 changes only the arrival-hour notation.
