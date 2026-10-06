# Obtain source data locally

Raw mobility and boundary data are not included in this repository. Download them from their providers under the applicable terms. MIT applies to the author's code/documentation, not these datasets. Official pages checked on 2026-10-06; portal menus may change. Historical file availability and account-specific downloads were not tested in this release preparation.

## 1. Seoul mobility estimates

Official source: [Seoul Living Mobility, Seoul Open Data Plaza](https://data.seoul.go.kr/dataVisual/seoul/seoulLivingMigration.do).

1. Open the source page and find **데이터 내려받기** (download data).
2. Choose **자치구 단위** (district level), not 행정동 단위.
3. Select **November 2024 (202411)** in the available date controls. Download the district-level files for all **24 arrival hours, 00–23**. If supplied as an archive, extract it locally. Exact historical archive names and deep download URLs have not been verified; do not substitute a different month or the newer metropolitan dataset silently.
4. Keep only those 24 CSV files in a dedicated local directory. The script reads every CSV in the directory and requires exactly 24 files. Preserve original Korean headers and CP949 encoding; avoid opening/resaving in a spreadsheet application.
5. Retrieve the source page's **서울생활이동 데이터 설명서** and **자치구 코드 정보** for interpreting the fields. If 202411 is no longer offered, use the page's **문의하기** link to request that historical release; it is not provided by this repository.

Required columns, checked by the script:

`대상연월`, `요일`, `도착시간`, `출발 시군구 코드`, `도착 시군구 코드`, `성별`, `나이`, `이동유형`, `이동인구(합)`.

Before running, confirm that 대상연월 is 202411 throughout. The script accepts a single month but does not enforce this particular month. Hours must span 0–23. Source rows are monthly weekday-category totals, not individual journeys. The script selects Seoul interdistrict flows and Monday–Friday, replaces masked `*` values with 1.5, sums the relevant records, and divides by the month's calendar weekday count. November 2024 has 21 Monday–Friday days. Output must contain 600 directed pairs × 24 hours = 14,400 rows.

The [portal terms, Articles 10–11](https://data.seoul.go.kr/etc/accessTerms.do) require attribution and distinguish third-party rights. Cite the provider and dataset in downstream use; this repository grants no data redistribution permission.

## 2. SGIS boundary data

Official source: [SGIS data provision](https://sgis.mods.go.kr/view/pss/openDataIntrcn) (the older sgis.kostat.go.kr address redirects here).

1. In **자료제공**, use **자료신청**. Complete the site's account/application requirements and review the conditions presented there.
2. Request **2019 센서스용 행정구역 경계**, **시군구** level, Seoul, in SHP format. Do not substitute administrative-dong polygons or a newer boundary year. The provider lists annual administrative boundaries, but the exact historical download filename is not verified here.
3. Use **신청자료 다운로드** after approval. Keep the `.shp`, `.shx`, `.dbf`, `.prj` and any accompanying files together. The provider describes EPSG:5179; verify the downloaded layer's CRS rather than assigning one blindly.
4. The current preprocessing script expects a layer containing **only 25 Seoul districts**, with the identifier field **SGG1_CD**. If the download contains the whole country, select Seoul districts in GIS software and save a separate local layer preserving attributes and CRS. If its identifier field or code scheme differs, stop and establish the mapping; do not merely rename an unrelated code field.
5. Confirm one feature per district, 25 unique codes, and correspondence with the mobility codes (the historical scheme includes 11010, 11020, etc.). Use the provider's code documentation, not a guessed ordering.

The official data page lists public/free provision; this is not treated here as blanket authorization to redistribute SHP files. Download locally and retain the terms accepted with the application.

## 3. Run preprocessing

Install R and the `arrow`, `sf`, and `tidyverse` packages. `sf` may need platform-specific geospatial libraries. This release does not pin a validated R environment.

```r
install.packages(c("arrow", "sf", "tidyverse"))
```

From the repository root, replacing placeholders with local paths:

```sh
Rscript scripts/01_preprocess_flow.R /path/to/202411_district_csv /path/to/seoul_2019.shp /path/to/new_mobility.parquet
```

Choose a new output filename ending in `.parquet`; existing output is refused. Inspect both the Parquet output and the adjacent `_preprocess_audit.csv`: month 202411, 21 weekdays, 24 input files, 600 directed pairs, 24 hours and 14,400 output rows. No source data were downloaded or preprocessing rerun during this documentation update.

## 4. What these downloads do not reproduce automatically

The script computes distance centroids in EPSG:5179 and transforms them to longitude/latitude. Those newly derived coordinates are **not** the historical prompt coordinates used by the published readouts and direct-input comparator. The historical `prompt_source.parquet`, cached representations and saved OOF predictions are not distributed in this code-only release. Do not substitute new coordinates while claiming exact reproduction of those results.

The `verify`/`reproduce` CLI needs the complete accepted artifact tree described in [INPUTS.md](INPUTS.md), not just the new Parquet file. Raw downloads alone do not supply that tree. Synthetic tests run without it; exact published-result replay remains unavailable from this public code archive alone.

Third-party Python/R dependencies retain their own licenses and are installed separately. Meta Llama 3 weights are not supplied: obtain the appropriate model from the official provider and review its [license](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/LICENSE). The repository's MIT license neither replaces those terms nor grants rights to model outputs or caches. Arial is also not distributed.
