#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(sf)
  library(tidyverse)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("사용법: Rscript preprocess_flow.R RAW_CSV_DIR GU_SHAPEFILE OUTPUT_PARQUET")
}

raw_dir <- normalizePath(args[[1]], mustWork = TRUE)
shape_path <- normalizePath(args[[2]], mustWork = TRUE)
output_path <- args[[3]]
if (file.exists(output_path)) {
  stop("기존 전처리 결과를 덮어쓰지 않습니다: ", output_path)
}

files <- sort(list.files(raw_dir, pattern = "[.]csv$", full.names = TRUE))
stopifnot("시간대별 CSV가 24개여야 합니다." = length(files) == 24)

raw <- map_dfr(
  files,
  ~ read_csv(
    .x,
    locale = locale(encoding = "CP949"),
    col_types = cols(.default = col_character()),
    show_col_types = FALSE
  )
)

required <- c(
  "대상연월", "요일", "도착시간", "출발 시군구 코드", "도착 시군구 코드",
  "성별", "나이", "이동유형", "이동인구(합)"
)
stopifnot("필수 열이 누락되었습니다." = all(required %in% names(raw)))

months <- unique(raw$대상연월)
stopifnot("한 번에 한 대상연월만 처리합니다." = length(months) == 1)
target_month <- months[[1]]
month_start <- as.Date(paste0(target_month, "01"), format = "%Y%m%d")
month_end <- seq(month_start, by = "month", length.out = 2)[[2]] - 1
calendar_days <- seq(month_start, month_end, by = "day")
weekday_days <- sum(as.integer(format(calendar_days, "%u")) <= 5)

# 서울시 공개자료의 한 행은 월×요일×시간×OD×성별×연령×이동유형의
# 이동인구 합계다. 전체 평일 OD-hour의 하루 평균은 세부집단과 월~금을
# 합산한 뒤 그 달의 실제 평일 수로 나눈다.
selected <- raw |>
  transmute(
    target_month = 대상연월,
    day = 요일,
    arrival_hour = as.integer(도착시간),
    orig = as.character(`출발 시군구 코드`),
    dest = as.character(`도착 시군구 코드`),
    sex = 성별,
    age = 나이,
    movement_type = 이동유형,
    masked = `이동인구(합)` == "*",
    flow = as.numeric(if_else(masked, "1.5", `이동인구(합)`))
  ) |>
  filter(
    str_starts(orig, "11"),
    str_starts(dest, "11"),
    orig != dest,
    day %in% c("월", "화", "수", "목", "금")
  )

stopifnot(
  "도착시간은 0~23이어야 합니다." = setequal(unique(selected$arrival_hour), 0:23),
  "이동인구에 결측값이 있습니다." = !anyNA(selected$flow),
  "이동인구는 음수가 아니어야 합니다." = all(selected$flow >= 0)
)

flow <- selected |>
  group_by(orig, dest, arrival_hour) |>
  summarise(flow = sum(flow) / weekday_days, .groups = "drop") |>
  arrange(orig, dest, arrival_hour)

stopifnot(
  "서울 자치구 간 유향 OD는 600개여야 합니다." = n_distinct(paste(flow$orig, flow$dest)) == 600,
  "최종 자료는 14,400행이어야 합니다." = nrow(flow) == 14400,
  "각 유향 OD에는 24시간이 모두 있어야 합니다." = all(count(flow, orig, dest)$n == 24)
)

gu <- st_read(shape_path, quiet = TRUE) |>
  mutate(SGG1_CD = as.character(SGG1_CD))
stopifnot("자치구 경계는 25개여야 합니다." = n_distinct(gu$SGG1_CD) == 25)

# Geometry for distances and newly derived display coordinates only.
# IMPORTANT: these lon/lat values are NOT the historical LLM prompt coordinates.
# Published readouts and polynomial ridge use the preserved prompt_source.parquet:
# its centroids were computed separately in EPSG:4326, stored at four decimals,
# then formatted at three decimals. Do not replace that source when reusing caches.
centroid_5179 <- gu |>
  st_transform(5179) |>
  st_centroid()
xy <- st_coordinates(centroid_5179)
centroid_4326 <- st_transform(centroid_5179, 4326)
lonlat <- st_coordinates(centroid_4326)

centroids <- tibble(
  gu_code = centroid_5179$SGG1_CD,
  x = xy[, 1],
  y = xy[, 2],
  lon = round(lonlat[, 1], 4),
  lat = round(lonlat[, 2], 4)
)

distance <- crossing(orig = sort(centroids$gu_code), dest = sort(centroids$gu_code)) |>
  filter(orig != dest) |>
  left_join(
    transmute(centroids, orig = gu_code, orig_x = x, orig_y = y),
    by = "orig"
  ) |>
  left_join(
    transmute(centroids, dest = gu_code, dest_x = x, dest_y = y),
    by = "dest"
  ) |>
  transmute(
    orig,
    dest,
    dist_km = sqrt((orig_x - dest_x)^2 + (orig_y - dest_y)^2) / 1000
  )

output <- flow |>
  left_join(distance, by = c("orig", "dest")) |>
  left_join(
    transmute(centroids, orig = gu_code, orig_lon = lon, orig_lat = lat),
    by = "orig"
  ) |>
  left_join(
    transmute(centroids, dest = gu_code, dest_lon = lon, dest_lat = lat),
    by = "dest"
  ) |>
  arrange(orig, dest, arrival_hour)

stopifnot(
  "공간변수 결합 후 결측값이 없어야 합니다." = !anyNA(output),
  "거리값은 양수여야 합니다." = all(output$dist_km > 0)
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write_parquet(output, output_path)

audit_path <- sub("[.]parquet$", "_preprocess_audit.csv", output_path)
write_csv(
  tibble(
    target_month = target_month,
    calendar_weekdays = weekday_days,
    input_csv_files = length(files),
    selected_raw_rows = nrow(selected),
    masked_rows_imputed_as_1_5 = sum(selected$masked),
    directed_od = n_distinct(paste(output$orig, output$dest)),
    hours = n_distinct(output$arrival_hour),
    output_rows = nrow(output),
    output_flow_mean = mean(output$flow),
    output_flow_sum = sum(output$flow),
    aggregation = "sum across sex, age, movement type, and Mon-Fri monthly totals; divide by actual weekday count"
  ),
  audit_path
)

message("PASS: ", output_path)
message("대상연월=", target_month, ", 평일수=", weekday_days, ", 행수=", nrow(output))
