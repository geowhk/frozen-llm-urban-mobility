"""Model identifiers and display labels shared by all stages."""

NEW = "coordinate_time_poly2_ridge"
LABELS = {
    "name_llm": "Name-based readout",
    "coordinate_llm": "Coordinate-based readout",
    NEW: "Coordinate–time polynomial ridge",
    "time_mean": "Hourly mean",
    "categorical_role_hour": "Region–hour categorical",
    "corrected_marginal_product": "Marginal product",
    "corrected_loglinear_gravity": "Log-linear gravity",
}
MODELS = {k: "prediction_" + k for k in LABELS}
BASE_MODELS = {k: v for k, v in MODELS.items() if k != NEW}
