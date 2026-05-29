import os

BASE_DIR         = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DIR          = os.path.join(BASE_DIR, "data/raw")
PORTWATCH_DIR    = os.path.join(RAW_DIR,  "PortWatch")
WEATHER_DIR      = os.path.join(RAW_DIR,  "Weather")
DISRUPTION_DIR   = os.path.join(RAW_DIR,  "Disruptions")
PROCESSED_DIR    = os.path.join(BASE_DIR, "data/processed")
PER_PORT_DIR     = os.path.join(PROCESSED_DIR, "per_port")
UNIFIED_DIR      = os.path.join(BASE_DIR, "data/unified")
HISTORICAL_DIR   = os.path.join(BASE_DIR, "data/historical")

PORTS = {
    "Singapore":   "singapore",
    "Rotterdam":   "rotterdam",
    "Shanghai":    "shanghai",
    "Nhava Sheva": "nhava_sheva",
    "Busan":       "busan",
    "Hamburg":     "hamburg",
    "Antwerp":     "antwerp",
}

VESSEL_TYPES_RAW = [
    "Container",
    "Dry Bulk",
    "General Cargo",
    "Roll-on/roll-off",
    "Tanker",
]

VESSEL_TYPES_SAFE = [
    "container",
    "dry_bulk",
    "general_cargo",
    "roro",
    "tanker",
]

VESSEL_RENAME = {
    "Container":          "container",
    "Dry Bulk":           "dry_bulk",
    "General Cargo":      "general_cargo",
    "Roll-on/roll-off":   "roro",
    "Tanker":             "tanker",
}

IMF_EXTRA_COLS = [
    "7-day Moving Average",
    "Prior Year: 7-day Moving Average",
]

STATE_THRESHOLDS = {
    "S0_max_congestion":      1.2,   
    "S1_max_congestion":      1.5,   
    "S2_max_congestion":      2.0,  
    "S3_min_congestion":      2.0,   
    "anomaly_sigma":          1.5,   
    "weather_risk_s1":        0.5,   
}

WEATHER_COLS = [
    "date",
    "port",
    "temperature_mean",
    "precipitation_sum",
    "wind_speed_max",
    "storm_alert",
    "heavy_rain_alert",
    "weather_risk_score",
]

PORT_KEYWORDS = {
    "Singapore":   ["singapore", "jurong", "psa"],
    "Rotterdam":   ["rotterdam", "europoort", "maasvlakte"],
    "Shanghai":    ["shanghai", "yangshan", "waigaoqiao"],
    "Nhava Sheva": ["nhava", "sheva", "jawaharlal", "jnpt", "mumbai port"],
    "Busan":       ["busan", "pusan", "hanjin"],
    "Hamburg":     ["hamburg", "hamburger hafen"],
    "Antwerp":     ["antwerp", "antwerpen", "zeebrugge"],
}

SPLIT_TRAIN = 0.70
SPLIT_VAL   = 0.15
SPLIT_TEST  = 0.15

COMBINED_START_DATE  = "2023-01-01"   
HISTORICAL_START     = "2019-01-01"   
HISTORICAL_END       = "2022-12-31"   
RED_SEA_START        = "2023-12-01"   
RED_SEA_END          = "2024-12-31"   

STATES       = ["S0", "S1", "S2", "S3", "S4"]
STATE_LABELS = {
    "S0": "Normal",
    "S1": "Delayed",
    "S2": "Congested",
    "S3": "Disrupted",
    "S4": "Recovery",
}
STATE_MAP    = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4}
INV_STATE    = {v: k for k, v in STATE_MAP.items()}