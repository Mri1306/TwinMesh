import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import time
import os

PORTS = [
    {
        "name": "Singapore",
        "lat": 1.2897,
        "lon": 103.8501
    },
    {
        "name": "Rotterdam",
        "lat": 51.9225,
        "lon": 4.4792
    },
    {
        "name": "Shanghai",
        "lat": 31.2304,
        "lon": 121.4737
    },
    {
        "name": "Nhava Sheva",
        "lat": 18.9498,
        "lon": 72.9500
    },
    {
        "name": "Busan",
        "lat": 35.1796,
        "lon": 129.0756
    },
    {
        "name": "Hamburg",
        "lat": 53.5511,
        "lon": 9.9937
    },
    {
        "name": "Antwerp",
        "lat": 51.2194,
        "lon": 4.4025
    }
]

START_DATE = "2023-02-02"
END_DATE   = "2026-02-02"

OUTPUT_FOLDER = "data/raw/weather"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def fetch_weather(port):

    print(f"\nCollecting weather data for {port['name']}...")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": port["lat"],
        "longitude": port["lon"],
        "start_date": START_DATE,
        "end_date": END_DATE,

        "daily": [
            "temperature_2m_mean",
            "precipitation_sum",
            "windspeed_10m_max"
        ],

        "timezone": "auto"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Failed for {port['name']}")
        return None

    data = response.json()

    daily = data["daily"]

    df = pd.DataFrame({
        "date": daily["time"],
        "port": port["name"],
        "temperature_mean": daily["temperature_2m_mean"],
        "precipitation_sum": daily["precipitation_sum"],
        "wind_speed_max": daily["windspeed_10m_max"]
    })

    df["storm_alert"] = (
        df["wind_speed_max"] > 40
    ).astype(int)

    df["heavy_rain_alert"] = (
        df["precipitation_sum"] > 50
    ).astype(int)

    df["weather_risk_score"] = (
        (df["storm_alert"] * 0.7) +
        (df["heavy_rain_alert"] * 0.3)
    )

    return df

all_dataframes = []

for port in tqdm(PORTS):

    df = fetch_weather(port)

    if df is not None:

        all_dataframes.append(df)

        # SAVE INDIVIDUAL PORT FILE
        output_file = f"{OUTPUT_FOLDER}/{port['name'].replace(' ', '_')}_weather.csv"

        df.to_csv(output_file, index=False)

        print(f"Saved: {output_file}")

    time.sleep(1)

final_df = pd.concat(all_dataframes, ignore_index=True)

master_file = f"{OUTPUT_FOLDER}/maritime_weather_full.csv"

final_df.to_csv(master_file, index=False)

print("\n===================================")
print("WEATHER DATA COLLECTION COMPLETE")
print("===================================")

print(f"Total Records: {len(final_df)}")
print(f"Saved Master File: {master_file}")