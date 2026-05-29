import requests
import pandas as pd
import time
import os

OUTPUT_FOLDER = "data/raw/disruptions"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

QUERIES = [
    "port congestion",
    "shipping delay",
    "supply chain disruption",
    "maritime incident",
    "port strike",
    "freight disruption",
    "container shortage",
    "logistics crisis",
    "cargo delay",
    "trade disruption"
]

START_DATE = "20230202000000"
END_DATE   = "20260202235959"

all_records = []

for query in QUERIES:

    print("\n===================================")
    print(f"Collecting Query: {query}")
    print("===================================")

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,
        "startdatetime": START_DATE,
        "enddatetime": END_DATE,
        "sort": "DateDesc"
    }

    success = False
    retries = 0
    max_retries = 5

    while not success and retries < max_retries:

        try:

            response = requests.get(
                GDELT_URL,
                params=params,
                timeout=60,
                headers={
                    "User-Agent": "TwinMeshResearchBot/1.0"
                }
            )

            print(f"HTTP Status: {response.status_code}")

            if response.status_code == 200:

                data = response.json()

                articles = data.get("articles", [])

                print(f"Articles found: {len(articles)}")

                for article in articles:

                    record = {

                        
                        "query": query,

                        
                        "title": article.get("title"),
                        "source": article.get("sourceCommonName"),
                        "domain": article.get("domain"),

                        
                        "seendate": article.get("seendate"),

                        
                        "url": article.get("url"),

                        
                        "language": article.get("language"),

                       
                        "socialimage": article.get("socialimage"),

                        
                        "tone": article.get("tone"),
                        "sourcecountry": article.get("sourceCountry"),
                        "theme": article.get("theme")
                    }

                    all_records.append(record)

                success = True

                print("Waiting 10 seconds before next query...")
                time.sleep(10)

            elif response.status_code == 429:

                retries += 1

                wait_time = 20 * retries

                print("RATE LIMITED BY GDELT")
                print(f"Retry {retries}/{max_retries}")
                print(f"Waiting {wait_time} seconds...")

                time.sleep(wait_time)

            else:

                print(f"Failed query: {query}")
                print(f"HTTP Status: {response.status_code}")

                break

        except Exception as e:

            retries += 1

            wait_time = 15 * retries

            print(f"Error: {e}")
            print(f"Retry {retries}/{max_retries}")
            print(f"Waiting {wait_time} seconds...")

            time.sleep(wait_time)

df = pd.DataFrame(all_records)

print("\n====================================")
print("Cleaning Dataset...")
print("====================================")

df.drop_duplicates(
    subset=["title", "url"],
    inplace=True
)

df.dropna(
    subset=["title"],
    inplace=True
)

if "language" in df.columns:
    df = df[df["language"] == "English"]

df.reset_index(drop=True, inplace=True)

output_csv = f"{OUTPUT_FOLDER}/gdelt_disruptions_2023_2026.csv"

df.to_csv(output_csv, index=False)

print("\n====================================")
print("DISRUPTION DATA COLLECTION COMPLETE")
print("====================================")

print(f"Final Records: {len(df)}")
print(f"Saved File: {output_csv}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())