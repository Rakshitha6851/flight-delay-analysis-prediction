"""
clean_dataset.py
================
Standalone script to clean the 2024 Flight Delay dataset and save
the result as a new CSV file — no Streamlit required.

Usage
-----
    python clean_dataset.py

Input files (place in same folder as this script):
    flight_data_2024_sample.csv   — 10K row sample
    flight_data_2024.csv          — full 7M row dataset

Output:
    flight_data_2024_sample_cleaned.csv
    flight_data_2024_cleaned.csv
"""

import os
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
DELAY_THRESHOLD = 15   # FAA standard: delayed if dep_delay > 15 min

CARRIER_NAMES = {
    "9E": "Endeavor Air",       "AA": "American Airlines",  "AS": "Alaska Airlines",
    "B6": "JetBlue",            "DL": "Delta Air Lines",    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",      "HA": "Hawaiian Airlines",  "MQ": "Envoy Air",
    "NK": "Spirit Airlines",    "OH": "PSA Airlines",       "OO": "SkyWest Airlines",
    "QX": "Horizon Air",        "UA": "United Airlines",    "WN": "Southwest Airlines",
    "YV": "Mesa Airlines",      "YX": "Republic Airways",   "ZW": "Air Wisconsin",
}

DAY_NAMES   = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",  6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# ──────────────────────────────────────────────────────────────────────────────
# CLEANING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def clean_dataset(input_path: str, output_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data …")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"      Shape after load : {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Step 1: Normalise column names ────────────────────────────────────────
    print("[2/7] Normalising column names …")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"      Columns : {list(df.columns)}")

    # ── Step 2: Drop full duplicates ──────────────────────────────────────────
    print("[3/7] Removing duplicate rows …")
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"      Duplicates removed : {removed:,}  (rows remaining: {len(df):,})")

    # ── Step 3: Fix data types ────────────────────────────────────────────────
    print("[4/7] Fixing data types …")

    # Flight date → datetime
    if "fl_date" in df.columns:
        df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce")
        bad_dates = df["fl_date"].isna().sum()
        print(f"      fl_date parsed (unparseable set to NaT: {bad_dates:,})")

    # All numeric columns → numeric (coerce garbage to NaN)
    numeric_cols = [
        "year", "month", "day_of_month", "day_of_week",
        "op_carrier_fl_num",
        "crs_dep_time", "dep_time", "dep_delay",
        "taxi_out", "wheels_off", "wheels_on", "taxi_in",
        "crs_arr_time", "arr_time", "arr_delay",
        "crs_elapsed_time", "actual_elapsed_time", "air_time",
        "distance",
        "carrier_delay", "weather_delay", "nas_delay",
        "security_delay", "late_aircraft_delay",
    ]
    for col in numeric_cols:
        if col in df.columns:
            before_nulls = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after_nulls = df[col].isna().sum()
            new_nulls = after_nulls - before_nulls
            if new_nulls > 0:
                print(f"      {col}: {new_nulls:,} non-numeric values coerced to NaN")

    # Cancelled / diverted → integer 0 or 1
    for col in ["cancelled", "diverted"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ── Step 4: Handle missing values ─────────────────────────────────────────
    print("[5/7] Handling missing values …")

    # Delay cause columns: NaN means no delay from that cause → fill 0
    delay_cause_cols = [
        "carrier_delay", "weather_delay", "nas_delay",
        "security_delay", "late_aircraft_delay",
    ]
    for col in delay_cause_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            df[col] = df[col].fillna(0)
            print(f"      {col}: filled {n:,} NaNs with 0")

    # Report remaining nulls in key columns (not dropped — preserved for analysis)
    key_cols = ["dep_delay", "arr_delay", "taxi_out", "taxi_in",
                "air_time", "actual_elapsed_time"]
    for col in key_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            pct = n / len(df) * 100
            if n > 0:
                print(f"      {col}: {n:,} NaN ({pct:.2f}%) - "
                      "retained (expected for cancelled flights)")

    # ── Step 5: Validate value ranges ─────────────────────────────────────────
    print("[6/7] Validating value ranges …")

    # Distance must be > 0
    if "distance" in df.columns:
        invalid = (df["distance"] <= 0).sum()
        if invalid > 0:
            print(f"      distance: {invalid:,} rows with distance <= 0 - set to NaN")
            df.loc[df["distance"] <= 0, "distance"] = np.nan

    # Month must be 1–12
    if "month" in df.columns:
        invalid = (~df["month"].between(1, 12)).sum()
        if invalid > 0:
            print(f"      month: {invalid:,} out-of-range values - set to NaN")
            df.loc[~df["month"].between(1, 12), "month"] = np.nan

    # Day of week must be 1–7
    if "day_of_week" in df.columns:
        invalid = (~df["day_of_week"].between(1, 7)).sum()
        if invalid > 0:
            print(f"      day_of_week: {invalid:,} out-of-range values - set to NaN")
            df.loc[~df["day_of_week"].between(1, 7), "day_of_week"] = np.nan

    # ── Step 6: Derive analytical features ────────────────────────────────────
    print("[7/7] Deriving analytical features …")

    # Binary delay target (FAA: delayed if dep_delay > 15 min)
    if "dep_delay" in df.columns:
        df["is_delayed"] = (df["dep_delay"] > DELAY_THRESHOLD).astype("Int64")
        n_delayed = df["is_delayed"].sum()
        total_with_delay = df["dep_delay"].notna().sum()
        pct = n_delayed / total_with_delay * 100 if total_with_delay else 0
        print(f"      is_delayed : {n_delayed:,} flights "
              f"({pct:.1f}% of flights with known dep_delay)")

    # Departure hour (0–23) from scheduled time in hhmm format
    if "crs_dep_time" in df.columns:
        df["dep_hour"] = (df["crs_dep_time"] // 100).astype("Int64")
        df.loc[~df["dep_hour"].between(0, 23), "dep_hour"] = pd.NA
        print(f"      dep_hour   : derived from crs_dep_time")

    # Season
    if "month" in df.columns:
        season_map = {
            12: "Winter", 1: "Winter",  2: "Winter",
            3:  "Spring", 4: "Spring",  5: "Spring",
            6:  "Summer", 7: "Summer",  8: "Summer",
            9:  "Fall",  10: "Fall",   11: "Fall",
        }
        df["season"] = df["month"].map(season_map)
        print(f"      season     : derived from month")

    # Route label
    if "origin" in df.columns and "dest" in df.columns:
        df["route"] = df["origin"].astype(str) + " -> " + df["dest"].astype(str)
        print(f"      route      : derived from origin + dest")

    # Carrier full name
    if "op_unique_carrier" in df.columns:
        df["carrier_name"] = (
            df["op_unique_carrier"].map(CARRIER_NAMES)
            .fillna(df["op_unique_carrier"])
        )
        print(f"      carrier_name : mapped from op_unique_carrier")

    # Day name / Month name (human-readable labels)
    if "day_of_week" in df.columns:
        df["day_name"] = df["day_of_week"].map(DAY_NAMES)
    if "month" in df.columns:
        df["month_name"] = df["month"].map(MONTH_NAMES)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving cleaned dataset …")
    df.to_csv(output_path, index=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved : {output_path}")
    print(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Size  : {size_mb:.1f} MB")

    # ── Summary report ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  CLEANING SUMMARY")
    print(f"{'─'*60}")
    print(f"  Original rows    : {before:,}")
    print(f"  Duplicates removed: {removed:,}")
    print(f"  Final rows       : {len(df):,}")
    print(f"  Final columns    : {df.shape[1]}")
    print(f"  New columns added: is_delayed, dep_hour, season, route, "
          f"carrier_name, day_name, month_name")

    # Null summary for key columns
    print(f"\n  Null counts (key columns):")
    check_cols = [
        "fl_date", "dep_delay", "arr_delay", "cancelled",
        "distance", "is_delayed", "dep_hour",
    ]
    for col in check_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            pct = n / len(df) * 100
            print(f"    {col:<22}: {n:>8,}  ({pct:.2f}%)")

    print(f"\nDone! Cleaned file saved to: {output_path}\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    files = {
        "flight_data_2024_sample.csv":  "flight_data_2024_sample_cleaned.csv",
        "flight_data_2024.csv":         "flight_data_2024_cleaned.csv",
    }

    found_any = False
    for input_file, output_file in files.items():
        if os.path.exists(input_file):
            found_any = True
            clean_dataset(input_file, output_file)
        else:
            print(f"[SKIP] {input_file} not found - skipping.")

    if not found_any:
        print(
            "\nERROR: No dataset files found in the current directory.\n"
            "  Please place one or both of the following files here:\n"
            "    - flight_data_2024_sample.csv\n"
            "    - flight_data_2024.csv\n"
            "  Then re-run:  python clean_dataset.py\n"
        )
