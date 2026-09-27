# ✈️ Flight Delay Analysis & Prediction System

A complete data analytics and machine learning project for analysing 2024 U.S. domestic flight delays, identifying key delay drivers, and predicting whether a flight will be delayed — packaged as an interactive Streamlit dashboard.

---

## Problem Statement

Flight delays cost the U.S. aviation industry billions of dollars annually and affect millions of passengers. This project analyses 2024 domestic flight data from the Bureau of Transportation Statistics (BTS) to:

- Identify the root causes and patterns of flight delays
- Benchmark airline and airport on-time performance
- Detect high-risk routes, time periods, and carriers
- Build a machine learning model to predict whether a specific flight will depart late

---

## Objectives

1. Clean and prepare the 2024 BTS flight dataset for analysis
2. Perform exploratory data analysis (EDA) to uncover delay trends
3. Identify the most important drivers of flight delays (carrier, weather, NAS, aircraft, security)
4. Benchmark airlines and airports on on-time performance
5. Build and evaluate a Random Forest classification model for delay prediction
6. Present findings in an interactive, business-friendly dashboard
7. Convert findings into actionable recommendations for airlines and airports

---

## Dataset

| Property | Details |
|---|---|
| **Source** | [BTS TranStats On-Time Performance](https://www.transtats.bts.gov/) |
| **Kaggle** | [Flight Delay Dataset — 2024 by Hrishit Patil](https://www.kaggle.com/datasets/hrishitpatil/flight-delay-dataset-2024) |
| **Full dataset** | `flight_data_2024.csv` — ~7 million rows, 35 columns |
| **Sample** | `flight_data_2024_sample.csv` — 10,000 rows |
| **Period** | January 2024 – December 2024 |
| **Coverage** | U.S. domestic flights only |
| **License** | CC0: Public Domain |

### Key Columns Used

| Column | Description |
|---|---|
| `fl_date` | Flight date |
| `op_unique_carrier` | Airline carrier code |
| `origin` / `dest` | Origin / destination airport |
| `dep_delay` | Departure delay in minutes |
| `arr_delay` | Arrival delay in minutes |
| `cancelled` | Cancellation indicator (0/1) |
| `carrier_delay` | Minutes caused by carrier |
| `weather_delay` | Minutes caused by weather |
| `nas_delay` | Minutes caused by NAS (air traffic) |
| `late_aircraft_delay` | Minutes caused by late incoming aircraft |
| `distance` | Route distance in miles |

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Streamlit ≥ 1.32 | Interactive dashboard |
| Pandas ≥ 2.1 | Data loading, cleaning, aggregation |
| NumPy ≥ 1.26 | Numerical operations |
| Plotly ≥ 5.18 | Interactive charts and visualisations |
| scikit-learn ≥ 1.4 | ML model, preprocessing, evaluation |

---

## Project Structure

```
flight-delay-project/
│
├── app.py                          # Main Streamlit app (dashboard + ML)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── report.md                       # Full project report
│
└── [data files — place here]
    ├── flight_data_2024_sample.csv     # 10K sample (required minimum)
    └── flight_data_2024.csv            # Full 7M row dataset (optional)
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Step 1 — Clone / download the project

```bash
# If using git
git clone <repository-url>
cd flight-delay-project

# Or just unzip into a folder named flight-delay-project
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add the dataset

Download the dataset from Kaggle and place the CSV files in the **same folder** as `app.py`:

- Minimum required: `flight_data_2024_sample.csv` (10,000 rows — fast)
- For full analysis: `flight_data_2024.csv` (~1.3 GB, ~7M rows — slower to load)

**Kaggle download link:** https://www.kaggle.com/datasets/hrishitpatil/flight-delay-dataset-2024

---

## How to Run

```bash
streamlit run app.py
```

The app will automatically open in your default browser.

---

## Local Host Instructions

Once running, the app is accessible at:

```
Local URL:  http://localhost:8501
Network URL: http://<your-ip>:8501
```

To run on a specific port:

```bash
streamlit run app.py --server.port 8080
```

To disable automatic browser opening:

```bash
streamlit run app.py --server.headless true
```

---

## Dashboard Sections

| Tab | Contents |
|---|---|
| 🏠 Executive Overview | KPI cards, monthly delay trend, delay cause breakdown, key findings |
| 📊 Exploratory Analysis | Delay distribution, day-of-week, departure hour, seasonal, cancellation trends |
| 🛫 Airline Analysis | Delay rate by airline, avg delay, volume vs performance scatter, scorecard table |
| 🗺️ Airport & Routes | Origin/destination delay rates, busiest routes, high-risk route identification |
| 🔍 Delay Drivers | Cause breakdown (carrier/weather/NAS/security/late aircraft), airline heatmap |
| 🤖 ML Prediction | Model metrics, confusion matrix, ROC curve, feature importance, live prediction tool |

---

## Machine Learning Model

- **Algorithm:** Random Forest Classifier (100 trees, max depth 10, class-balanced)
- **Target variable:** `is_delayed` — binary flag where `dep_delay > 15 minutes` (FAA standard)
- **Features:** Month, day of week, departure hour, distance, carrier (encoded), origin airport (encoded)
- **Split:** 80% train / 20% test, stratified by target
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- **Live prediction:** Interactive form in the ML tab allows prediction for any custom flight parameters

---

## Key Business Insights (Summary)

1. **Delay rate** is approximately 18–22% of all departures depending on the period and airline.
2. **Late aircraft delays** cascade through the network — improving early-morning on-time performance has a compound positive effect.
3. **Summer and holiday periods** (June–August, December) see the highest delay volumes.
4. **Early morning departures** (5–7 AM) have the lowest delay rates — recommend for time-sensitive travel.
5. **Airline performance** varies significantly — a 10+ percentage point gap exists between best and worst carriers.

---

## License

Dataset: CC0 Public Domain.  
Code: MIT License — free to use, modify, and distribute.
