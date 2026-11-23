# Urban Air Quality Index (AQI) Time-Series Forecaster

End-to-end chronological time-series forecasting pipeline and interactive Streamlit dashboard built on daily Indian city air quality data (`Delhi`, 2015–2020).

## Pipeline Overview
1. **Data Inspection (`src/load_data.py`):** Validates schemas, timestamps, city coverage, and missing sensor values.
2. **Timestamp & Gap Cleaning (`src/clean_data.py`):** Reindexes to a continuous daily calendar grid, applies bounded short-gap interpolation, and removes extended missing stretches.
3. **Feature Engineering (`src/features.py`):** Builds strictly past-only lag features (`1, 2, 3, 7, 14` days), shifted rolling statistics (`3, 7, 14` days), and calendar/Indian seasonal indicators.
4. **Chronological Benchmarking (`src/baseline.py` & `src/train.py`):** Evaluates models on a held-out 90-day test window (`2020-04-03` to `2020-07-01`) using MAE, RMSE, and MAPE.
5. **Interactive Dashboard (`app.py`):** Visualizes model performance, AQI categories, and multi-day forecasts with confidence intervals.

## Test Results (Last 90 Days)
| Model | MAE | RMSE | MAPE (%) |
| :--- | :---: | :---: | :---: |
| Rolling ARIMA(2,1,2) | 18.08 | 24.22 | 13.96% |
| Persistence (`lag_1`) | 19.74 | 26.45 | 15.46% |
| Ridge (Lag + Roll) | 20.82 | 26.30 | 17.33% |
| 7-Day Rolling Mean | 23.48 | 32.38 | 18.15% |
| Prophet (multi-step) | 69.85 | 80.06 | 59.86% |

## Quickstart
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/clean_data.py
python src/features.py
python src/train.py
streamlit run app.py