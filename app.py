from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA


st.set_page_config(page_title="Urban AQI Time-Series Forecaster", layout="wide")


def aqi_category(val: float) -> str:
    if val <= 50:
        return "Good"
    if val <= 100:
        return "Satisfactory"
    if val <= 200:
        return "Moderate"
    if val <= 300:
        return "Poor"
    if val <= 400:
        return "Very Poor"
    return "Severe"


@st.cache_data
def load_outputs(
    features_path: str = "data/processed/delhi_aqi_features.csv",
    preds_path: str = "data/processed/test_predictions.csv",
):
    feat_df = pd.read_csv(features_path)
    feat_df["Date"] = pd.to_datetime(feat_df["Date"])
    feat_df = feat_df.sort_values("Date").reset_index(drop=True)

    preds_df = pd.read_csv(preds_path)
    preds_df["Date"] = pd.to_datetime(preds_df["Date"])
    preds_df = preds_df.sort_values("Date").reset_index(drop=True)
    return feat_df, preds_df


st.title("Urban Air Quality Index (AQI) Time-Series Forecaster")
st.caption("Chronological AQI modeling & evaluation on daily historical air quality data (Delhi)")

features_file = Path("data/processed/delhi_aqi_features.csv")
preds_file = Path("data/processed/test_predictions.csv")

if not features_file.exists() or not preds_file.exists():
    st.error("Processed files not found. Run `src/clean_data.py`, `src/features.py`, and `src/train.py` first.")
    st.stop()

feat_df, preds_df = load_outputs()

# Sidebar controls
st.sidebar.header("Forecast & View Settings")
model_options = [c for c in preds_df.columns if c not in ("Date", "Actual_AQI")]
selected_models = st.sidebar.multiselect(
    "Test-Set Models to Plot",
    options=model_options,
    default=[m for m in ["ARIMA_2_1_2", "Ridge_Lag_Features", "Persistence_lag1"] if m in model_options],
)
horizon_days = st.sidebar.slider("Future Forecast Horizon (Days)", min_value=1, max_value=14, value=7)

# Top summary KPIs on held-out test set
latest_actual = float(preds_df["Actual_AQI"].iloc[-1])
arima_mae = float(np.mean(np.abs(preds_df["Actual_AQI"] - preds_df["ARIMA_2_1_2"]))) if "ARIMA_2_1_2" in preds_df else np.nan

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Test AQI", f"{latest_actual:.1f}", aqi_category(latest_actual))
col2.metric("Test Window", f"{len(preds_df)} Days")
col3.metric("ARIMA(2,1,2) MAE", f"{arima_mae:.2f}" if not np.isnan(arima_mae) else "N/A")
col4.metric("Total Clean Days", f"{len(feat_df):,}")

# Section 1: Held-out Test Set Comparison
st.subheader("Held-Out Test Set: Actual vs. Predicted AQI")
fig_test = go.Figure()
fig_test.add_trace(
    go.Scatter(
        x=preds_df["Date"],
        y=preds_df["Actual_AQI"],
        mode="lines+markers",
        name="Actual AQI",
        line=dict(width=2.5),
    )
)
for m in selected_models:
    fig_test.add_trace(
        go.Scatter(
            x=preds_df["Date"],
            y=preds_df[m],
            mode="lines",
            name=m,
        )
    )
fig_test.update_layout(xaxis_title="Date", yaxis_title="AQI", hovermode="x unified", height=420)
st.plotly_chart(fig_test, width="stretch")

# Model metrics table on test set
metrics_rows = []
y_true = preds_df["Actual_AQI"].to_numpy(dtype=float)
non_zero = y_true != 0
for col in model_options:
    y_hat = preds_df[col].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(y_true - y_hat)))
    rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
    mape = float(np.mean(np.abs((y_true[non_zero] - y_hat[non_zero]) / y_true[non_zero])) * 100)
    metrics_rows.append({"Model": col, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)})

st.dataframe(pd.DataFrame(metrics_rows).sort_values("MAE"), width="stretch", hide_index=True)

# Section 2: Multi-Day Outlook from End of Dataset
st.subheader(f"Next {horizon_days}-Day Forecast from Latest Dataset Date ({feat_df['Date'].max().date()})")
series_full = feat_df["AQI"].astype(float).tolist()
arima_full = ARIMA(series_full, order=(2, 1, 2)).fit()
forecast_res = arima_full.get_forecast(steps=horizon_days)
mean_fc = np.asarray(forecast_res.predicted_mean, dtype=float)
conf_int = np.asarray(forecast_res.conf_int(alpha=0.2), dtype=float)  # 80% interval

future_dates = pd.date_range(start=feat_df["Date"].max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
fc_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast_AQI": np.round(mean_fc, 1),
    "Lower_80": np.round(conf_int[:, 0], 1),
    "Upper_80": np.round(conf_int[:, 1], 1),
})
fc_df["Category"] = fc_df["Forecast_AQI"].apply(aqi_category)

recent_hist = feat_df.tail(30)
fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(x=recent_hist["Date"], y=recent_hist["AQI"], mode="lines", name="Recent Actual AQI"))
fig_fc.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Forecast_AQI"], mode="lines+markers", name="ARIMA Forecast"))
fig_fc.add_trace(
    go.Scatter(
        x=list(fc_df["Date"]) + list(fc_df["Date"])[::-1],
        y=list(fc_df["Upper_80"]) + list(fc_df["Lower_80"])[::-1],
        fill="toself",
        name="80% Confidence Band",
        line=dict(width=0),
        opacity=0.2,
    )
)
fig_fc.update_layout(xaxis_title="Date", yaxis_title="AQI", hovermode="x unified", height=380)
st.plotly_chart(fig_fc, width="stretch")
st.dataframe(fc_df, width="stretch", hide_index=True)