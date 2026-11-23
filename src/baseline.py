from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, name: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    non_zero = y_true != 0
    mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
    print(f"{name:<24} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | MAPE: {mape:5.2f}%")
    return {"model": name, "mae": mae, "rmse": rmse, "mape": mape}


def run_baselines(
    input_path: str | Path = "data/processed/delhi_aqi_features.csv",
    test_days: int = 90,
) -> None:
    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) <= test_days + 30:
        raise ValueError("Dataset is too small for the requested test_days split.")

    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()

    print(f"Train rows: {len(train_df)} ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"Test rows:  {len(test_df)} ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})\n")

    y_test = test_df["AQI"]

    # Baseline 1: Naive persistence (tomorrow = today / aqi_lag_1)
    pred_persistence = test_df["aqi_lag_1"]
    evaluate_predictions(y_test, pred_persistence, "Persistence (lag_1)")

    # Baseline 2: Prior 7-day rolling mean
    pred_roll7 = test_df["aqi_roll_mean_7"]
    evaluate_predictions(y_test, pred_roll7, "7-Day Rolling Mean")


if __name__ == "__main__":
    run_baselines(test_days=90)