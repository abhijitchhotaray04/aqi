from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    non_zero = y_true != 0
    mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
    print(f"{name:<24} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | MAPE: {mape:5.2f}%")
    return {"model": name, "mae": mae, "rmse": rmse, "mape": mape}


def fit_rolling_arima(train_y: pd.Series, test_y: pd.Series, order: tuple[int, int, int] = (2, 1, 2)) -> np.ndarray:
    history = list(train_y.astype(float).values)
    preds = []
    for actual in test_y.astype(float).values:
        model = ARIMA(history, order=order)
        fitted = model.fit()
        yhat = float(fitted.forecast(steps=1)[0])
        preds.append(yhat)
        history.append(float(actual))
    return np.asarray(preds, dtype=float)


def train_and_compare(
    input_path: str | Path = "data/processed/delhi_aqi_features.csv",
    preds_out_path: str | Path = "data/processed/test_predictions.csv",
    test_days: int = 90,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()

    y_train = train_df["AQI"].astype(float)
    y_test = test_df["AQI"].astype(float)

    print(f"Benchmarking on {test_days} test days ({test_df['Date'].min().date()} to {test_df['Date'].max().date()}):\n")

    results = []
    pred_table = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Actual_AQI": y_test.values,
        "Persistence_lag1": test_df["aqi_lag_1"].astype(float).values,
    })

    # Baselines
    results.append(evaluate_predictions(y_test.values, pred_table["Persistence_lag1"].values, "Persistence (lag_1)"))
    results.append(evaluate_predictions(y_test.values, test_df["aqi_roll_mean_7"].astype(float).values, "7-Day Rolling Mean"))

    # 1. Ridge Regression on engineered lag + rolling + calendar features
    feature_cols = [
        "day_of_week",
        "day_of_year",
        "month",
        "is_weekend",
        "season",
        "aqi_lag_1",
        "aqi_lag_2",
        "aqi_lag_3",
        "aqi_lag_7",
        "aqi_lag_14",
        "aqi_roll_mean_3",
        "aqi_roll_std_3",
        "aqi_roll_mean_7",
        "aqi_roll_std_7",
        "aqi_roll_mean_14",
        "aqi_roll_std_14",
    ]
    for optional_col in ("pm25_lag_1", "pm10_lag_1"):
        if optional_col in df.columns:
            feature_cols.append(optional_col)

    cat_cols = ["season"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    ridge_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", Ridge(alpha=10.0)),
    ])
    ridge_pipe.fit(train_df[feature_cols], y_train)
    ridge_preds = ridge_pipe.predict(test_df[feature_cols])
    pred_table["Ridge_Lag_Features"] = ridge_preds
    results.append(evaluate_predictions(y_test.values, ridge_preds, "Ridge (Lag + Roll)"))

    # 2. Rolling 1-step ARIMA(2, 1, 2)
    arima_preds = fit_rolling_arima(y_train, y_test, order=(2, 1, 2))
    pred_table["ARIMA_2_1_2"] = arima_preds
    results.append(evaluate_predictions(y_test.values, arima_preds, "Rolling ARIMA(2,1,2)"))

    # 3. Optional Prophet model if installed
    try:
        from prophet import Prophet

        prophet_train = pd.DataFrame({"ds": train_df["Date"], "y": y_train})
        prophet_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        prophet_model.fit(prophet_train)
        future_df = pd.DataFrame({"ds": test_df["Date"]})
        forecast = prophet_model.predict(future_df)
        prophet_preds = forecast["yhat"].astype(float).values
        pred_table["Prophet"] = prophet_preds
        results.append(evaluate_predictions(y_test.values, prophet_preds, "Prophet"))
    except ImportError:
        print("\nNote: 'prophet' not installed—skipping Prophet model.")

    out_path = Path(preds_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_table.to_csv(out_path, index=False)
    print(f"\nSaved test predictions to: {out_path}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    train_and_compare(test_days=90)