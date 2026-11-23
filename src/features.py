from pathlib import Path
import pandas as pd


def get_indian_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Summer"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-Monsoon"


def build_features(
    input_path: str | Path = "data/processed/delhi_aqi_clean.csv",
    output_path: str | Path = "data/processed/delhi_aqi_features.csv",
    lags: tuple[int, ...] = (1, 2, 3, 7, 14),
    windows: tuple[int, ...] = (3, 7, 14),
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 1. Calendar & seasonal features
    df["day_of_week"] = df["Date"].dt.day_of_week
    df["day_of_year"] = df["Date"].dt.day_of_year
    df["month"] = df["Date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["season"] = df["month"].apply(get_indian_season)

    # 2. Past-only lag features on target AQI
    for lag in lags:
        df[f"aqi_lag_{lag}"] = df["AQI"].shift(lag)

    # 3. Past-only rolling statistics (shifted by 1 so current day's AQI is excluded)
    shifted_aqi = df["AQI"].shift(1)
    for win in windows:
        df[f"aqi_roll_mean_{win}"] = shifted_aqi.rolling(window=win, min_periods=win).mean()
        df[f"aqi_roll_std_{win}"] = shifted_aqi.rolling(window=win, min_periods=win).std()

    # Optional: 1-day lag of PM2.5 and PM10 if present
    for col in ("PM2.5", "PM10"):
        if col in df.columns:
            clean_col = col.lower().replace(".", "")
            df[f"{clean_col}_lag_1"] = df[col].shift(1)

    # Drop rows where lag/rolling features aren't fully formed yet
    before_rows = len(df)
    featured_df = df.dropna(subset=[f"aqi_lag_{max(lags)}", f"aqi_roll_mean_{max(windows)}"]).reset_index(drop=True)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(out_file, index=False)

    print(f"Input rows: {before_rows}")
    print(f"Rows after lag/rolling warmup drop: {len(featured_df)}")
    print(f"Total columns: {featured_df.shape[1]}")
    print("New feature columns:", [c for c in featured_df.columns if "lag_" in c or "roll_" in c or c in ("day_of_week", "day_of_year", "month", "is_weekend", "season")])
    print(f"Saved feature dataset to: {out_file}")
    return featured_df


if __name__ == "__main__":
    build_features()