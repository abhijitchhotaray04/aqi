from pathlib import Path
import pandas as pd


def aqi_to_bucket(aqi: float) -> str | float:
    if pd.isna(aqi):
        return pd.NA
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def clean_city_aqi(
    csv_path: str | Path = "data/raw/city_day.csv",
    output_path: str | Path = "data/processed/delhi_aqi_clean.csv",
    city_name: str = "Delhi",
    max_gap_days: int = 2,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Filter to a single city and ensure unique daily timestamps
    city_df = df[df["City"].astype(str).str.strip().str.lower() == city_name.strip().lower()].copy()
    if city_df.empty:
        raise ValueError(f"No rows found for city '{city_name}'.")

    city_df = city_df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    city_df = city_df.set_index("Date")

    # Reindex to an explicit daily calendar grid
    full_idx = pd.date_range(start=city_df.index.min(), end=city_df.index.max(), freq="D")
    city_df = city_df.reindex(full_idx)
    city_df.index.name = "Date"
    city_df["City"] = city_name

    # Inspect missingness before gap filling
    raw_missing_aqi = int(city_df["AQI"].isna().sum())

    # Interpolate short gaps (up to max_gap_days) on numeric columns only
    numeric_cols = city_df.select_dtypes(include="number").columns.tolist()
    city_df[numeric_cols] = city_df[numeric_cols].interpolate(
        method="time",
        limit=max_gap_days,
        limit_area="inside",
    )

    # Drop remaining rows where target AQI is still missing (avoids long synthetic stretches)
    before_drop = len(city_df)
    cleaned = city_df.dropna(subset=["AQI"]).copy()
    cleaned["AQI_Bucket"] = cleaned["AQI"].apply(aqi_to_bucket)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned.reset_index().to_csv(out_file, index=False)

    print(f"City: {city_name}")
    print(f"Full calendar days: {len(full_idx)} ({full_idx.min().date()} to {full_idx.max().date()})")
    print(f"Missing AQI before cleaning: {raw_missing_aqi}")
    print(f"Rows after short-gap interpolation & dropping long gaps: {len(cleaned)} (dropped {before_drop - len(cleaned)} rows)")
    print(f"Remaining missing AQI: {int(cleaned['AQI'].isna().sum())}")
    print(f"Saved cleaned dataset to: {out_file}")
    return cleaned


if __name__ == "__main__":
    clean_city_aqi(
        csv_path="data/raw/aqi_data.csv",  # change to data/raw/city_day.csv if named that way
        output_path="data/processed/delhi_aqi_clean.csv",
        city_name="Delhi",
        max_gap_days=2,
    )