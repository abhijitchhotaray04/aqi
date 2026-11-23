from pathlib import Path
import pandas as pd


def load_and_inspect(csv_path: str | Path, date_col: str = "Date", city_col: str | None = "City", city_name: str | None = None) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Place your CSV inside data/raw/ first.")

    df = pd.read_csv(path)
    print(f"Loaded shape: {df.shape}")
    print("Columns:", list(df.columns))

    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found. Check printed columns above.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    invalid_dates = df[date_col].isna().sum()
    if invalid_dates:
        print(f"Warning: {invalid_dates} rows had unparseable dates and will be dropped.")
        df = df.dropna(subset=[date_col])

    if city_col and city_col in df.columns:
        print("Available cities/stations (top 10):", df[city_col].value_counts().head(10).to_dict())
        if city_name:
            df = df[df[city_col].astype(str).str.strip().str.lower() == city_name.strip().lower()].copy()
            print(f"Filtered to '{city_name}' -> shape: {df.shape}")

    df = df.sort_values(date_col).reset_index(drop=True)
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print("\nMissing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0])
    print("\nFirst 5 rows:")
    print(df.head())
    return df


if __name__ == "__main__":
    # Update date_col / city_name to match your CSV columns
    load_and_inspect(
        csv_path="data/raw/aqi_data.csv",
        date_col="Date",
        city_col="City",
        city_name=None,  # e.g. "Bhubaneswar" or "Delhi" once you inspect cities
    )