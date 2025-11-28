import pandas as pd
from typing import Tuple, List

TARGET_COL = "traffic_volume"
TIME_COL = "date_time"

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TIME_COL not in df.columns:
        raise ValueError(f"{TIME_COL} missing from dataframe")
    df["hour"] = df[TIME_COL].dt.hour
    df["day"] = df[TIME_COL].dt.day
    df["month"] = df[TIME_COL].dt.month
    df["weekday"] = df[TIME_COL].dt.weekday
    return df

def add_lag_features(df: pd.DataFrame, lags: List[int] = [1, 6, 24]) -> pd.DataFrame:
    df = df.copy().sort_values(TIME_COL)
    for lag in lags:
        df[f"lag_{lag}h"] = df[TARGET_COL].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

def one_hot_encode(df: pd.DataFrame, cols=("weather_main","holiday")) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    df = pd.get_dummies(df, columns=present, drop_first=True)
    return df

def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop redundant columns
    df = df.drop(columns=["weather_description"], errors="ignore")
    df = df.dropna(subset=[TARGET_COL])
    if TARGET_COL in df.columns:
        df = df[df[TARGET_COL].between(0, 10000)]
    # Clip some weather extremes to reduce wild outliers
    for c in ["temp","rain_1h","snow_1h","clouds_all"]:
        if c in df.columns:
            df[c] = df[c].clip(lower=df[c].quantile(0.001), upper=df[c].quantile(0.999))
    return df

def train_test_time_split(df: pd.DataFrame, cutoff="2018-01-01"):
    train = df[df[TIME_COL] < pd.to_datetime(cutoff)].copy()
    test  = df[df[TIME_COL] >= pd.to_datetime(cutoff)].copy()
    return train, test

def feature_target_split(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL, TIME_COL], errors="ignore")
    y = df[TARGET_COL]
    return X, y

def prepare_dataset(csv_path: str, cutoff="2018-01-01"):
    df = load_data(csv_path)
    df = engineer_time_features(df)
    df = clean_and_prepare(df)
    df = add_lag_features(df, [1, 6, 24])
    df = one_hot_encode(df, cols=("weather_main", "holiday"))
    train, test = train_test_time_split(df, cutoff=cutoff)
    X_train, y_train = feature_target_split(train)
    X_test, y_test = feature_target_split(test)
    return df, train, test, X_train, y_train, X_test, y_test
