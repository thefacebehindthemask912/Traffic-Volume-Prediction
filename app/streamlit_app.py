import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from src.data_prep import load_data, engineer_time_features, one_hot_encode, clean_and_prepare, add_lag_features

st.set_page_config(page_title="Traffic Volume Prediction — Advanced", layout="wide")
st.title("🚦 Traffic Volume Prediction — Advanced (Lag + XGBoost)")

model_path = Path("models/best_model.joblib")
data_path = Path("data/Metro_Interstate_Traffic_Volume.csv")

with st.sidebar:
    st.header("Prediction Inputs")
    date_input = st.date_input("Date", value=datetime(2018, 1, 2))
    time_input = st.time_input("Time", value=datetime.strptime("08:00", "%H:%M").time())
    dt = datetime.combine(date_input, time_input)
    temp = st.slider("Temperature (K)", min_value=240.0, max_value=320.0, value=275.0, step=0.5)
    rain_1h = st.slider("Rain (mm last hour)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
    snow_1h = st.slider("Snow (mm last hour)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
    clouds_all = st.slider("Clouds (%)", min_value=0, max_value=100, value=20, step=1)
    weather_main = st.selectbox("Weather", ["Clear","Clouds","Drizzle","Fog","Haze","Mist","Rain","Snow","Squall","Thunderstorm"])
    holiday = st.selectbox("Holiday", ["None","Christmas Day","Independence Day","Labor Day","Martin Luther King Jr Day","Memorial Day","New Years Day","State Fair","Thanksgiving Day","Veterans Day","Washingtons Birthday"])
    lag_1h = st.number_input("Lag 1h traffic (optional, 0 to auto-estimate)", value=0)
    lag_6h = st.number_input("Lag 6h traffic (optional, 0 to auto-estimate)", value=0)
    lag_24h = st.number_input("Lag 24h traffic (optional, 0 to auto-estimate)", value=0)

def build_row(dt, temp, rain_1h, snow_1h, clouds_all, weather_main, holiday, lag_1h, lag_6h, lag_24h):
    return pd.DataFrame([{
        "date_time": pd.to_datetime(dt),
        "temp": temp,
        "rain_1h": rain_1h,
        "snow_1h": snow_1h,
        "clouds_all": clouds_all,
        "weather_main": weather_main,
        "holiday": holiday if holiday != "None" else "None",
        "traffic_volume": np.nan,  # placeholder
        "lag_1h": lag_1h if lag_1h>0 else np.nan,
        "lag_6h": lag_6h if lag_6h>0 else np.nan,
        "lag_24h": lag_24h if lag_24h>0 else np.nan
    }])

def align_columns(df_infer, columns):
    for c in columns:
        if c not in df_infer.columns:
            df_infer[c] = 0
    return df_infer[columns]

tab_pred, tab_explore = st.tabs(["🔮 Predict", "📈 Explore & Importance"])

with tab_pred:
    if not model_path.exists():
        st.warning("Train a model first: `python -m src.train --data_path data/Metro_Interstate_Traffic_Volume.csv --out_dir models`")
        st.stop()

    model = joblib.load(model_path)
    # Build one row
    row = build_row(dt, temp, rain_1h, snow_1h, clouds_all, weather_main, holiday, lag_1h, lag_6h, lag_24h)
    
    # If we have dataset, use it to infer correct one-hot/columns and to auto-fill lags if missing
    if data_path.exists():
        df = load_data(str(data_path))
        df = engineer_time_features(df)
        df = clean_and_prepare(df)
        df = add_lag_features(df, [1, 6, 24])
        df = one_hot_encode(df, cols=("weather_main","holiday"))
        # infer common columns
        X_columns = [c for c in df.columns if c not in ("traffic_volume","date_time")]
        
        # auto-fill lags using recent historical patterns if user left zeros
        if lag_1h == 0:
            lag_1h = df[(df["hour"]==row["date_time"].dt.hour.iloc[0])]["traffic_volume"].median()
        if lag_6h == 0:
            lag_6h = df["traffic_volume"].rolling(6).median().dropna().median()
        if lag_24h == 0:
            lag_24h = df[(df["weekday"]==row["date_time"].dt.weekday.iloc[0])]["traffic_volume"].median()
    else:
        X_columns = getattr(model, "feature_names_in_", None)
        if X_columns is None:
            st.error("Cannot infer model columns without data file. Please place the CSV at data/...")
            st.stop()

    # Process the row: engineer time features, clean, encode
    row = engineer_time_features(row)
    # Don't call clean_and_prepare on prediction row - it will drop NaN traffic_volume
    # Just drop redundant columns manually
    row = row.drop(columns=["weather_description"], errors="ignore")
    # Manually add lag features (don't call add_lag_features which drops NaN)
    row["lag_1h"] = lag_1h if lag_1h > 0 else 0
    row["lag_6h"] = lag_6h if lag_6h > 0 else 0
    row["lag_24h"] = lag_24h if lag_24h > 0 else 0
    row = one_hot_encode(row, cols=("weather_main","holiday"))
    
    # Prepare features for prediction
    X = row.drop(columns=["traffic_volume","date_time"], errors="ignore")
    X = align_columns(X, X_columns)
    
    # Make prediction
    if len(X) > 0:
        pred = float(model.predict(X)[0])
        st.success(f"Predicted traffic volume: **{pred:.2f}**")
    else:
        st.error("Could not prepare data for prediction")

with tab_explore:
    if data_path.exists() and model_path.exists():
        df = load_data(str(data_path))
        df = engineer_time_features(df)
        df = clean_and_prepare(df)
        df = add_lag_features(df, [1, 6, 24])
        df = one_hot_encode(df, cols=("weather_main","holiday"))
        test = df[df["date_time"] >= pd.to_datetime("2018-01-01")].copy()
        X_test = test.drop(columns=["traffic_volume","date_time"], errors="ignore")
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        test["predicted"] = y_pred

        st.write("Sample of test predictions:")
        st.dataframe(test[["date_time","traffic_volume","predicted"]].head(500))

        fig = plt.figure(figsize=(12,4))
        plt.plot(test["date_time"].iloc[:500], test["traffic_volume"].iloc[:500], label="Actual")
        plt.plot(test["date_time"].iloc[:500], test["predicted"].iloc[:500], label="Predicted")
        plt.legend()
        plt.title("Actual vs Predicted (sample)")
        st.pyplot(fig)

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            # Align to used columns
            names = X_test.columns.to_list()
            idx = np.argsort(importances)[::-1][:20]
            st.subheader("Top 20 Feature Importances")
            st.bar_chart(pd.DataFrame({"importance": importances[idx]}, index=[names[i] for i in idx]))
    else:
        st.info("Provide dataset CSV and train the model to explore.")
