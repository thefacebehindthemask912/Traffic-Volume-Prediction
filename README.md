# Traffic Prediction System — Advanced (Metro Interstate)

Upgraded version with:
- **Lag features** (1h, 6h, 24h)
- **Multiple Models** (Linear Regression, Random Forest, XGBoost)
- **Streamlit web app** for interactive predictions
- **Time-based train/test split** (cutoff 2018-01-01)
- **Feature importance visualization**

---

## Project Structure

```
traffic-prediction-advanced/
├── app/
│   └── streamlit_app.py          # Interactive web application
├── src/
│   ├── __init__.py               # Package initialization
│   ├── data_prep.py              # Data loading, cleaning, feature engineering
│   ├── train.py                  # Model training and evaluation
│   └── evaluate.py               # Visualization and metrics
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv  # Dataset (48,204 hourly records)
├── models/
│   ├── best_model.joblib         # Trained LinearRegression model
│   ├── metrics.json              # Model performance metrics
│   ├── actual_vs_predicted.png   # Evaluation plot
│   └── feature_importance.png    # Feature importance chart
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Dataset Information

**Metro Interstate Traffic Volume** - Hourly traffic volume data from Minneapolis-St. Paul area

- **Records**: 48,204 hourly observations
- **Date Range**: Oct 2, 2012 - Sep 30, 2018
- **Target**: `traffic_volume` (vehicles/hour) - Range: 0-7,280
- **Features**:
  - `date_time`: Timestamp of measurement
  - `temp`: Temperature (Kelvin)
  - `rain_1h`: Rain in last hour (mm)
  - `snow_1h`: Snow in last hour (mm)
  - `clouds_all`: Cloud coverage (%)
  - `weather_main`: Weather condition (Clouds, Rain, Clear, etc.)
  - `holiday`: Holiday type (if applicable)

---

## Model Performance

| Model | MAE | RMSE | R² Score | Accuracy |
|-------|-----|------|----------|----------|
| **LinearRegression** ⭐ | 77.77 | 95.46 | 0.887 | **88.73%** |
| RandomForest | 114.22 | 132.77 | 0.782 | 78.21% |
| XGBoost | 147.52 | 159.81 | 0.684 | 68.42% |

**Best Model**: LinearRegression (selected based on lowest RMSE)

---

## Quick Start

### 1. Clone and Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages**:
- pandas
- numpy
- scikit-learn
- matplotlib
- streamlit
- joblib
- xgboost

### 3. Prepare Data

Place your CSV file at: `data/Metro_Interstate_Traffic_Volume.csv`

**Expected columns**:
```
holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description, date_time, traffic_volume
```

### 4. Train Models

```bash
python -m src.train --data_path data/Metro_Interstate_Traffic_Volume.csv --out_dir models
```

**Output**:
- `models/best_model.joblib` - Trained model
- `models/metrics.json` - Performance metrics for all 3 models

### 5. Generate Evaluation Plots

```bash
python -m src.evaluate --data_path data/Metro_Interstate_Traffic_Volume.csv --model_path models/best_model.joblib --out_dir models
```

**Output**:
- `models/actual_vs_predicted.png` - Actual vs Predicted traffic volume
- `models/feature_importance.png` - Top 20 most important features

### 6. Run Web Application

```bash
python -m streamlit run app/streamlit_app.py
```

**Access**: Open your browser and go to `http://localhost:8501`

---

## Using the Streamlit App

### **Prediction Tab** 🔮

1. **Set Prediction Parameters** (left sidebar):
   - Date and Time
   - Temperature (240-320 K)
   - Rain (0-20 mm/hour)
   - Snow (0-50 mm/hour)
   - Cloud Coverage (0-100%)
   - Weather Condition
   - Holiday (if applicable)
   - Optional: Traffic lags (auto-estimated if left as 0)

2. **View Prediction**: Displays predicted traffic volume in vehicles/hour

### **Explore Tab** 📈

- View sample of test set predictions
- Actual vs Predicted traffic volume plot
- Top 20 feature importances chart

---

## Important Notes

### ⚠️ Common Issues & Solutions

1. **ImportError: attempted relative import with no known parent package**
   - Always run scripts using `-m` flag: `python -m src.train`
   - Ensures `src/` is treated as a package

2. **Feature mismatch errors**
   - Ensure your CSV has all required columns
   - Column `weather_description` is automatically dropped
   - Lag features are auto-estimated if not provided

3. **Streamlit app shows "Could not prepare data for prediction"**
   - Ensure dataset is at `data/Metro_Interstate_Traffic_Volume.csv`
   - Check that all required weather fields are present

### ✅ File Dependencies

- `src/__init__.py` must exist (created automatically)
- `models/best_model.joblib` required for predictions
- `data/Metro_Interstate_Traffic_Volume.csv` required for app

---

## Advanced Usage

### Custom Training Parameters

Edit `src/train.py` to modify:
- Number of estimators
- Learning rates
- Tree depths
- Train/test split date

### Custom Data Preparation

Modify `src/data_prep.py` for:
- Different lag periods
- Additional feature engineering
- Data cleaning thresholds
- One-hot encoding columns

---

## Technical Stack

- **Python 3.7+**
- **scikit-learn**: Machine learning models
- **pandas**: Data manipulation
- **Streamlit**: Web application framework
- **XGBoost**: Gradient boosting
- **Matplotlib**: Data visualization

---

## License

This project is provided as-is for educational purposes.

---

## Author Notes

- Linear Regression outperforms tree-based models on this dataset
- Time-based split ensures realistic evaluation (no data leakage)
- Lag features capture temporal patterns effectively
- Consider ensemble methods for production use
