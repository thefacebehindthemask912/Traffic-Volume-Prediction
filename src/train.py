import argparse, json, os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from .data_prep import prepare_dataset

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    accuracy = r2 * 100 if r2 > 0 else 0
    return {"mae": float(mae), "rmse": float(rmse), "r2_score": float(r2), "accuracy": float(accuracy)}

def main(args):
    _, _, _, X_train, y_train, X_test, y_test = prepare_dataset(args.data_path)

    results = {}
    models = {}

    # Linear Regression baseline
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results["LinearRegression"] = evaluate(y_test, lr_pred)
    models["LinearRegression"] = lr

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["RandomForest"] = evaluate(y_test, rf_pred)
    models["RandomForest"] = rf

    # XGBoost
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        results["XGBoost"] = evaluate(y_test, xgb_pred)
        models["XGBoost"] = xgb
    except Exception as e:
        print("XGBoost unavailable:", e)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model = models[best_name]
    joblib.dump(best_model, os.path.join(args.out_dir, "best_model.joblib"))

    print("Saved metrics:", json.dumps(results, indent=2))
    print("Best model:", best_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()
    main(args)
