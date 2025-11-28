import argparse, os, json
import joblib
import matplotlib.pyplot as plt
import numpy as np

from .data_prep import prepare_dataset

def plot_actual_vs_pred(date_time, y_true, y_pred, out_path):
    plt.figure(figsize=(12, 5))
    plt.plot(date_time, y_true, label="Actual")
    plt.plot(date_time, y_pred, label="Predicted")
    plt.title("Traffic Volume: Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Traffic Volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_feature_importance(model, feature_names, out_path):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:25]
        names = [feature_names[i] for i in idx]
        vals = importances[idx]
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(vals)), vals[::-1])
        plt.yticks(range(len(vals)), names[::-1])
        plt.title("Feature Importance (top 25)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

def main(args):
    df, train, test, X_train, y_train, X_test, y_test = prepare_dataset(args.data_path)
    model = joblib.load(args.model_path)
    y_pred = model.predict(X_test)

    os.makedirs(args.out_dir, exist_ok=True)
    plot_actual_vs_pred(test["date_time"], y_test, y_pred, os.path.join(args.out_dir, "actual_vs_predicted.png"))
    plot_feature_importance(model, X_test.columns.to_list(), os.path.join(args.out_dir, "feature_importance.png"))
    print("Saved plots to", args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()
    main(args)
