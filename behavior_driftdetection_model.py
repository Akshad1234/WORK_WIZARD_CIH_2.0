import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import joblib
import json
from datetime import datetime

# Constants
DATA_PATH = 'behavior_log.csv'
MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load and preprocess data
df = pd.read_csv(DATA_PATH)
print("Initial Data Overview:")
print(df.info())
print(df.describe())

df.dropna(inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Split features and target
X = df.drop(columns=['timestamp', 'drift_label'])
y = df['drift_label']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Class weight ratio (for imbalance)
pos_ratio = sum(y_train == 0) / sum(y_train == 1)

# Optuna objective function
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", pos_ratio * 0.8, pos_ratio * 1.2),
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "verbosity": 0
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return roc_auc_score(y_test, preds)

# Run optimization
print("🔍 Starting hyperparameter tuning with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
best_params = study.best_params
print("✅ Best hyperparameters found:", best_params)

# Train final model with best parameters
final_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss'
)
final_model.fit(X_train, y_train)

# Evaluate
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_score = roc_auc_score(y_test, y_proba)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Behavior Drift Detection")
plt.legend()
roc_path = os.path.join(MODEL_DIR, "roc_curve.png")
plt.savefig(roc_path)
plt.close()

# Save versioned model artifacts
version = datetime.now().strftime("%Y%m%d_%H%M%S")
version_dir = os.path.join(MODEL_DIR, f"v_{version}")
os.makedirs(version_dir, exist_ok=True)

# Save model and scaler
joblib.dump(final_model, os.path.join(version_dir, "xgb_model.pkl"))
joblib.dump(scaler, os.path.join(version_dir, "scaler.pkl"))

# Save best parameters
with open(os.path.join(version_dir, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=4)

# Save evaluation metrics
metrics = {
    "classification_report": report,
    "confusion_matrix": conf_matrix.tolist(),
    "roc_auc_score": roc_score
}
with open(os.path.join(version_dir, "eval_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# Save feature importance
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": final_model.feature_importances_
}).sort_values(by="importance", ascending=False)
importance_df.to_csv(os.path.join(version_dir, "feature_importance.csv"), index=False)

# Optional: Plot feature importance
xgb.plot_importance(final_model)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(version_dir, "feature_importance.png"))
plt.show()

# Save for deployment
joblib.dump(final_model, 'behavior_drift_detector.pkl')  # Global model
print("\n✅ Model and artifacts saved in:", version_dir)
print(f"🧠 ROC AUC Score: {roc_score:.4f}")
