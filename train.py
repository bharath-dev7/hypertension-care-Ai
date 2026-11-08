import os
import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                             f1_score, confusion_matrix)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False          # falls back to HistGradientBoosting below

# 1. Load the cleaned data (adjust the paths if the files were saved elsewhere)
train = pd.read_csv("train_clean.csv")
test = pd.read_csv("test_clean.csv")
X_tr, y_tr = train.drop(columns="Risk"), train["Risk"]
X_te, y_te = test.drop(columns="Risk"), test["Risk"]

# 2. Two modes: current status (with BP) and early risk (without BP-related columns)
bp_cols = ["sysBP", "diaBP", "BPMeds"]
modes = {
    "current_status": (X_tr, X_te),
    "early_risk": (X_tr.drop(columns=bp_cols), X_te.drop(columns=bp_cols)),
}

# 3. Models, with class imbalance handled
neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()

def get_models():
    models = {
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                          scale_pos_weight=neg / pos,
                                          eval_metric="logloss", random_state=42)
    else:
        models["HistGB"] = HistGradientBoostingClassifier(class_weight="balanced",
                                                          random_state=42)
    return models

# 4 and 5. Cross-validate on train, fit, then evaluate once on test
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows, best = [], {}

for mode, (Xa, Xb) in modes.items():
    for name, model in get_models().items():
        cv_auc = cross_val_score(model, Xa, y_tr, cv=cv, scoring="roc_auc").mean()
        model.fit(Xa, y_tr)
        prob = model.predict_proba(Xb)[:, 1]
        pred = (prob >= 0.5).astype(int)              # default threshold, tuned later
        rows.append({
            "mode": mode, "model": name, "cv_auc": cv_auc,
            "test_auc": roc_auc_score(y_te, prob),
            "recall": recall_score(y_te, pred),
            "precision": precision_score(y_te, pred),
            "f1": f1_score(y_te, pred),
        })
        if mode not in best or cv_auc > best[mode][0]:   # pick best by CV, not by test
            best[mode] = (cv_auc, name, model, pred)

print(pd.DataFrame(rows).round(3).to_string(index=False))

# 6. Confusion matrix and save the best model for each mode
os.makedirs("models", exist_ok=True)
for mode, (score, name, model, pred) in best.items():
    print(f"\nBest for {mode}: {name} (cv_auc={score:.3f})")
    print(confusion_matrix(y_te, pred))                  # rows = actual, cols = predicted
    joblib.dump(model, f"models/{mode}_{name}.joblib")