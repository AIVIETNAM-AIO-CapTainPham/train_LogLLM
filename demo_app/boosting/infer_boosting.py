"""Inference boosting (XGBoost / CatBoost) trên held-out, 2 kiểu chia window.

Dùng chung weights đã train (model/), 2 cách "split data":
  - time-window : sliding 5 phút / step 1 phút  (giống pipeline gốc đã train)
  - line-window : 100 dòng / session            (giống LogLLM)

Mỗi window -> vector đếm 227 EventIdx -> tfidf.transform -> model.predict_proba.
df đầu vào (heldout_boosting.parquet): cột [Time, Label(0/1), EventIdx].
"""

import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path("/home/angle/jupyterlab/DEV/LogLLM_train/LogLLM")
MODEL_DIR = ROOT / "train_boosting" / "BGL_log_error_classification" / "model"
HELDOUT_PARQUET = ROOT / "demo_app" / "heldout_boosting.parquet"


def load_artifacts():
    """Trả về dict: xgb, cat, tfidf, n_templates."""
    xgb = XGBClassifier()
    xgb.load_model(str(MODEL_DIR / "xgboost_model.json"))
    cat = CatBoostClassifier()
    cat.load_model(str(MODEL_DIR / "catboost_model.cbm"))
    tfidf = joblib.load(MODEL_DIR / "tfidf.joblib")
    order = json.loads((MODEL_DIR / "template_event_ids.json").read_text())
    return {"xgb": xgb, "cat": cat, "tfidf": tfidf, "n_templates": len(order)}


def _normalize_template(t):
    t = str(t).strip()
    if t.startswith("(") and t.endswith(")"):
        try:
            return " ".join(str(x) for x in ast.literal_eval(t))
        except (ValueError, SyntaxError):
            pass
    return t


def idx_to_template():
    """idx (0..226) -> text template (để hiện chi tiết window)."""
    order = json.loads((MODEL_DIR / "template_event_ids.json").read_text())
    tdf = pd.read_csv(MODEL_DIR / "templates.csv")
    eid2txt = {str(e): _normalize_template(t) for e, t in zip(tdf["EventId"], tdf["EventTemplate"])}
    return {i: eid2txt.get(str(eid), "?") for i, eid in enumerate(order)}


def load_heldout():
    df = pd.read_parquet(HELDOUT_PARQUET)
    df["Time"] = pd.to_datetime(
        df["Time"], format="%Y-%m-%d-%H.%M.%S.%f", errors="coerce"
    )
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return df


def build_time_windows(df, n_templates, window_minutes=5, step_minutes=1):
    """Sliding time window -> (X_counts, y, meta_df). Giống NB4 cell 6."""
    window = pd.Timedelta(minutes=window_minutes)
    step = pd.Timedelta(minutes=step_minutes)
    time_arr = df["Time"].to_numpy(dtype="datetime64[ns]")
    event_idx_arr = df["EventIdx"].to_numpy(dtype=np.int32)
    anom_arr = df["Label"].to_numpy(dtype=np.int8)
    anom_prefix = np.concatenate(([0], np.cumsum(anom_arr)))

    start = df["Time"].min().floor("min")
    last = df["Time"].max() - window
    starts = pd.date_range(start=start, end=last, freq=step)

    X, y, meta = [], [], []
    for ws in starts:
        we = ws + window
        left = np.searchsorted(time_arr, np.datetime64(ws), side="left")
        right = np.searchsorted(time_arr, np.datetime64(we), side="left")
        if right <= left:
            continue
        idx = event_idx_arr[left:right]
        idx = idx[idx >= 0]
        X.append(np.bincount(idx, minlength=n_templates).astype(np.float32))
        y.append(int((anom_prefix[right] - anom_prefix[left]) > 0))
        meta.append((str(ws), str(we), int(right - left)))
    meta_df = pd.DataFrame(meta, columns=["window_start", "window_end", "num_logs"])
    return np.vstack(X), np.array(y, dtype=np.int8), meta_df


def build_line_windows(df, n_templates, window=100, step=100):
    """Fixed-size 100 dòng/session -> (X_counts, y, meta_df). Giống LogLLM."""
    event_idx_arr = df["EventIdx"].to_numpy(dtype=np.int32)
    anom_arr = df["Label"].to_numpy(dtype=np.int8)
    n = len(df)
    X, y, meta = [], [], []
    for s in range(0, n, step):
        e = min(s + window, n)
        if e <= s:
            continue
        idx = event_idx_arr[s:e]
        idx = idx[idx >= 0]
        X.append(np.bincount(idx, minlength=n_templates).astype(np.float32))
        y.append(int(anom_arr[s:e].max() > 0))
        meta.append((s, e, int(e - s)))
    meta_df = pd.DataFrame(meta, columns=["line_start", "line_end", "num_logs"])
    return np.vstack(X), np.array(y, dtype=np.int8), meta_df


def predict(art, model_name, X_counts):
    """X_counts (n, 227) -> prob anomaly. model_name: 'xgb' | 'cat'."""
    X_tfidf = art["tfidf"].transform(X_counts)
    X_dense = X_tfidf.toarray().astype(np.float32)
    model = art[model_name]
    return model.predict_proba(X_dense)[:, 1]


def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # AUC chỉ tính được khi có cả 2 lớp
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out
