"""Shared preprocessing + model zoo for Cardioscope (Z-Alizadeh Sani extension dataset)."""
from pathlib import Path
import os
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "z_alizadeh_sani.xlsx"
ART = ROOT / "artifacts"
LEAKAGE = ["LAD", "LCX", "RCA", "Cath"]          # never used as inputs (target leakage)
TARGETS = {"overall": ("Cath", "CAD"), "LAD": ("LAD", "Stenotic"), "LCX": ("LCX", "Stenotic"), "RCA": ("RCA", "Stenotic")}
MODEL_NAMES = ["xgboost", "svm", "rf", "mlp"]


def load_raw():
    d = pd.read_excel(DATA, sheet_name=0)
    d.columns = [c.strip() for c in d.columns]
    return d


def encode(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[c for c in LEAKAGE if c in df.columns]).copy()
    X["Sex"] = (X["Sex"] == "Male").astype(int)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]) and set(X[c].unique()) <= {"Y", "N"}:
            X[c] = (X[c] == "Y").astype(int)
    X["VHD"] = X["VHD"].map({"N": 0, "mild": 1, "Moderate": 2, "Severe": 3})
    X["BBB_LBBB"] = (X["BBB"] == "LBBB").astype(int)
    X["BBB_RBBB"] = (X["BBB"] == "RBBB").astype(int)
    X = X.drop(columns="BBB")
    return X.loc[:, X.nunique() > 1].astype(float)   # drops constant columns (e.g. Exertional CP)


def labels(df, target):
    col, pos = TARGETS[target]
    return (df[col] == pos).astype(int).values


class TorchMLP(BaseEstimator, ClassifierMixin):
    """Small PyTorch MLP with a scikit-learn API (so CV, pickling and SHAP all work)."""
    def __init__(self, hidden=(64, 32), epochs=150, lr=2e-3, wd=1e-3, dropout=0.3, seed=42):
        self.hidden, self.epochs, self.lr, self.wd, self.dropout, self.seed = hidden, epochs, lr, wd, dropout, seed

    def fit(self, X, y):
        import torch, torch.nn as nn
        torch.manual_seed(self.seed); torch.set_num_threads(2)
        X = torch.tensor(np.asarray(X, dtype=np.float32)); y = torch.tensor(np.asarray(y, dtype=np.float32))
        layers, d = [], X.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]; d = h
        self.net_ = nn.Sequential(*layers, nn.Linear(d, 1))
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = nn.BCEWithLogitsLoss(); g = torch.Generator().manual_seed(self.seed)
        self.net_.train()
        for _ in range(self.epochs):
            for idx in torch.randperm(len(X), generator=g).split(32):
                opt.zero_grad(); loss_fn(self.net_(X[idx]).squeeze(1), y[idx]).backward(); opt.step()
        self.classes_ = np.array([0, 1]); return self

    def predict_proba(self, X):
        import torch
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.tensor(np.asarray(X, dtype=np.float32))).squeeze(1)).numpy()
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _mlp():
    import importlib.util
    if importlib.util.find_spec("torch") and os.environ.get("NO_TORCH") != "1":
        return TorchMLP()
    from sklearn.neural_network import MLPClassifier   # fallback only if PyTorch is not installed
    return MLPClassifier((64, 32), alpha=1e-2, max_iter=600, early_stopping=False, random_state=42)


MONOTONE_UP = {"Age", "BP", "LDL", "DLP", "HTN", "DM", "Current Smoker", "St Elevation", "Tinversion", "Region RWMA"}  # clinical prior


def make_models(feats=None):
    mc = tuple(1 if f in MONOTONE_UP else 0 for f in feats) if feats else None
    return {
        "xgboost": XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                                 reg_lambda=2.0, eval_metric="logloss", random_state=42, n_jobs=2, monotone_constraints=mc),
        "svm": Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=42))]),
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=2),
        "mlp": Pipeline([("sc", StandardScaler()), ("m", _mlp())]),
    }
