"""Train 4 models (XGBoost, SVM-RBF, Random Forest, PyTorch MLP) x 4 targets (Overall CAD, LAD, LCX, RCA).
Validation: stratified 5-fold CV. Leakage columns (LAD, LCX, RCA, Cath) are never used as inputs.
Run:  python train.py"""
import json, warnings, numpy as np, joblib, shap
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from common import *
warnings.filterwarnings("ignore")

raw = load_raw(); X = encode(raw); feats = list(X.columns); Xv = X.values
print(f"{len(raw)} patients, {len(feats)} input features (leakage cols removed: {LEAKAGE})")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
metrics, oof_all, final = {}, {}, {}

def score(y, p):
    yh = (p >= .5).astype(int)
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh), f1=f1_score(y, yh), roc_auc=roc_auc_score(y, p))

for t in TARGETS:
    y = labels(raw, t); names = MODEL_NAMES + ["ensemble"]
    folds = {n: [] for n in names}; oof = {n: np.zeros(len(y)) for n in names}
    for tr, te in skf.split(Xv, y):
        P = {}
        for n, m in make_models(feats).items():
            P[n] = m.fit(Xv[tr], y[tr]).predict_proba(Xv[te])[:, 1]
        P["ensemble"] = np.mean([P[n] for n in MODEL_NAMES], axis=0)
        for n in names:
            folds[n].append(score(y[te], P[n])); oof[n][te] = P[n]
    metrics[t] = {n: {k: {"mean": float(np.mean([f[k] for f in folds[n]])), "std": float(np.std([f[k] for f in folds[n]]))}
                      for k in folds[n][0]} | {"n_pos": int(y.sum()), "n": int(len(y))} for n in names}
    oof_all[t] = (y, oof)
    final[t] = {n: m.fit(Xv, y) for n, m in make_models(feats).items()}
    print(t.ljust(8), " ".join(f"{n}:{metrics[t][n]['roc_auc']['mean']:.3f}" for n in names), "(CV ROC-AUC)")

# Global importance (mean |SHAP|, XGBoost) + defaults/ranges used by the API
glob = {}
for t in TARGETS:
    sv = shap.TreeExplainer(final[t]["xgboost"]).shap_values(Xv)
    imp = np.abs(sv).mean(0); order = np.argsort(-imp)[:15]
    glob[t] = [{"feature": feats[i], "mean_abs_shap": float(imp[i])} for i in order]
meta = {"features": feats, "defaults": X.median().to_dict(), "min": X.min().to_dict(), "max": X.max().to_dict(),
        "global_importance": glob, "n_patients": int(len(raw)), "leakage_excluded": LEAKAGE}
ART.mkdir(exist_ok=True)
json.dump(meta, open(ART / "meta.json", "w"), indent=1); json.dump(metrics, open(ART / "metrics.json", "w"), indent=1)
joblib.dump(final, ART / "models.joblib", compress=3)
np.save(ART / "background.npy", shap.kmeans(Xv, 10).data)

# Markdown table + plots for the documentation
with open(ART / "results.md", "w") as f:
    f.write("| Target | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\n|---|---|---|---|---|---|---|\n")
    for t in TARGETS:
        for n in MODEL_NAMES + ["ensemble"]:
            m = metrics[t][n]; f.write(f"| {t} | {n} | " + " | ".join(f"{m[k]['mean']:.3f}±{m[k]['std']:.3f}" for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]) + " |\n")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for a, t in zip(ax, TARGETS):
        y, oof = oof_all[t]
        for n in MODEL_NAMES + ["ensemble"]:
            fpr, tpr, _ = roc_curve(y, oof[n]); a.plot(fpr, tpr, lw=2.4 if n == "ensemble" else 1.2, label=f"{n} {roc_auc_score(y, oof[n]):.2f}")
        a.plot([0, 1], [0, 1], "k--", lw=.6); a.set_title(f"{t} (5-fold CV)"); a.set_xlabel("FPR"); a.set_ylabel("TPR"); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(ART / "roc_curves.png", dpi=130); plt.close()
    g = glob["overall"][:12][::-1]; plt.figure(figsize=(6, 4.2)); plt.barh([i["feature"] for i in g], [i["mean_abs_shap"] for i in g], color="#ff7a45")
    plt.title("Global importance - Overall CAD (mean |SHAP|, XGBoost)"); plt.tight_layout(); plt.savefig(ART / "global_shap.png", dpi=130)
except Exception as e:
    print("plot skipped:", e)
print("Saved artifacts to", ART)
