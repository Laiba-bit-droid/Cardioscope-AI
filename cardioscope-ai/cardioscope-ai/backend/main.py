"""Cardioscope FastAPI backend  ->  POST http://localhost:8000/predict
Run:  uvicorn main:app --reload --port 8000"""
import json, numpy as np, joblib, shap
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import common  # noqa: F401  (needed to unpickle TorchMLP)
from common import ART, TARGETS, MODEL_NAMES

MODELS = joblib.load(ART / "models.joblib")
META = json.load(open(ART / "meta.json")); METRICS = json.load(open(ART / "metrics.json"))
BG = np.load(ART / "background.npy"); FEATS = META["features"]; DEF = META["defaults"]; IDX = {f: i for i, f in enumerate(FEATS)}
LABEL = {"Age": "Age", "BP": "Blood pressure", "LDL": "Cholesterol (LDL)", "DLP": "Dyslipidemia", "HTN": "Hypertension", "St Elevation": "ST-Elevation",
         "Tinversion": "T-Inversion", "Region RWMA": "RWMA", "EF-TTE": "Ejection fraction", "FBS": "Fasting glucose", "DM": "Diabetes",
         "Typical Chest Pain": "Typical chest pain", "Sex": "Sex (male)", "Current Smoker": "Smoker", "FH": "Family history", "Atypical": "Atypical chest pain",
         "PR": "Pulse rate", "TG": "Triglycerides", "HDL": "HDL", "ESR": "ESR", "HB": "Hemoglobin", "Na": "Sodium", "K": "Potassium", "BMI": "BMI",
         "Q Wave": "Q wave", "St Depression": "ST-Depression", "LVH": "LVH", "Function Class": "Functional class", "Neut": "Neutrophils", "Lymph": "Lymphocytes"}
UNITS = {"Age": " yrs", "BP": " mmHg", "LDL": " mg/dL", "EF-TTE": " %", "FBS": " mg/dL", "PR": " bpm", "TG": " mg/dL", "HDL": " mg/dL"}

app = FastAPI(title="Cardioscope AI pipeline")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Patient(BaseModel):
    age: float = 58; bp: float = 130; cholesterol: float = 200          # total cholesterol, mg/dL
    st_elevation: str = "absent"; t_inversion: str = "absent"; rwma: str = "none"   # none | single | multiple
    sex: Optional[str] = None; diabetes: Optional[str] = None; hypertension: Optional[str] = None; smoker: Optional[str] = None
    family_history: Optional[str] = None; typical_chest_pain: Optional[str] = None; ef: Optional[float] = None; fbs: Optional[float] = None
    model: str = "ensemble"                                              # ensemble | xgboost | svm | rf | mlp


def yes(v): return str(v).strip().lower() in {"1", "yes", "y", "true", "present", "male"}


def build_row(p: Patient):
    r = dict(DEF); given = {"Age", "BP", "LDL", "DLP", "HTN", "St Elevation", "Tinversion", "Region RWMA"}
    r["Age"], r["BP"] = p.age, p.bp
    # dataset stores LDL, not total cholesterol: LDL = TC - HDL - TG/5 (Friedewald), HDL/TG at cohort median
    r["LDL"] = max(10.0, p.cholesterol - r["HDL"] - r["TG"] / 5)
    r["DLP"] = 1.0 if p.cholesterol >= 200 else 0.0                       # dyslipidemia flag from guideline cut-off
    r["HTN"] = 1.0 if (p.bp >= 140 or yes(p.hypertension)) else 0.0       # hypertension if BP >= 140 or known history
    r["St Elevation"], r["Tinversion"] = float(yes(p.st_elevation)), float(yes(p.t_inversion))
    r["Region RWMA"] = {"none": 0.0, "single": 1.0, "multiple": 3.0}.get(p.rwma, 0.0)
    for key, col, val in [("sex", "Sex", None), ("diabetes", "DM", None), ("smoker", "Current Smoker", None), ("family_history", "FH", None), ("typical_chest_pain", "Typical Chest Pain", None)]:
        v = getattr(p, key)
        if v is not None: r[col] = float(yes(v)); given.add(col)
    if p.ef is not None: r["EF-TTE"] = p.ef; given.add("EF-TTE")
    if p.fbs is not None: r["FBS"] = p.fbs; given.add("FBS")
    return np.array([[r[f] for f in FEATS]]), r, given


def disp(f, v):
    if f in UNITS: return f"{v:.0f}{UNITS[f]}"
    if META["min"][f] >= 0 and META["max"][f] <= 1: return "Yes" if v >= 0.5 else "No"
    return f"{v:.1f}"


sig = lambda z: 1 / (1 + np.exp(-z))
_kernel = {}


def tree_pp(expl, row, logit):
    sv = np.array(expl.shap_values(row)); sv = sv[..., 1] if sv.ndim == 3 else sv; v = sv[0]
    base = float(np.ravel(expl.expected_value)[-1]); tot = float(v.sum())
    if logit:
        pb, pf = sig(base), sig(base + tot); scale = (pf - pb) / tot if abs(tot) > 1e-4 else pb * (1 - pb)
        return v * scale * 100, pb * 100
    return v * 100, base * 100


def explain(t, model, row):
    """SHAP contributions in percentage points (they sum to prediction - base rate)."""
    if model in ("ensemble", "xgboost", "rf"):
        parts = []
        if model in ("ensemble", "xgboost"): parts.append(tree_pp(shap.TreeExplainer(MODELS[t]["xgboost"]), row, True))
        if model in ("ensemble", "rf"): parts.append(tree_pp(shap.TreeExplainer(MODELS[t]["rf"]), row, False))
        v = np.mean([p[0] for p in parts], 0); base = float(np.mean([p[1] for p in parts])); src = {"ensemble": "XGBoost + Random Forest (averaged TreeSHAP)", "xgboost": "XGBoost TreeSHAP", "rf": "Random Forest TreeSHAP"}[model]
    else:
        k = (t, model)
        if k not in _kernel: _kernel[k] = shap.KernelExplainer(lambda X, m=MODELS[t][model]: m.predict_proba(X)[:, 1], BG)
        e = _kernel[k]; v = np.array(e.shap_values(row, nsamples=150, silent=True)).reshape(-1) * 100; base = float(e.expected_value) * 100
        src = f"{model.upper()} KernelSHAP"
    return v, base, src


@app.post("/predict")
def predict(p: Patient):
    row, full, given = build_row(p); m = p.model if p.model in MODEL_NAMES + ["ensemble"] else "ensemble"
    per_model = {t: {n: float(MODELS[t][n].predict_proba(row)[0, 1]) * 100 for n in MODEL_NAMES} for t in TARGETS}
    for t in TARGETS: per_model[t]["ensemble"] = float(np.mean([per_model[t][n] for n in MODEL_NAMES]))
    out = {t: round(per_model[t][m], 1) for t in TARGETS}
    shap_out, bases, src = {}, {}, ""
    for t in TARGETS:
        v, base, src = explain(t, m, row); bases[t] = round(base, 1)
        top = np.argsort(-np.abs(v))[:8]
        shap_out[t] = [{"feature": FEATS[i], "label": LABEL.get(FEATS[i], FEATS[i]), "value": round(float(v[i]), 2),
                        "display": disp(FEATS[i], full[FEATS[i]]),
                        "is_default": FEATS[i] not in given} for i in top]
    return {"overall_cad": out["overall"], "lad_stenosis": out["LAD"], "lcx_stenosis": out["LCX"], "rca_stenosis": out["RCA"],
            "model": m, "per_model": per_model, "shap_values": shap_out, "shap_base": bases, "shap_source": src,
            "disclaimer": "Decision support / educational use only. Not a substitute for diagnostic imaging."}


@app.get("/metrics")
def metrics(): return {"cv": "stratified 5-fold", "n_patients": META["n_patients"], "metrics": METRICS, "global_importance": META["global_importance"]}


@app.get("/health")
def health(): return {"status": "ok", "features": len(FEATS), "models": MODEL_NAMES}


if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="127.0.0.1", port=8000)
