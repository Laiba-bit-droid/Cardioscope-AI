# 🫀 Cardio-3D Nexus AI

### Multimodal ML Predictive Pipeline & Dynamic Interactive Multi-Vessel Stenosis Spatial Visualizer
*Developed as a Healthcare Decision-Support Node | MPhil Physiology Research Initiative*

---

##System Overview
Cardio-3D Nexus bridges the critical workflow gap between abstract statistical cardiac risk percentages and intuitive anatomical visualization. Raw numbers often fail to give clinicians or patients a clear structural understanding of developing localized pathology. 

This ecosystem implements an **Advanced Ensemble Machine Learning Pipeline** paired with a highly responsive, custom-shaded browser engine to dynamically predict and map multi-vessel coronary artery disease (CAD) stenosis probabilities straight onto an interactive anatomical myocardial volume workspace.

---

##Key Out-of-the-Box Features
1. **Anti-Leakage Data Processing Matrix:** To strictly enforce diagnostic validation criteria, all direct invasive targets and angiography markers (`LAD`, `LCX`, `RCA`, `Cath`) were explicitly pruned from the training parameter indices, guaranteeing inference relies solely on non-invasive biomarkers.
2. **Automated Medication 'What-If' Simulator:** Features an integrated clinical therapeutic dropdown (Statins / Beta-Blockers). Intercepts dynamic input variables before inference, simulating standard physiological threshold relief and calculating the reduced risk profile.
3. **Triple Vessel Disease (TVD) Emergency Warning Engine:** Flashes a neon warning alert banner instantly when critical occlusion limits (≥60%) are breached across the LAD, LCX, and RCA tree segments simultaneously.
4. **Explainable AI (XAI Node):** Embedded horizontal **SHAP (SHapley Additive exPlanations)** attribution charts that parse mathematical contribution weights per incoming patient parameter in real-time.

---

##Evaluation & Validation Metrics (Stratified 5-Fold CV)
The model backend uses layered ensemble trees configured across stratified cross-validation routines to protect against over-fitting on baseline clinical parameters.

| Target Node Branch | Validation Accuracy | Mean AUROC Vector | F1-Score Optima | Clinical Sensitivity (Recall) |
| :--- | :--- | :--- | :--- | :--- |
| **Overall CAD Status** | 88.45% | **0.93** | 0.864 | 89.2% |
| **LAD Branch Stenosis** | 84.10% | **0.85** | 0.812 | 86.5% |
| **LCX Branch Stenosis** | 76.80% | **0.71** | 0.710 | 73.0% |
| **RCA Branch Stenosis** | 78.20% | **0.73** | 0.735 | 75.4% |

*Note: Model documentation acknowledges natural data skew adjustments on the LCX and RCA branches inherent to the real-world dataset feature distribution constraints.*

---

##Repository Architecture & Tech Stack
```text
├── backend/
│   ├── main.py                 # FastAPI operational endpoints engine
│   ├── train.py                # Pre-processing, scaling, and Stratified K-Fold pipeline
│   ├── requirements.txt        # Backend clinical dependencies array
│   └── artifacts/              # Serialized XGBoost/RandomForest weights and SHAP graphs
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── CardiacDashboard.tsx   # Core premium responsive user interface forms
    │   │   └── HeartCanvas.tsx        # Three.js client-side 3D volumetric pulse view
```
- **Backend:** Python, FastAPI, Scikit-Learn, XGBoost, Pandas, Numpy.
- **Frontend:** React, TypeScript, Three.js, React Three Fiber, Tailwind CSS.

---

##Local Deployment Workflow

### 1. Launch Machine Learning Backend Core
Navigate to the localized backend structural directory, initialize packages, and start the local API listener node:
```bash
cd backend
pip install -r requirements.txt
python main.py
```
*The endpoint will bind securely to `http://127.0.0.1:8000` exposing the inference engine pipeline.*

### 2. Launch Client UI Workspace
Open a secondary terminal workspace console to fire up the native client server:
```bash
cd frontend
npm install --legacy-peer-deps
npm run dev
```

---

## Clinical Safety Disclaimer
This operational interactive prototype functions strictly for educational testing and clinical spec workflows. It is an active screening decision-support proxy and does NOT function as a substitute for standard invasive coronary angiography or formal clinical imaging pathways.
