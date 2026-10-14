export type Vessel = 'LAD' | 'LCX' | 'RCA';
export type Target = 'overall' | Vessel;
export type ModelKey = 'ensemble' | 'xgboost' | 'svm' | 'rf' | 'mlp';
export type ClinicalInputs = {
  age: number; pressure: number; cholesterol: number; st: string; t: string; rwma: string;
  sex: string; diabetes: string; hypertension: string; smoker: string; family: string; chestPain: string; ef: number; fbs: number;
};
export const DEFAULT_INPUTS: ClinicalInputs = {
  age: 58, pressure: 145, cholesterol: 240, st: 'absent', t: 'present', rwma: 'none',
  sex: 'male', diabetes: 'no', hypertension: 'no', smoker: 'no', family: 'no', chestPain: 'no', ef: 50, fbs: 98,
};
export const MODEL_OPTIONS: [ModelKey, string][] = [['xgboost', 'XGBoost'], ['ensemble', 'Ensemble'], ['svm', 'SVM'], ['rf', 'RF'], ['mlp', 'MLP']];
export function riskLevel(score: number) { return score >= 60 ? 'high' : score >= 35 ? 'moderate' : 'low'; }

// ---- What-if engine: illustrative, literature-typical effect sizes (not patient-specific) ----
export type Drug = 'none' | 'statin' | 'beta' | 'combo';
export const DRUGS: { value: Drug; label: string; note: string }[] = [
  { value: 'none', label: 'No intervention (baseline)', note: '' },
  { value: 'statin', label: 'Statins (Atorvastatin 20 mg)', note: 'Total cholesterol ≈ −30%' },
  { value: 'beta', label: 'Beta-blockers (Metoprolol)', note: 'Systolic BP ≈ −12 mmHg' },
  { value: 'combo', label: 'Statin + Beta-blocker', note: 'Cholesterol −30%, BP −12 mmHg' },
];
export function applyIntervention(base: ClinicalInputs, drug: Drug): ClinicalInputs {
  const s = drug === 'statin' || drug === 'combo', b = drug === 'beta' || drug === 'combo';
  return { ...base, cholesterol: s ? Math.max(100, Math.round(base.cholesterol * 0.7)) : base.cholesterol, pressure: b ? Math.max(90, base.pressure - 12) : base.pressure };
}

// ---- Exploratory clinical risk matrix (guideline reference ranges) ----
export type Status = 'normal' | 'borderline' | 'high';
export function riskMatrix(i: ClinicalInputs) {
  const rows: { name: string; value: string; range: string; status: Status; load: number }[] = [
    { name: 'Systolic BP', value: `${i.pressure} mmHg`, range: '<120 normal · ≥140 hypertension (WHO/ESC)', status: i.pressure >= 140 ? 'high' : i.pressure >= 120 ? 'borderline' : 'normal', load: i.pressure / 120 * 100 },
    { name: 'Total cholesterol', value: `${i.cholesterol} mg/dL`, range: '<200 desirable · ≥240 high (NCEP/AHA)', status: i.cholesterol >= 240 ? 'high' : i.cholesterol >= 200 ? 'borderline' : 'normal', load: i.cholesterol / 200 * 100 },
    { name: 'Fasting glucose', value: `${i.fbs} mg/dL`, range: '<100 normal · ≥126 diabetes (WHO)', status: i.fbs >= 126 ? 'high' : i.fbs >= 100 ? 'borderline' : 'normal', load: i.fbs / 100 * 100 },
    { name: 'Ejection fraction', value: `${i.ef} %`, range: '≥55 normal · ≤40 reduced (ASE)', status: i.ef <= 40 ? 'high' : i.ef < 55 ? 'borderline' : 'normal', load: 55 / Math.max(i.ef, 1) * 100 },
  ];
  const htn = i.pressure >= 140 ? 'Structural hypertension strain likely' : i.pressure >= 120 ? 'Elevated blood-pressure load' : 'No hypertensive strain';
  const flags = [i.cholesterol >= 240, i.fbs >= 126 || i.diabetes === 'yes', i.smoker === 'yes'].filter(Boolean).length;
  const met = flags >= 2 ? 'Metabolic strain state: yes' : flags === 1 ? 'Metabolic strain: mild' : 'No metabolic strain';
  return { rows, htn, met, htnHigh: i.pressure >= 140, metHigh: flags >= 2 };
}
