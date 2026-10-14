import type { ClinicalInputs, ModelKey, Target } from './cardiac-risk';
export const API_URL = (import.meta as any).env?.VITE_API_URL ?? 'http://localhost:8000';
export type ShapItem = { feature: string; label: string; value: number; display: string; is_default: boolean };
export type Prediction = {
  overall_cad: number; lad_stenosis: number; lcx_stenosis: number; rca_stenosis: number; model: string;
  per_model: Record<Target, Record<string, number>>; shap_values: Record<Target, ShapItem[]>; shap_base: Record<Target, number>; shap_source: string;
};
export type MetricsResponse = { n_patients: number; metrics: Record<Target, Record<string, Record<'accuracy' | 'precision' | 'recall' | 'f1' | 'roc_auc', { mean: number; std: number }>>> };
export const toPayload = (i: ClinicalInputs, model: ModelKey) => ({
  age: i.age, bp: i.pressure, cholesterol: i.cholesterol, st_elevation: i.st, t_inversion: i.t, rwma: i.rwma,
  sex: i.sex, diabetes: i.diabetes, hypertension: i.hypertension, smoker: i.smoker, family_history: i.family, typical_chest_pain: i.chestPain, ef: i.ef, fbs: i.fbs, model,
});
export async function predict(i: ClinicalInputs, model: ModelKey, signal?: AbortSignal): Promise<Prediction> {
  const r = await fetch(`${API_URL}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(toPayload(i, model)), signal: signal ?? null });
  if (!r.ok) throw new Error(`API ${r.status}`); return r.json();
}
export async function fetchMetrics(): Promise<MetricsResponse> { const r = await fetch(`${API_URL}/metrics`); if (!r.ok) throw new Error('metrics'); return r.json(); }
