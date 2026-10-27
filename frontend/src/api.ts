export type ModelStatus = {
  task: "classifier" | "segmenter" | string;
  status: "ready" | "missing" | "invalid" | "runtime_unavailable" | string;
  model_name?: string | null;
  detail: string;
  weights_size_bytes?: number | null;
};

export type ScanInfo = {
  filename: string;
  input_type: string;
  width?: number | null;
  height?: number | null;
  slices?: number | null;
  modality?: string | null;
  note: string;
};

export type Segmentation = {
  mask_png_base64: string;
  mask_overlay_png_base64?: string;
  probability_png_base64?: string;
  threshold: number;
  positive_pixels: number;
  image_pixels: number;
  coverage_percent: number;
  bounding_box: null | { x_min: number; y_min: number; x_max: number; y_max: number };
  model_name: string;
  checkpoint: string;
  mask_semantics: string;
};

export type Explanation = {
  method: string;
  target_class: string;
  target_score: number;
  heatmap_png_base64: string;
  region_bbox: null | { x_min: number; y_min: number; x_max: number; y_max: number };
  region_threshold: number;
  peak_activation: number;
  supporting_evidence: string;
  segmentation_overlap_iou?: number;
  segmentation_overlap_pixels?: number;
  segmentation_overlap_note?: string;
};

export type Analysis = {
  scan: ScanInfo;
  classification: null | {
    label: string;
    scores: Record<string, number>;
    score_semantics: string;
    score_margin: number | null;
    model_name: string;
    checkpoint: string;
  };
  segmentation: Segmentation | null;
  explanation: Explanation | null;
  source_png_base64: string | null;
  models: ModelStatus[];
  warnings: string[];
  research_use_only: boolean;
};

export type EvaluationReport = {
  task: "classifier" | "segmenter" | string;
  model_name?: string;
  architecture?: string;
  split: string;
  dataset?: { id?: string; manifest?: string; test_rows?: number };
  created_utc: string;
  metrics: Record<string, any>;
  limitations?: string[];
};

export type EvaluationSummary = {
  available: boolean;
  detail: string;
  metrics?: Record<string, EvaluationReport> | null;
};

export type PatientDetails = {
  name: string;
  age: number;
  email: string;
  phone: string;
  date_of_birth: string | null;
};

export type AdminSession = { access_token: string; expires_in: number };

export type ActivityRecord = {
  at: string;
  kind: string;
  role: string;
  result_label?: string;
  model_score_percent?: number | null;
  class_scores_percent?: Record<string, number>;
  score_margin_percent?: number | null;
  score_semantics?: string | null;
  input_type?: string | null;
  width?: number | null;
  height?: number | null;
  slices?: number | null;
  classification_model?: string | null;
  segmentation_model?: string | null;
  mask_generated?: boolean;
  has_positive_mask?: boolean;
  mask_coverage_percent?: number | null;
  mask_bbox?: null | { x_min: number; y_min: number; x_max: number; y_max: number };
  mask_threshold?: number | null;
  gradcam_generated?: boolean;
  gradcam_bbox?: null | { x_min: number; y_min: number; x_max: number; y_max: number };
  gradcam_mask_overlap_iou?: number | null;
  warning_count?: number;
};

export type AdminActivity = {
  started_at: string;
  patient_sessions_started: number;
  active_patient_sessions: number;
  admin_signins: number;
  active_admin_sessions: number;
  scans_analyzed: number;
  masks_with_positive_regions: number;
  gradcam_maps_produced: number;
  result_counts: Record<string, number>;
  recent: ActivityRecord[];
  retention_note: string;
};

export const IS_HOSTED_DEMO = import.meta.env.VITE_PUBLIC_DEMO === "true";
const API = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api").trim().replace(/\/+$/, "");

async function readJson<T>(response: Response, fallback: string): Promise<T> {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : fallback);
  return data as T;
}

export async function getHealth() {
  return readJson<{ status: string; version: string; models: ModelStatus[] }>(
    await fetch(API + "/health"), "API did not respond.",
  );
}

export async function loginAdmin(username: string, password: string) {
  return readJson<AdminSession>(await fetch(API + "/auth/admin/login", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  }), "Admin sign-in failed.");
}

export async function logoutAdmin(token: string) {
  await readJson<{ ok: boolean }>(await fetch(API + "/auth/admin/logout", {
    method: "POST", headers: { Authorization: `Bearer ${token}` },
  }), "Could not end the administrator session.");
}

export async function startPatientActivitySession() {
  return readJson<{ session_id: string }>(await fetch(API + "/activity/patient-session/start", { method: "POST" }), "Activity tracking is unavailable.");
}

export async function endPatientActivitySession(sessionId: string) {
  await readJson<{ ok: boolean }>(await fetch(API + "/activity/patient-session/end", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  }), "Could not close the patient activity session.");
}

export async function getAdminActivity(token: string) {
  return readJson<AdminActivity>(await fetch(API + "/admin/activity", { headers: { Authorization: `Bearer ${token}` } }), "Admin activity is unavailable.");
}

export async function getEvaluation(token: string) {
  return readJson<EvaluationSummary>(await fetch(API + "/evaluation", { headers: { Authorization: `Bearer ${token}` } }), "Evaluation status is unavailable.");
}

export async function runEvaluation(token: string) {
  return readJson<EvaluationSummary>(await fetch(API + "/evaluation/run", { method: "POST", headers: { Authorization: `Bearer ${token}` } }), "Could not rerun evaluation.");
}

export async function analyzeFile(file: File, patientSessionId?: string | null): Promise<Analysis> {
  const data = new FormData();
  data.append("file", file);
  const headers = patientSessionId ? { "X-Patient-Session": patientSessionId } : undefined;
  return readJson<Analysis>(await fetch(API + "/analyze", { method: "POST", body: data, headers }), "Could not process this scan.");
}

export async function downloadReport(result: Analysis, patient: PatientDetails) {
  const response = await fetch(API + "/reports/analysis", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis: result, patient_details: patient }),
  });
  if (!response.ok) throw new Error("Could not create the report.");
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "neurolab-patient-mri-summary.pdf";
  link.click();
  URL.revokeObjectURL(url);
}

export function downloadEvaluation(records: EvaluationSummary) {
  const blob = new Blob([JSON.stringify(records, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "neurolab-held-out-evaluation.json";
  link.click();
  URL.revokeObjectURL(url);
}
