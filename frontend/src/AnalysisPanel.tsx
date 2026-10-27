import { useState } from "react";
import { AlertTriangle, ArrowDownToLine, Check, LoaderCircle, ShieldCheck } from "lucide-react";
import { downloadReport, IS_HOSTED_DEMO, type Analysis, type PatientDetails } from "./api";

type Props = {
  result: Analysis | null;
  error: string;
  busy: boolean;
  hasFile: boolean;
  accepted: boolean;
  patient: PatientDetails | null;
  onAccept: (value: boolean) => void;
  onRun: () => void;
};

const tumorName = (label: string) => ({
  glioma: "Glioma",
  meningioma: "Meningioma",
  pituitary: "Pituitary tumor",
  no_tumor: "No tumor pattern identified",
}[label] ?? label.replace(/_/g, " ").replace(/\b\w/g, (character) => character.toUpperCase()));

function displayedRegion(result: Analysis) {
  const box = result.segmentation?.bounding_box;
  const width = result.scan.width;
  const height = result.scan.height;
  if (!box || !width || !height) return null;
  const x = (box.x_min + box.x_max) / 2 / width;
  const y = (box.y_min + box.y_max) / 2 / height;
  const horizontal = x < 1 / 3 ? "left" : x > 2 / 3 ? "right" : "";
  const vertical = y < 1 / 3 ? "upper" : y > 2 / 3 ? "lower" : "";
  return [vertical, horizontal].filter(Boolean).join("-") || "center";
}

export function AnalysisPanel({ result, error, busy, hasFile, accepted, patient, onAccept, onRun }: Props) {
  const [reportError, setReportError] = useState("");
  const [reportBusy, setReportBusy] = useState(false);
  const classification = result?.classification;
  const segmentation = result?.segmentation;
  const explanation = result?.explanation;
  const predictedScore = classification ? classification.scores[classification.label] : null;
  const maskRegion = result ? displayedRegion(result) : null;
  const canReport = Boolean(patient && result && (classification || segmentation));

  const saveReport = async () => {
    if (!result || !patient) return;
    setReportBusy(true);
    try { await downloadReport(result, patient); setReportError(""); }
    catch (e) { setReportError(e instanceof Error ? e.message : "Could not create the patient summary."); }
    finally { setReportBusy(false); }
  };

  return <aside className="analysis-column patient-result-column">
    <section className="panel analysis-panel patient-result-panel">
      <div className="patient-result-heading"><span className="result-check"><Check size={17} /></span><div><h2>{patient ? "Your scan summary" : "Scan summary"}</h2><p>{patient ? `For ${patient.name}` : "Administrator demonstration"}</p></div></div>
      {!result ? <div className="patient-empty"><p>Choose a 2D MRI image to prepare an AI-assisted summary.</p><span>Your original scan stays on this device until you request analysis.</span></div> : <>
        <div className="patient-finding-card">
          <span className="patient-field-label">WHAT THE MODEL SUGGESTED</span>
          <strong>{classification ? tumorName(classification.label) : "A result is not available"}</strong>
          {classification?.label === "no_tumor" && <small>The classifier did not identify a typical tumor-class pattern in this image. This cannot rule out a tumor or replace radiology review.</small>}
          {classification && classification.label !== "no_tumor" && <small>This is a model category suggestion from one 2D image, not a confirmed tumor type. A clinician must interpret the full MRI.</small>}
        </div>
        {classification && <div className="patient-score-card">
          <div><span className="patient-field-label">AI MODEL SCORE</span><strong>{typeof predictedScore === "number" && Number.isFinite(predictedScore) ? `${Math.round(predictedScore * 100)}%` : "Unavailable"}</strong></div>
          <p>Score for the model's top category. It is uncalibrated and is not the chance that you have a tumor, a measure of severity, or a diagnosis.</p>
        </div>}
        <div className="patient-location-card">
          <span className="patient-field-label">WHERE ON THE SCAN</span>
          <p>{segmentation?.bounding_box ? `The coral highlight marks the ${maskRegion ? `${maskRegion} part` : "area"} selected by the model in this displayed image.` : segmentation ? "The model did not highlight an area in this image." : "A highlighted area was not available for this image."}</p>
          {segmentation && <small>Predicted mask coverage: {segmentation.coverage_percent.toFixed(2)}% of this image. Pixel coverage is not tumor size or severity.</small>}
          {segmentation?.bounding_box && <small>Screen position does not establish anatomical location or the patient's left or right side. The mask and category are separate model outputs; the mask does not confirm tumor type.</small>}
          {segmentation && !segmentation.bounding_box && segmentation.probability_png_base64 && <small>Use the advanced mask threshold control to inspect a wider or narrower candidate region. Changing it only changes the preview.</small>}
          {!segmentation && result && <small>The segmentation model did not return a mask for this image. Check the model status and warnings in the administrator activity view.</small>}
        </div>
        {patient && explanation && <div className="patient-location-card patient-explanation-card">
          <span className="patient-field-label">WHAT INFLUENCED THE AI SUGGESTION</span>
          <p>Grad-CAM marks image areas that influenced the model's category score. It cannot explain why a tumor developed or confirm that a tumor is present. Choose <b>Grad-CAM</b> in the image controls to view it.</p>
        </div>}
        {patient && classification && <details className="patient-cause-info">
          <summary>Why do brain tumors develop?</summary>
          <div>
            <p>This is general information, not an explanation of your scan. The exact cause of most brain tumors is unknown. Prior radiation treatment to the head and some rare inherited conditions can increase risk for certain types, but a risk factor does not prove what caused an individual tumor.</p>
            <p>This AI cannot determine whether a tumor is present or identify its cause. A qualified clinician can explain confirmed findings and whether personal risk factors are relevant.</p>
            <a href="https://www.cancer.gov/types/brain/patient/adult-brain-treatment-pdq" target="_blank" rel="noreferrer">National Cancer Institute: adult brain and spinal cord tumors</a>
          </div>
        </details>}
        <div className="patient-result-note">AI results can be wrong. Please ask a qualified clinician to review your MRI and explain what it means for you.</div>
        {canReport && <button className="secondary-button report-button patient-report-button" onClick={saveReport} disabled={reportBusy}>{reportBusy ? <LoaderCircle className="spin" size={15} /> : <ArrowDownToLine size={15} />} Download patient summary</button>}
      </>}
      {(error || reportError) && <div className="error-message patient-error"><AlertTriangle size={15} />{error || reportError}</div>}
      <label className="consent-check patient-consent"><input type="checkbox" checked={accepted} onChange={(event) => onAccept(event.target.checked)} /><span>I understand this is an experimental AI result, not a diagnosis.</span></label>
      <button className="primary-button patient-run-button" disabled={busy || !accepted || !hasFile} onClick={onRun}>{busy ? <LoaderCircle className="spin" size={16} /> : <Check size={16} />}{busy ? "Reviewing scan…" : result ? "Review scan again" : "Prepare scan summary"}</button>
      <div className="privacy-foot patient-privacy"><ShieldCheck size={14} /><span>{IS_HOSTED_DEMO ? "Your scan is transmitted to this hosted demo's API for analysis and discarded after the request. Use public, de-identified sample images only." : "Your scan is sent to this computer's local app for analysis and is not saved by it."}</span></div>
    </section>
  </aside>;
}
