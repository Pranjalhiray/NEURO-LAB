import { Check, ShieldCheck, X } from "lucide-react";
import { IS_HOSTED_DEMO, type ModelStatus } from "./api";

type Props = {
  models: ModelStatus[];
  evaluation: { available: boolean; detail: string } | null;
  onClose: () => void;
};

export function SafetyModal({ models, evaluation, onClose }: Props) {
  return <div className="modal-scrim" onClick={onClose}><section className="limits-modal" onClick={(e) => e.stopPropagation()}>
    <button className="icon-button modal-close" onClick={onClose} aria-label="Close"><X size={17} /></button>
    <div className="modal-mark"><ShieldCheck size={19} /></div>
    <div className="eyebrow">MODEL READINESS & SAFETY</div><h2>Research workstation</h2>
    <p>This app is a research prototype. It does not diagnose disease and must not guide treatment or triage.</p>
    <div className="model-status-list">{models.map((model) => <div key={model.task}><i className={"status-dot " + model.status} /><strong>{model.task === "classifier" ? "Classification" : "Segmentation"}</strong><span>{model.status === "ready" ? model.model_name : model.detail}</span><em>{model.status.toUpperCase()}</em></div>)}</div>
    <div className="modal-section"><strong>Evaluation record</strong><span>{evaluation?.available ? "A recorded test evaluation is available." : "No test-set metrics are installed. The interface will not invent or reuse published paper results."}</span></div>
    <div className="modal-section"><strong>Input handling</strong><span>{IS_HOSTED_DEMO ? "2D images are transmitted to this hosted demo's API for inference and discarded after the request. Use only public, de-identified sample images; do not upload patient scans or personal information. NIfTI preview and metadata are read in your browser; the volume is not uploaded." : "2D images are sent only to the local API for inference and discarded after the request. NIfTI preview and metadata are read in your browser; the volume is not uploaded."}</span></div>
    <div className="modal-section"><strong>Not yet validated for clinical use</strong><span>Dataset coverage, scanner/site variation, patient-level separation, calibration, and clinical testing remain open limitations.</span></div>
    <button className="primary-button modal-done" onClick={onClose}>Understood <Check size={15} /></button>
  </section></div>;
}
