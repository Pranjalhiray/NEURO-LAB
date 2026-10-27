import { Activity, AlertTriangle, Brain, Eye, Layers3, RefreshCw, ShieldCheck, Users } from "lucide-react";
import type { ReactNode } from "react";
import type { ActivityRecord, AdminActivity } from "./api";

type Props = {
  activity: AdminActivity | null;
  loading: boolean;
  error: string;
  onRefresh: () => void;
};

const resultName = (label: string) => ({
  glioma: "Glioma pattern",
  meningioma: "Meningioma pattern",
  pituitary: "Pituitary pattern",
  no_tumor: "No tumor pattern identified",
  unavailable: "No classification result",
}[label] ?? label.replace(/_/g, " "));

const scoreBreakdown = (scores?: Record<string, number>) => Object.entries(scores ?? {})
  .map(([label, score]) => `${resultName(label)} ${score}%`)
  .join(" · ");

const eventName = (record: ActivityRecord) => ({
  patient_sign_in: "Patient session started",
  patient_sign_out: "Patient signed out",
  admin_sign_in: "Administrator signed in",
  scan_analysis: "MRI analysis",
}[record.kind] ?? record.kind.replace(/_/g, " "));

const boxName = (box?: ActivityRecord["mask_bbox"]) => box
  ? `${box.x_min}, ${box.y_min} → ${box.x_max}, ${box.y_max}`
  : "No region";

function Metric({ icon, label, value, detail }: { icon: ReactNode; label: string; value: number; detail: string }) {
  return <article className="activity-metric-card"><div className="activity-metric-icon">{icon}</div><div><span>{label}</span><strong>{value}</strong><small>{detail}</small></div></article>;
}

export function ActivityDashboard({ activity, loading, error, onRefresh }: Props) {
  const latest = activity?.recent ?? [];
  return <section className="activity-dashboard">
    <div className="activity-dashboard-heading">
      <div><div className="eyebrow">ADMIN CONTROL ROOM</div><h1>Activity &amp; results</h1><p>Live sign-ins and anonymized MRI model outputs since the API started.</p></div>
      <button className="secondary-button" onClick={onRefresh} disabled={loading}><RefreshCw size={14} className={loading ? "spin" : ""} /> Refresh</button>
    </div>

    {error && <div className="activity-error"><AlertTriangle size={15} />{error}</div>}
    {!activity && loading && <div className="activity-empty">Loading activity…</div>}
    {activity && <>
      <div className="activity-metric-grid">
        <Metric icon={<Users size={18} />} label="Patient sessions" value={activity.patient_sessions_started} detail={`${activity.active_patient_sessions} currently active`} />
        <Metric icon={<ShieldCheck size={18} />} label="Admin sign-ins" value={activity.admin_signins} detail={`${activity.active_admin_sessions} active sessions`} />
        <Metric icon={<Brain size={18} />} label="Scans analyzed" value={activity.scans_analyzed} detail="All results since API start" />
        <Metric icon={<Layers3 size={18} />} label="Masks with regions" value={activity.masks_with_positive_regions} detail="A non-empty predicted mask" />
        <Metric icon={<Eye size={18} />} label="Grad-CAM maps" value={activity.gradcam_maps_produced} detail="Maps successfully generated" />
      </div>

      <div className="activity-result-breakdown">
        <div className="activity-section-heading"><div><span className="eyebrow">RESULT COUNTS</span><h2>Model suggestions</h2></div><small>Experimental image categories</small></div>
        <div className="activity-result-counts">
          {["glioma", "meningioma", "pituitary", "no_tumor", "unavailable"].map((label) => <div key={label}><span>{resultName(label)}</span><strong>{activity.result_counts[label] ?? 0}</strong></div>)}
        </div>
      </div>

      <div className="activity-recent-panel">
        <div className="activity-section-heading"><div><span className="eyebrow">RECENT EVENTS</span><h2>Recent sign-ins and scan analyses</h2></div><small>{latest.length} most recent</small></div>
        {latest.length === 0 ? <div className="activity-empty"><Activity size={17} />No activity yet. Patient sessions and completed scan analyses will appear here.</div> : <div className="activity-table-scroll"><table className="activity-table">
          <thead><tr><th>Time</th><th>Event</th><th>Role</th><th>Suggested result</th><th>Model score</th><th>Mask</th><th>Grad-CAM</th><th>Run details</th></tr></thead>
          <tbody>{latest.map((record, index) => <tr key={`${record.at}-${record.kind}-${index}`}>
            <td>{new Date(record.at).toLocaleString()}</td>
            <td>{eventName(record)}</td>
            <td>{record.role === "workbench" ? "Admin / workstation" : record.role}</td>
            <td>{record.kind === "scan_analysis" ? resultName(record.result_label ?? "unavailable") : "—"}</td>
            <td>{record.kind === "scan_analysis" && record.model_score_percent != null ? <><span title={record.score_semantics ?? "Uncalibrated model score"}>{record.model_score_percent}% <small>top score</small></span><br /><span className="activity-run-details">{scoreBreakdown(record.class_scores_percent)}{record.score_margin_percent != null ? ` · top-two gap ${record.score_margin_percent}%` : ""}</span></> : "—"}</td>
            <td>{record.kind !== "scan_analysis" ? "—" : !record.mask_generated ? "Unavailable" : !record.has_positive_mask ? "No region" : `${record.mask_coverage_percent?.toFixed(2)}% area`}</td>
            <td>{record.kind === "scan_analysis" ? record.gradcam_generated ? "Generated" : "Unavailable" : "—"}</td>
            <td>{record.kind === "scan_analysis" ? <span className="activity-run-details">{record.input_type} · {record.width}×{record.height}{record.slices ? ` · ${record.slices} slices` : ""}<br />Mask threshold {record.mask_threshold?.toFixed(2) ?? "—"} · coverage {record.mask_coverage_percent?.toFixed(2) ?? "—"}% · box {boxName(record.mask_bbox)}<br />Grad-CAM box {boxName(record.gradcam_bbox)} · overlap {record.gradcam_mask_overlap_iou == null ? "—" : `${(record.gradcam_mask_overlap_iou * 100).toFixed(1)}%`} · {record.warning_count ?? 0} warnings</span> : "Session event"}</td>
          </tr>)}</tbody>
        </table></div>}
      </div>
      <p className="activity-retention-note"><ShieldCheck size={15} />{activity.retention_note}</p>
      <p className="activity-started-at">Activity window started {new Date(activity.started_at).toLocaleString()}.</p>
    </>}
  </section>;
}
