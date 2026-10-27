import { useMemo } from "react";
import { Activity, AlertTriangle, ArrowDownToLine, Brain, Layers3, LoaderCircle, Play } from "lucide-react";
import { downloadEvaluation, type EvaluationSummary } from "./api";

type Props = { evaluation: EvaluationSummary | null; loading: boolean; error: string; onRerun: () => void };
const pct = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
const number = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "—";
const label = (value: string) => value.replace(/_/g, " ");

export function EvaluationPage({ evaluation, loading, error, onRerun }: Props) {
  const classifier = evaluation?.metrics?.classifier;
  const segmenter = evaluation?.metrics?.segmenter;
  const classMetrics: Record<string, any> = classifier?.metrics?.per_class ?? {};
  const classOrder: string[] = classifier?.metrics?.class_order ?? Object.keys(classMetrics);
  const confusion: number[][] = classifier?.metrics?.confusion_matrix ?? [];
  const classRows = useMemo(() => classOrder.map((name) => ({
    name,
    ...(classMetrics[name] ?? { f1: classifier?.metrics?.class_f1?.[name] }),
  })), [classOrder.join("|"), JSON.stringify(classMetrics), JSON.stringify(classifier?.metrics?.class_f1)]);
  const timestamps = [classifier?.created_utc, segmenter?.created_utc].filter(Boolean).sort();
  const created = timestamps[timestamps.length - 1];

  return <div className="evaluation-page">
    <div className="page-heading evaluation-heading"><div><div className="eyebrow">RESEARCH TOOLS <i /> HELD-OUT METRICS</div><h1>Model evaluation</h1><p>View recorded performance or rerun installed models against the configured test split.</p></div><div className="evaluation-actions">{evaluation?.available && <button className="secondary-button" onClick={() => downloadEvaluation(evaluation)}><ArrowDownToLine size={15} /> Export JSON</button>}<button className="primary-button rerun-button" disabled={loading} onClick={onRerun}>{loading ? <LoaderCircle className="spin" size={16} /> : <Play size={15} />}{loading ? "Evaluating test split…" : "Re-run test evaluation"}</button></div></div>
    <div className="notice"><strong>Test split only</strong><span>Evaluation reads the configured held-out test records. It does not use uploaded scans or validation/training records.</span></div>
    {error && <div className="evaluation-error"><AlertTriangle size={16} />{error}</div>}
    {loading && <div className="evaluation-progress"><LoaderCircle className="spin" size={18} /><div><b>Calculating model metrics</b><span>This may take a few minutes on CPU. The page will update when both tasks finish.</span></div></div>}
    {!evaluation?.available && !loading ? <section className="panel no-evaluation"><Activity size={24} /><h2>No evaluation report available</h2><p>{evaluation?.detail ?? "Connect the API to load the saved test results."}</p><button className="primary-button rerun-button" onClick={onRerun}><Play size={15} /> Run evaluation</button></section> : <>
      {evaluation?.detail && <div className="evaluation-detail">{evaluation.detail}</div>}
      <div className="metric-grid">
        <section className="panel metric-card"><div className="metric-card-heading"><span className="metric-icon"><Brain size={16} /></span><div><b>Classification</b><span>{classifier?.model_name ?? "Classifier"}</span></div></div><div className="large-metric">{pct(classifier?.metrics?.accuracy)}<small>accuracy</small></div><div className="metric-pair"><span>Macro precision <b>{pct(classifier?.metrics?.macro_precision)}</b></span><span>Macro recall <b>{pct(classifier?.metrics?.macro_recall)}</b></span><span>Macro F1 <b>{pct(classifier?.metrics?.macro_f1)}</b></span><span>Macro ROC-AUC <b>{pct(classifier?.metrics?.roc_auc_ovr_macro)}</b></span></div><div className="metric-foot">{number(classifier?.metrics?.sample_count ?? classifier?.dataset?.test_rows)} test images · {classifier?.architecture ?? "four-class image classifier"}. Macro metrics are unweighted; ROC-AUC uses one-vs-rest scoring.</div></section>
        <section className="panel metric-card"><div className="metric-card-heading"><span className="metric-icon violet"><Layers3 size={16} /></span><div><b>Segmentation</b><span>{segmenter?.model_name ?? "Segmenter"}</span></div></div><div className="large-metric">{pct(segmenter?.metrics?.dice)}<small>mean Dice</small></div><div className="metric-pair"><span>Mean IoU <b>{pct(segmenter?.metrics?.iou)}</b></span><span>Test images <b>{number(segmenter?.metrics?.sample_count ?? segmenter?.dataset?.test_rows)}</b></span><span>Empty-mask convention <b>{segmenter?.metrics?.empty_mask_convention ? "Both empty = 1.0" : segmenter?.metrics?.empty_mask_cases_count_as_perfect_if_prediction_is_empty ? "Perfect if both empty" : "Not specified"}</b></span></div><div className="metric-foot">Overlap is measured against the dataset masks; empty-mask convention is shown explicitly.</div></section>
      </div>
      <div className="evaluation-grid">
        <section className="panel evaluation-card"><div className="evaluation-card-heading"><div><h2>Classification by class</h2><p>Precision, recall, and F1 against the held-out labels</p></div><span>{classRows.length} classes</span></div><div className="table-scroll"><table className="data-table"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Samples</th></tr></thead><tbody>{classRows.map((row) => <tr key={row.name}><td>{label(row.name)}</td><td>{pct(row.precision)}</td><td>{pct(row.recall)}</td><td>{pct(row.f1)}</td><td>{number(row.support)}</td></tr>)}</tbody></table></div></section>
        <section className="panel evaluation-card"><div className="evaluation-card-heading"><div><h2>Confusion matrix</h2><p>Rows are true class; columns are predicted class</p></div><Brain size={17} /></div>{confusion.length ? <div className="table-scroll"><table className="matrix-table"><thead><tr><th>True ↓ / Predicted →</th>{classOrder.map((name) => <th key={name}>{label(name)}</th>)}</tr></thead><tbody>{confusion.map((row, i) => <tr key={classOrder[i] ?? i}><th>{label(classOrder[i] ?? `class ${i + 1}`)}</th>{row.map((value, j) => <td className={i === j ? "matrix-diagonal" : ""} key={j}>{number(value)}</td>)}</tr>)}</tbody></table></div> : <div className="not-recorded">Confusion matrix is not included in the saved classifier report.</div>}</section>
      </div>
      <section className="panel evaluation-card"><div className="evaluation-card-heading"><div><h2>Segmentation by class</h2><p>Mean region overlap for each dataset label</p></div><span>{Object.keys(segmenter?.metrics?.per_class ?? {}).length || "—"} classes</span></div>{segmenter?.metrics?.per_class ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Class</th><th>Mean Dice</th><th>Mean IoU</th><th>Samples</th></tr></thead><tbody>{Object.entries(segmenter.metrics.per_class as Record<string, any>).map(([name, row]) => <tr key={name}><td>{label(name)}</td><td>{pct(row.dice)}</td><td>{pct(row.iou)}</td><td>{number(row.support)}</td></tr>)}</tbody></table></div> : <div className="not-recorded">The saved report has overall Dice and IoU only. Re-run evaluation to calculate per-class overlap.</div>}</section>
      <section className="evaluation-meta"><span>Dataset <b>{classifier?.dataset?.id ?? "BRISC test split"}</b></span><span>Split <b>test</b></span><span>Last report <b>{created ? new Date(created).toLocaleString() : "Saved report timestamp unavailable"}</b></span></section>
      <div className="evaluation-limits">These metrics describe this dataset split and do not establish clinical performance. The dataset does not provide patient identifiers, so patient-level separation cannot be verified.</div>
    </>}
  </div>;
}
