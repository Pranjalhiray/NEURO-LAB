import { useEffect, useRef, useState } from "react";
import { Brain, LogOut } from "lucide-react";
import { analyzeFile, endPatientActivitySession, getAdminActivity, getEvaluation, getHealth, IS_HOSTED_DEMO, loginAdmin, logoutAdmin, runEvaluation, startPatientActivitySession, type AdminActivity, type Analysis, type EvaluationSummary, type PatientDetails } from "./api";
import { ScanViewer } from "./ScanViewer";
import { AnalysisPanel } from "./AnalysisPanel";
import { EvaluationPage } from "./EvaluationPage";
import { ActivityDashboard } from "./ActivityDashboard";
import { PatientAccess } from "./PatientAccess";
import { readNifti, type Volume } from "./nifti";

type Page = "workbench" | "evaluation" | "activity";
type Role = "patient" | "admin";
const pageFromPath = (): Page => {
  const path = window.location.pathname.replace(/\/$/, "");
  return path === "/evaluation" ? "evaluation" : path === "/activity" ? "activity" : "workbench";
};

function AdminNavigation({ page, navigate }: { page: Page; navigate: (next: Page) => void }) {
  return <nav className="admin-nav" aria-label="Administrator workspace">
    <button className={page === "workbench" ? "active" : ""} onClick={() => navigate("workbench")}>MRI workbench</button>
    <button className={page === "activity" ? "active" : ""} onClick={() => navigate("activity")}>Activity &amp; results</button>
    <button className={page === "evaluation" ? "active" : ""} onClick={() => navigate("evaluation")}>Model evaluation</button>
  </nav>;
}

export default function App() {
  const [page, setPage] = useState<Page>(pageFromPath);
  const [role, setRole] = useState<Role | null>(null);
  const [adminToken, setAdminToken] = useState<string | null>(null);
  const [patient, setPatient] = useState<PatientDetails | null>(null);
  const [patientSessionId, setPatientSessionId] = useState<string | null>(null);
  const [apiState, setApiState] = useState<"checking" | "online" | "offline">("checking");
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);
  const [evaluationError, setEvaluationError] = useState("");
  const [evaluationBusy, setEvaluationBusy] = useState(false);
  const [activity, setActivity] = useState<AdminActivity | null>(null);
  const [activityError, setActivityError] = useState("");
  const [activityBusy, setActivityBusy] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [volume, setVolume] = useState<Volume | null>(null);
  const [result, setResult] = useState<Analysis | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [accepted, setAccepted] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const previewRequest = useRef(0);
  const isVolumeFile = Boolean(file && /\.nii(?:\.gz)?$/i.test(file.name));

  useEffect(() => {
    const onPopState = () => setPage(pageFromPath());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    getHealth().then(() => setApiState("online")).catch(() => setApiState("offline"));
  }, []);

  useEffect(() => {
    if (role !== "admin" || !adminToken) { setEvaluation(null); return; }
    getEvaluation(adminToken).then(setEvaluation).catch((e) => {
      setEvaluation(null);
      setEvaluationError(e instanceof Error ? e.message : "Evaluation status is unavailable.");
    });
  }, [role, adminToken]);

  useEffect(() => {
    if (role !== "admin" || !adminToken || page !== "activity") return;
    let active = true;
    const refresh = async () => {
      try {
        const next = await getAdminActivity(adminToken);
        if (active) { setActivity(next); setActivityError(""); }
      } catch (e) {
        if (active) setActivityError(e instanceof Error ? e.message : "Admin activity is unavailable.");
      }
    };
    setActivityBusy(true);
    void refresh().finally(() => { if (active) setActivityBusy(false); });
    const timer = window.setInterval(() => { void refresh(); }, 15000);
    return () => { active = false; window.clearInterval(timer); };
  }, [role, adminToken, page]);

  useEffect(() => {
    if (page !== "workbench" && role === "patient") {
      window.history.replaceState({}, "", "/");
      setPage("workbench");
    }
  }, [page, role]);

  const navigate = (next: Page) => {
    const path = next === "evaluation" ? "/evaluation" : next === "activity" ? "/activity" : "/";
    if (window.location.pathname !== path) window.history.pushState({}, "", path);
    setPage(next);
  };

  const clearScan = () => {
    setFile(null);
    setVolume(null);
    setResult(null);
    setError("");
    setAccepted(false);
    previewRequest.current += 1;
    if (fileInput.current) fileInput.current.value = "";
  };

  const endSession = () => {
    if (role === "patient" && patientSessionId) void endPatientActivitySession(patientSessionId).catch(() => undefined);
    if (role === "admin" && adminToken) void logoutAdmin(adminToken).catch(() => undefined);
    clearScan();
    setPatient(null);
    setPatientSessionId(null);
    setRole(null);
    setAdminToken(null);
    navigate("workbench");
  };

  const signInPatient = async (details: PatientDetails) => {
    let sessionId: string | null = null;
    try { sessionId = (await startPatientActivitySession()).session_id; } catch { /* local scan access remains available if activity tracking is offline */ }
    setPatient(details);
    setPatientSessionId(sessionId);
    setRole("patient");
    setAdminToken(null);
    navigate("workbench");
  };

  const signInAdmin = async (username: string, password: string) => {
    const session = await loginAdmin(username, password);
    setRole("admin");
    setPatient(null);
    setAdminToken(session.access_token);
    navigate("workbench");
  };

  const pickFile = (next?: File) => {
    const request = ++previewRequest.current;
    setError("");
    setResult(null);
    setFile(null);
    setVolume(null);
    setAccepted(false);
    if (!next) return;
    const name = next.name.toLowerCase();
    const isNifti = /\.nii(?:\.gz)?$/.test(name);
    const isImage = /\.(png|jpe?g|bmp|tiff?)$/.test(name);
    if (!isImage && !(role === "admin" && isNifti)) {
      setError(role === "admin" ? "Choose a JPG, PNG, TIFF, BMP, or NIfTI MRI scan." : "Choose a 2D MRI image in JPG, PNG, TIFF, or BMP format.");
      if (fileInput.current) fileInput.current.value = "";
      return;
    }
    if (next.size > 64 * 1024 * 1024) {
      setError("This image is larger than the 64 MB limit.");
      if (fileInput.current) fileInput.current.value = "";
      return;
    }
    setFile(next);
    if (isNifti) {
      readNifti(next).then((nextVolume) => {
        if (previewRequest.current === request) setVolume(nextVolume);
      }).catch((e) => {
        if (previewRequest.current === request) setError(e instanceof Error ? e.message : "This volume could not be previewed.");
      });
    }
  };

  const run = async () => {
    if (!file || isVolumeFile || !accepted) return;
    setBusy(true);
    setError("");
    try { setResult(await analyzeFile(file, role === "patient" ? patientSessionId : null)); }
    catch (e) { setError(e instanceof Error ? e.message : "The scan could not be reviewed."); }
    finally { setBusy(false); }
  };

  const rerunEvaluation = async () => {
    if (!adminToken) { setEvaluationError("Sign in with an administrator account to evaluate models."); return; }
    setEvaluationBusy(true);
    setEvaluationError("");
    try { setEvaluation(await runEvaluation(adminToken)); }
    catch (e) { setEvaluationError(e instanceof Error ? e.message : "Evaluation failed."); }
    finally { setEvaluationBusy(false); }
  };

  if (!role) return <PatientAccess onPatientLogin={signInPatient} onAdminLogin={signInAdmin} />;

  if (page === "evaluation" && role === "admin") return <div className="app-shell evaluation-shell">
    <main className="main">
      <header className="topbar evaluation-topbar"><div className="evaluation-brand"><span className="brand-mark"><Brain size={19} /></span><strong>NEURO<span>/</span>LAB</strong><small>RESEARCHER / JUDGE VIEW</small></div><AdminNavigation page={page} navigate={navigate} /><button className="secondary-button" onClick={endSession}><LogOut size={14} /> Sign out</button></header>
      <section className="page-wrap"><EvaluationPage evaluation={evaluation} loading={evaluationBusy} error={evaluationError} onRerun={rerunEvaluation} /></section>
    </main>
  </div>;

  if (page === "activity" && role === "admin") return <div className="patient-app">
    <header className="patient-topbar">
      <div className="patient-brand"><span className="brand-mark"><Brain size={20} /></span><div><b>NEURO<span>/</span>LAB</b><small>ADMINISTRATOR SESSION</small></div></div>
      <div className="patient-topbar-right"><AdminNavigation page={page} navigate={navigate} /><span className={"patient-connection " + apiState}><i />{apiState === "online" ? "Ready" : apiState === "offline" ? "Service unavailable" : "Connecting"}</span><button className="secondary-button" onClick={endSession}><LogOut size={15} /> Sign out</button></div>
    </header>
    <main className="admin-activity-main"><ActivityDashboard activity={activity} loading={activityBusy} error={activityError} onRefresh={() => {
      if (!adminToken) return;
      setActivityBusy(true);
      getAdminActivity(adminToken).then((next) => { setActivity(next); setActivityError(""); }).catch((e) => setActivityError(e instanceof Error ? e.message : "Admin activity is unavailable.")).finally(() => setActivityBusy(false));
    }} /></main>
  </div>;

  return <div className="patient-app">
    <header className="patient-topbar">
      <div className="patient-brand"><span className="brand-mark"><Brain size={20} /></span><div><b>NEURO<span>/</span>LAB</b><small>{role === "admin" ? "ADMINISTRATOR SESSION" : "PATIENT MRI SESSION"}</small></div></div>
      <div className="patient-topbar-right">{role === "admin" && <AdminNavigation page={page} navigate={navigate} />}<span className={"patient-connection " + apiState}><i />{apiState === "online" ? "Ready" : apiState === "offline" ? "Service unavailable" : "Connecting"}</span><button className="secondary-button" onClick={endSession}><LogOut size={15} /> Sign out</button></div>
    </header>
    <main className="patient-page-wrap">
      <div className="page-heading patient-page-heading"><div><div className="eyebrow">{role === "admin" ? "RESEARCH WORKSPACE" : "PATIENT MRI REVIEW"}</div><h1>{role === "admin" ? "MRI workbench" : `Hello, ${patient?.name ?? "Patient"}`}</h1><p>Choose an MRI image to view an experimental AI-assisted summary.</p></div>{patient && <div className="patient-age">Age <strong>{patient.age}</strong></div>}</div>
      <div className="patient-safety-banner"><strong>{role === "admin" ? "Research workspace" : "AI-assisted research result"}</strong><span>{role === "admin" ? "Administrator view includes research display controls and access to model evaluation." : "This experimental model can be wrong, especially on scans from a different source or scanner. It cannot diagnose or guide treatment; ask a qualified clinician to review every MRI."}</span></div>
      <div className="workbench patient-workbench">
        <ScanViewer file={file} volume={role === "admin" ? volume : null} result={result} busy={busy} fileInput={fileInput} onPick={pickFile} onClear={clearScan} patientMode={role === "patient"} />
        <AnalysisPanel result={result} error={error} busy={busy} hasFile={Boolean(file && !isVolumeFile)} accepted={accepted} patient={patient} onAccept={setAccepted} onRun={run} />
      </div>
      <footer className="patient-page-footer"><span>NEURO/LAB · RESEARCH PROTOTYPE</span><span>{IS_HOSTED_DEMO ? "Hosted demo · use only de-identified sample data." : patient ? "Patient details are cleared when this session ends." : "No patient record is created in this administrator session."}</span></footer>
    </main>
  </div>;
}
