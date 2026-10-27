import tempfile
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .config import APP_VERSION, CORS_ORIGINS, MAX_UPLOAD_BYTES, PROJECT_ROOT, SCRATCH_ROOT
from .activity import activity_ledger
from .auth import ADMIN_SESSION_SECONDS, active_admin_sessions_count, create_admin_session, require_admin, revoke_admin_session
from .imaging import inspect_scan
from .inference import analyze_image, registry
from .reporting import make_report
from .schemas import AdminLoginRequest, AdminSessionResponse, AnalysisReportRequest, AnalysisResult, EvaluationResponse, HealthResponse, ModelStatus, PatientActivitySessionEndRequest, PatientActivitySessionResponse, ScanInspection

@asynccontextmanager
async def lifespan(_app):
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(SCRATCH_ROOT)
    yield


app = FastAPI(title="BTDS Research Workstation API", version=APP_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ORIGINS),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return {"message": "NEURO/LAB local research workstation API", "version": APP_VERSION, "docs": "/docs"}

@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version=APP_VERSION, models=registry.statuses())


@app.get("/api/models", response_model=list[ModelStatus])
def models():
    return registry.statuses()


@app.post("/api/auth/admin/login", response_model=AdminSessionResponse)
def admin_login(payload: AdminLoginRequest):
    token = create_admin_session(payload.username, payload.password)
    activity_ledger.record_admin_signin()
    return AdminSessionResponse(access_token=token, expires_in=ADMIN_SESSION_SECONDS)


@app.post("/api/auth/admin/logout")
def admin_logout(authorization: str = Header(...), _admin: str = Depends(require_admin)):
    revoke_admin_session(authorization[7:].strip())
    return {"ok": True}


@app.post("/api/activity/patient-session/start", response_model=PatientActivitySessionResponse)
def patient_activity_session_start():
    return PatientActivitySessionResponse(session_id=activity_ledger.start_patient_session())


@app.post("/api/activity/patient-session/end")
def patient_activity_session_end(payload: PatientActivitySessionEndRequest):
    activity_ledger.end_patient_session(payload.session_id)
    return {"ok": True}


@app.get("/api/admin/activity")
def admin_activity(_admin: str = Depends(require_admin)):
    return activity_ledger.snapshot(active_admin_sessions_count())


@app.get("/api/evaluation", response_model=EvaluationResponse)
def evaluation(_admin: str = Depends(require_admin)):
    folder = PROJECT_ROOT / "assets" / "evaluations"
    recorded = {}
    for task in ("classifier", "segmenter"):
        path = folder / (task + ".json")
        if path.is_file():
            try:
                import json
                report = json.loads(path.read_text(encoding="utf-8"))
                if report.get("split") == "test" and isinstance(report.get("metrics"), dict):
                    recorded[task] = report
            except (OSError, ValueError):
                continue
    if recorded:
        return EvaluationResponse(available=True, metrics=recorded, detail="Recorded held-out test-set results from the local evaluation reports.")
    return EvaluationResponse(available=False, detail="No evaluation results are installed. Metrics appear only after a recorded test-set evaluation.")


@app.post("/api/evaluation/run", response_model=EvaluationResponse)
def run_evaluation(_admin: str = Depends(require_admin)):
    """Rerun test-only evaluation for each compatible, installed local model."""
    recorded = {}
    failures = []
    ready_tasks = []
    for task in ("classifier", "segmenter"):
        status = registry.status(task)
        if status.status != "ready":
            failures.append(f"{task}: {status.detail}")
            continue
        ready_tasks.append(task)
    if not ready_tasks:
        return EvaluationResponse(available=False, detail="No compatible local models are ready for evaluation. " + " ".join(failures))
    try:
        from backend.training.evaluate import evaluate_task
    except Exception:
        return EvaluationResponse(available=False, detail="Local evaluation dependencies are unavailable. Install the documented PyTorch and dataset dependencies.")
    for task in ready_tasks:
        try:
            recorded[task] = evaluate_task(task)
        except Exception as exc:
            failures.append(f"{task}: evaluation could not run ({type(exc).__name__}).")
    detail = "Evaluation ran only against the manifest's held-out test split."
    if failures:
        detail += " " + " ".join(failures)
    return EvaluationResponse(available=bool(recorded), metrics=recorded or None, detail=detail)


async def _read_upload(file: UploadFile) -> bytes:
    declared_size = getattr(file, "size", None)
    if declared_size is not None and declared_size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    return content


@app.post("/api/scan/inspect", response_model=ScanInspection)
async def scan_inspect(file: UploadFile = File(...)):
    try:
        scan, _ = inspect_scan(file.filename or "upload", await _read_upload(file))
        return scan
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...), patient_session: str | None = Header(default=None, alias="X-Patient-Session")):
    try:
        scan, image = inspect_scan(file.filename or "upload", await _read_upload(file))
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    inference = analyze_image(image) if image is not None else {
        "classification": None, "segmentation": None,
        "warnings": ["NIfTI volume browsing and inference are unavailable until compatible model metadata and volume viewer are installed."],
    }
    activity_ledger.record_analysis(scan.model_dump(), inference, patient_session)
    statuses = registry.statuses()
    return AnalysisResult(scan=scan, models=statuses, **inference)


@app.post("/api/reports/analysis")
def report(payload: AnalysisReportRequest):
    body = make_report(payload.analysis.model_dump(), payload.patient_details.model_dump())
    return Response(
        content=body, media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="neurolab-patient-mri-summary.pdf"'},
    )
