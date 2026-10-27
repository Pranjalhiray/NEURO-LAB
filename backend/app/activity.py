"""In-memory, de-identified activity counters for the local admin dashboard."""
from collections import Counter, deque
from datetime import datetime, timezone
import secrets
import threading
import time

PATIENT_ACTIVITY_SESSION_SECONDS = 8 * 60 * 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActivityLedger:
    """Keep short-lived operational summaries; never retain uploaded images or patient details."""

    def __init__(self, max_recent: int = 100):
        self._lock = threading.RLock()
        self.started_at = _now()
        self._patient_sessions: dict[str, float] = {}
        self._patient_sessions_started = 0
        self._admin_signins = 0
        self._scans_analyzed = 0
        self._masks_produced = 0
        self._gradcam_maps_produced = 0
        self._result_counts: Counter[str] = Counter()
        self._recent: deque[dict] = deque(maxlen=max_recent)

    def _append(self, event: dict) -> None:
        self._recent.appendleft(event)

    def start_patient_session(self) -> str:
        session_id = secrets.token_urlsafe(24)
        with self._lock:
            self._patient_sessions[session_id] = datetime.now(timezone.utc).timestamp()
            self._patient_sessions_started += 1
            self._append({"at": _now(), "kind": "patient_sign_in", "role": "patient"})
        return session_id

    def end_patient_session(self, session_id: str | None) -> None:
        if not session_id or len(session_id) > 128:
            return
        with self._lock:
            if self._patient_sessions.pop(session_id, None) is not None:
                self._append({"at": _now(), "kind": "patient_sign_out", "role": "patient"})

    def is_patient_session(self, session_id: str | None) -> bool:
        if not session_id or len(session_id) > 128:
            return False
        with self._lock:
            started = self._patient_sessions.get(session_id)
            if started is None:
                return False
            if time.time() - started >= PATIENT_ACTIVITY_SESSION_SECONDS:
                self._patient_sessions.pop(session_id, None)
                return False
            return True

    def record_admin_signin(self) -> None:
        with self._lock:
            self._admin_signins += 1
            self._append({"at": _now(), "kind": "admin_sign_in", "role": "admin"})

    def record_analysis(self, scan: dict, result: dict, patient_session_id: str | None = None) -> None:
        classification = result.get("classification") or {}
        segmentation = result.get("segmentation") or {}
        explanation = result.get("explanation") or {}
        label = str(classification.get("label") or "unavailable")
        scores = classification.get("scores") or {}
        raw_score = scores.get(label)
        score = round(float(raw_score) * 100, 1) if isinstance(raw_score, (int, float)) else None
        class_scores = {
            str(name): round(float(value) * 100, 1)
            for name, value in scores.items()
            if isinstance(value, (int, float))
        }
        raw_margin = classification.get("score_margin")
        mask_generated = bool(segmentation)
        has_positive_mask = bool(segmentation.get("bounding_box"))
        gradcam_generated = bool(explanation.get("heatmap_png_base64"))
        event = {
            "at": _now(),
            "kind": "scan_analysis",
            "role": "patient" if self.is_patient_session(patient_session_id) else "workbench",
            "result_label": label,
            "model_score_percent": score,
            "class_scores_percent": class_scores,
            "score_margin_percent": round(float(raw_margin) * 100, 1) if isinstance(raw_margin, (int, float)) else None,
            "score_semantics": classification.get("score_semantics") if classification else None,
            "input_type": scan.get("input_type"),
            "width": scan.get("width"),
            "height": scan.get("height"),
            "slices": scan.get("slices"),
            "classification_model": classification.get("model_name"),
            "segmentation_model": segmentation.get("model_name"),
            "mask_generated": mask_generated,
            "has_positive_mask": has_positive_mask,
            "mask_coverage_percent": round(float(segmentation.get("coverage_percent", 0)), 3) if segmentation else None,
            "mask_bbox": segmentation.get("bounding_box"),
            "mask_threshold": segmentation.get("threshold"),
            "gradcam_generated": gradcam_generated,
            "gradcam_bbox": explanation.get("region_bbox"),
            "gradcam_mask_overlap_iou": explanation.get("segmentation_overlap_iou"),
            "warning_count": len(result.get("warnings") or []),
        }
        with self._lock:
            self._scans_analyzed += 1
            self._result_counts[label] += 1
            self._masks_produced += int(mask_generated and has_positive_mask)
            self._gradcam_maps_produced += int(gradcam_generated)
            self._append(event)

    def snapshot(self, active_admin_sessions: int = 0) -> dict:
        with self._lock:
            now = time.time()
            self._patient_sessions = {
                key: started for key, started in self._patient_sessions.items()
                if now - started < PATIENT_ACTIVITY_SESSION_SECONDS
            }
            return {
                "started_at": self.started_at,
                "patient_sessions_started": self._patient_sessions_started,
                "active_patient_sessions": len(self._patient_sessions),
                "admin_signins": self._admin_signins,
                "active_admin_sessions": active_admin_sessions,
                "scans_analyzed": self._scans_analyzed,
                "masks_with_positive_regions": self._masks_produced,
                "gradcam_maps_produced": self._gradcam_maps_produced,
                "result_counts": dict(self._result_counts),
                "recent": list(self._recent),
                "retention_note": "Patient sign-ins are unverified local sessions, not confirmed individual identities. Anonymous counters and recent model outputs stay in API memory and reset when the API restarts. Patient names, contact details, filenames, and scan pixels are not stored.",
            }


activity_ledger = ActivityLedger()
