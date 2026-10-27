import importlib.util
import io
import asyncio
import base64
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient
from PIL import Image
from fastapi import HTTPException

from backend.app import inference
from backend.app import main as api
from backend.app.main import evaluation, health
from backend.app.auth import create_admin_session, require_admin
from backend.app.schemas import PatientDetails


async def post_upload(filename: str, content: bytes):
    async with api.app.router.lifespan_context(api.app):
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url="http://test") as client:
            return await client.post("/api/analyze", files={"file": (filename, content, "image/png")})


class ApiReadinessTests(unittest.TestCase):
    def test_health_exposes_both_independent_model_slots(self):
        result = health()
        self.assertEqual(result.status, "ok")
        self.assertEqual({model.task for model in result.models}, {"classifier", "segmenter"})
        self.assertTrue(all(model.status in {"missing", "invalid", "ready"} for model in result.models))

    def test_evaluation_does_not_claim_unrecorded_metrics(self):
        result = evaluation(_admin="admin")
        self.assertIsInstance(result.available, bool)
        if not result.available:
            self.assertIsNone(result.metrics)

    def test_admin_session_requires_configured_credentials(self):
        with patch.dict(os.environ, {"BTDS_ADMIN_USERNAME": "researcher", "BTDS_ADMIN_PASSWORD": "test-long-secret"}):
            token = create_admin_session("researcher", "test-long-secret")
            self.assertEqual(require_admin("Bearer " + token), "researcher")
            with self.assertRaises(HTTPException):
                create_admin_session("researcher", "wrong-secret")

    def test_admin_guard_rejects_missing_token(self):
        with self.assertRaises(HTTPException):
            require_admin(None)

    def test_patient_age_must_match_date_of_birth(self):
        dob = date(2000, 1, 1)
        age = date.today().year - dob.year - ((date.today().month, date.today().day) < (dob.month, dob.day))
        patient = PatientDetails(name="Sample Patient", age=age, date_of_birth=dob, email="patient@example.com", phone="+1 555 123 4567")
        self.assertEqual(patient.age, age)
        with self.assertRaises(ValueError):
            PatientDetails(name="Sample Patient", age=age + 1, date_of_birth=dob, email="patient@example.com", phone="+1 555 123 4567")

    def test_upload_returns_no_predictions_without_compatible_weights(self):
        output = io.BytesIO()
        Image.new("RGB", (16, 16), (30, 40, 50)).save(output, format="PNG")
        with tempfile.TemporaryDirectory() as tmp:
            old_root = inference.registry.root
            inference.registry.root = Path(tmp)
            try:
                response = asyncio.run(post_upload("scan.png", output.getvalue()))
            finally:
                inference.registry.root = old_root
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIsNone(result["classification"])
        self.assertIsNone(result["segmentation"])
        self.assertTrue(result["research_use_only"])

    def test_upload_size_limit_is_applied_before_inference(self):
        output = io.BytesIO()
        Image.new("RGB", (16, 16), (30, 40, 50)).save(output, format="PNG")
        with patch.object(api, "MAX_UPLOAD_BYTES", 4):
            response = asyncio.run(post_upload("scan.png", output.getvalue()))
        self.assertEqual(response.status_code, 413)

    @unittest.skipUnless(importlib.util.find_spec("reportlab"), "reportlab is an optional PDF dependency")
    def test_report_is_a_pdf(self):
        from backend.app.reporting import make_report
        dob = date(2000, 1, 1)
        age = date.today().year - dob.year - ((date.today().month, date.today().day) < (dob.month, dob.day))
        image = io.BytesIO()
        Image.new("RGB", (32, 32), (40, 80, 120)).save(image, format="PNG")
        patient = PatientDetails(name="Sample Patient", age=age, date_of_birth=dob, email="patient@example.com", phone="+1 555 123 4567")
        analysis = {
            "scan": {"filename": "scan.png", "input_type": "2D image"},
            "classification": {"label": "glioma"}, "segmentation": None,
            "source_png_base64": base64.b64encode(image.getvalue()).decode("ascii"),
        }
        data = make_report(analysis, patient.model_dump())
        self.assertTrue(data.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
