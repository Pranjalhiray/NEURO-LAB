from datetime import date
import re
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelStatus(BaseModel):
    task: str
    status: str
    model_name: str | None = None
    detail: str
    weights_size_bytes: int | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models: list[ModelStatus]


class ScanInspection(BaseModel):
    filename: str
    input_type: str
    width: int | None = None
    height: int | None = None
    slices: int | None = None
    modality: str | None = None
    note: str


class AnalysisResult(BaseModel):
    scan: ScanInspection
    classification: dict[str, Any] | None = None
    segmentation: dict[str, Any] | None = None
    explanation: dict[str, Any] | None = None
    source_png_base64: str | None = None
    models: list[ModelStatus]
    warnings: list[str]
    research_use_only: bool = True


class PatientDetails(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    age: int = Field(ge=0, le=130)
    date_of_birth: date
    email: str = Field(min_length=5, max_length=254)
    phone: str = Field(min_length=7, max_length=25)

    @field_validator("email")
    @classmethod
    def valid_email(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("Enter a valid email address.")
        return value

    @field_validator("phone")
    @classmethod
    def valid_phone(cls, value: str) -> str:
        value = value.strip()
        digits = re.sub(r"\D", "", value)
        if len(digits) < 7 or not re.fullmatch(r"[+\d][\d\s().-]{6,24}", value):
            raise ValueError("Enter a valid contact number.")
        return value

    @model_validator(mode="after")
    def age_matches_date_of_birth(self):
        today = date.today()
        if self.date_of_birth > today:
            raise ValueError("Date of birth cannot be in the future.")
        calculated_age = today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )
        if self.age != calculated_age:
            raise ValueError("Age must match the date of birth.")
        return self


class AnalysisReportRequest(BaseModel):
    analysis: AnalysisResult
    patient_details: PatientDetails


class AdminLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


class AdminSessionResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class PatientActivitySessionResponse(BaseModel):
    session_id: str


class PatientActivitySessionEndRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class EvaluationResponse(BaseModel):
    available: bool
    metrics: dict[str, Any] | None = None
    detail: str


