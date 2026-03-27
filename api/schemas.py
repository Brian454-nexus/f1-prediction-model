"""
F1 APEX — API Schemas (Pydantic v2)
======================================
Defines the full request/response schema for the F1 APEX REST API.
Matches the PRD §7.1 FR-5 specification exactly.

PRD Output Schema:
{
    race, predictions: [
        { driver, team, predicted_position, win_probability, top3_probability,
          top10_probability, dnf_probability, confidence_interval }
    ], model_version, generated_at
}
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Driver Prediction ─────────────────────────────────────────────────────────

class ConfidenceInterval(BaseModel):
    p10: float = Field(..., description="10th percentile finishing position")
    p90: float = Field(..., description="90th percentile finishing position")


class ExplanationFactor(BaseModel):
    factor: str
    description: str


class DriverPredictionSchema(BaseModel):
    driver:             str
    team:               str
    predicted_position: float = Field(..., ge=1, le=20)
    win_probability:    float = Field(..., ge=0, le=1)
    top3_probability:   float = Field(..., ge=0, le=1)
    top10_probability:  float = Field(..., ge=0, le=1)
    dnf_probability:    float = Field(..., ge=0, le=1)
    confidence_interval: ConfidenceInterval
    dark_horse:         bool  = False
    wet_specialist:     bool  = False
    explanation:        Optional[list[ExplanationFactor]] = None


# ── Race Prediction Response ──────────────────────────────────────────────────

class RacePredictionResponse(BaseModel):
    race:           str
    round_id:       int
    season:         int = 2026
    circuit_name:   str
    model_version:  str = "1.0"
    model_branch:   str = Field(..., description="DRY_STANDARD or WET_WEATHER")
    generated_at:   datetime
    weather: dict = Field(default_factory=dict)
    predictions:    list[DriverPredictionSchema]

    @field_validator("predictions")
    @classmethod
    def sort_by_predicted_position(cls, v):
        return sorted(v, key=lambda x: x.predicted_position)


# ── Post-race Calibration Input ───────────────────────────────────────────────

class ActualDriverResult(BaseModel):
    driver:     str
    team:       str
    finish_pos: int = Field(..., ge=1, le=20)
    dnf:        bool = False
    cause:      str  = ""


class PostRaceCalibrationRequest(BaseModel):
    round_id:    int
    race_name:   str
    results:     list[ActualDriverResult]


class CalibrationResponse(BaseModel):
    round_id:        int
    spearman_rho:    float
    target_met:      bool
    avg_rho_to_date: float
    races_logged:    int
    updated_weights: dict


# ── Health Check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str = "ok"
    model_version: str = "1.0"
    races_logged:  int = 0


# ── Prediction Card Export ────────────────────────────────────────────────────

class PredictionCardRequest(BaseModel):
    race:   str
    top_n:  int = Field(default=5, ge=1, le=10)
    theme:  str = Field(default="dark", pattern="^(dark|light)$")
