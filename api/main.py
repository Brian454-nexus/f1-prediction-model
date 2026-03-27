"""
F1 APEX — FastAPI Application
================================
REST API exposing prediction outputs for downstream consumers:
web dashboard, mobile app, and social media automation.

ENDPOINTS:
    GET  /health                      ─ Health check + model version
    POST /predict/{round_id}          ─ Generate race prediction for a round
    GET  /prediction/{round_id}       ─ Retrieve last stored prediction
    POST /calibrate                   ─ Post-race calibration with actual results
    POST /export/prediction-card      ─ PNG prediction card for social media

DESIGN DECISIONS:
    - Redis cache on predictions (60-minute TTL) for fast repeated reads.
      Avoids re-running 50k MC simulation on each API call.
    - The /predict endpoint triggers the full ML pipeline synchronously
      for simplicity in v1.0. In v2.0 this becomes an async Prefect job.
    - Prediction PNG cards use Pillow (no external image API dependency).
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse

from api.schemas import (
    RacePredictionResponse,
    DriverPredictionSchema,
    ConfidenceInterval,
    ExplanationFactor,
    PostRaceCalibrationRequest,
    CalibrationResponse,
    HealthResponse,
    PredictionCardRequest,
)
from models.meta_ensemble import MetaEnsemble
from models.calibration import AutoCalibrator
from models.explainer import ModelExplainer
from utils.logger import get_logger

logger = get_logger("APEX.API")

# ── App init ──────────────────────────────────────────────────────────────────

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="F1 APEX — Prediction Engine API",
    description=(
        "Advanced ML prediction engine for 2026 Formula 1 race outcomes. "
        "Powered by Xtension Labs × Antigravity."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory prediction cache (keyed by round_id) ───────────────────────────
# Replace with Redis in production
_prediction_cache: dict[int, dict] = {}

# ── Prediction storage dir ────────────────────────────────────────────────────
PREDICTIONS_DIR = Path("data/predictions")
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return empty response to quiet browser favicon requests."""
    return JSONResponse(content={})


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns API health status and current model version."""
    accuracy_log_path = Path("models/state/accuracy_log.json")
    races_logged = 0
    if accuracy_log_path.exists():
        with open(accuracy_log_path) as f:
            races_logged = len(json.load(f))
    return HealthResponse(races_logged=races_logged)


@app.post(
    "/predict/{round_id}",
    response_model=RacePredictionResponse,
    tags=["Predictions"],
    summary="Generate race prediction for a given round",
)
async def generate_prediction(
    round_id: int,
    circuit_name: str = Query(..., description="e.g. 'Suzuka'"),
    total_laps: int = Query(53),
    race_utc: str = Query(..., description="Race UTC start ISO-8601 e.g. '2026-03-29T14:00:00Z'"),
):
    """
    Trigger the full prediction pipeline for a race weekend.
    Runs all 7 sub-models + MetaEnsemble + Explainer.
    Result is cached for 60 minutes.
    """
    logger.info(f"POST /predict/{round_id} — {circuit_name}")

    if round_id in _prediction_cache:
        logger.info(f"Cache hit for round {round_id}")
        return JSONResponse(_prediction_cache[round_id])

    try:
        from orchestration.race_weekend_dag import (
            task_run_ess, task_fetch_fia_docs,
            task_build_driver_inputs, task_dry_prediction,
            task_wet_prediction, task_generate_explanations,
            task_check_weather,
        )
        from data.data_pipeline import CIRCUIT_MAP
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Pipeline import error: {e}")

    # Weather routing
    coords = CIRCUIT_MAP.get(round_id, {}).get("coords", (0.0, 0.0))
    weather = task_check_weather.fn(coords[0], coords[1], race_utc)

    fia_docs    = task_fetch_fia_docs.fn(round_id)
    ess_results = task_run_ess.fn(circuit_name, total_laps)
    driver_inputs = task_build_driver_inputs.fn(circuit_name, fia_docs, ess_results)

    if weather["route_to_wet_model"]:
        raw_preds = task_wet_prediction.fn(driver_inputs, weather["rain_probability"])
        branch    = "WET_WEATHER"
    else:
        raw_preds = task_dry_prediction.fn(driver_inputs, circuit_name, round_id)
        branch    = "DRY_STANDARD"

    explanations = task_generate_explanations.fn(driver_inputs, raw_preds, circuit_name)
    exp_map  = {e["driver"]: e for e in explanations}

    predictions = []
    for p in raw_preds:
        exp = exp_map.get(p["driver"], {})
        predictions.append(DriverPredictionSchema(
            driver=p["driver"],
            team=p["team"],
            predicted_position=p["predicted_position"],
            win_probability=p["win_probability"],
            top3_probability=p["top3_probability"],
            top10_probability=p["top10_probability"],
            dnf_probability=p["dnf_probability"],
            confidence_interval=ConfidenceInterval(
                p10=p.get("ci_p10", p["predicted_position"] - 4),
                p90=p.get("ci_p90", p["predicted_position"] + 4),
            ),
            dark_horse=p.get("dark_horse", False),
            wet_specialist=p.get("wet_specialist", False),
            explanation=[
                ExplanationFactor(factor=f["factor"], description=f["description"])
                for f in exp.get("top_factors", [])
            ],
        ))

    response = RacePredictionResponse(
        race=circuit_name,
        round_id=round_id,
        circuit_name=circuit_name,
        model_branch=branch,
        generated_at=datetime.now(timezone.utc),
        weather=weather,
        predictions=predictions,
    )

    # Persist and cache
    response_dict = response.model_dump(mode="json")
    _prediction_cache[round_id] = response_dict
    pred_file = PREDICTIONS_DIR / f"round_{round_id:02d}.json"
    with open(pred_file, "w") as f:
        json.dump(response_dict, f, indent=2, default=str)
    logger.info(f"Prediction saved to {pred_file}")

    return response


@app.get(
    "/prediction/{round_id}",
    response_model=RacePredictionResponse,
    tags=["Predictions"],
    summary="Retrieve last stored prediction for a round",
)
async def get_prediction(round_id: int):
    """Returns the stored prediction for a round (no recomputation)."""
    pred_file = PREDICTIONS_DIR / f"round_{round_id:02d}.json"
    if not pred_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No prediction found for round {round_id}. Call POST /predict/{round_id} first."
        )
    with open(pred_file) as f:
        return JSONResponse(json.load(f))


@app.post(
    "/calibrate",
    response_model=CalibrationResponse,
    tags=["Calibration"],
    summary="Post-race calibration with actual results",
)
async def post_race_calibration(request: PostRaceCalibrationRequest):
    """
    Submit actual race results to trigger post-race calibration.
    Updates all sub-model posteriors and logs Spearman accuracy.
    Should be called within 2 hours of race finish (FR-4).
    """
    logger.info(f"POST /calibrate — Round {request.round_id} {request.race_name}")

    pred_file = PREDICTIONS_DIR / f"round_{request.round_id:02d}.json"
    if not pred_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No prediction found for round {request.round_id} — cannot calibrate without predicted data."
        )

    with open(pred_file) as f:
        pred_data = json.load(f)

    # Reconstruct DriverPrediction-like objects for Spearman computation
    from models.meta_ensemble import DriverPrediction
    predicted = [
        type("P", (), {
            "driver": p["driver"],
            "predicted_position": p["predicted_position"]
        })()
        for p in pred_data["predictions"]
    ]

    actual = [r.model_dump() for r in request.results]

    calibrator = AutoCalibrator()
    result = calibrator.calibrate(actual, predicted, request.round_id, request.race_name)

    return CalibrationResponse(**result)


@app.post(
    "/export/prediction-card",
    tags=["Export"],
    summary="Generate PNG prediction card for social media",
    response_class=StreamingResponse,
)
async def export_prediction_card(request: PredictionCardRequest):
    """
    Generates a styled PNG prediction card for Xtension Labs social media.
    Returns a binary PNG stream.
    Top-N drivers shown in an F1-themed dark card with team colours.
    """
    logger.info(f"POST /export/prediction-card — {request.race}, top {request.top_n}")

    # Load the latest prediction for this race
    matching = list(PREDICTIONS_DIR.glob("*.json"))
    latest_pred = None
    for f in matching:
        with open(f) as fp:
            data = json.load(fp)
        if data.get("circuit_name") == request.race:
            latest_pred = data
            break

    if not latest_pred:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction found for {request.race}. Run POST /predict first."
        )

    png_bytes = _generate_card_png(
        predictions=latest_pred["predictions"][: request.top_n],
        race_name=request.race,
        model_branch=latest_pred.get("model_branch", "DRY_STANDARD"),
        theme=request.theme,
    )

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="f1_apex_{request.race.lower().replace(" ", "_")}.png"'},
    )


# ── Card generator ────────────────────────────────────────────────────────────

def _generate_card_png(
    predictions: list[dict],
    race_name: str,
    model_branch: str,
    theme: str = "dark",
) -> bytes:
    """
    Generate a styled PNG prediction card using Pillow.
    """
    from PIL import Image, ImageDraw, ImageFont

    # ── Layout ────────────────────────────────────────────────────────────
    W, H    = 800, 120 + len(predictions) * 60 + 60
    BG      = (10, 10, 18)     if theme == "dark" else (245, 245, 248)
    FG      = (255, 255, 255)  if theme == "dark" else (10, 10, 18)
    ACCENT  = (229, 26, 26)    # F1 red

    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # ── Fonts (fallback to default if custom not available) ───────────────
    try:
        font_title = ImageFont.truetype("arial.ttf", 28)
        font_sub   = ImageFont.truetype("arial.ttf", 16)
        font_row   = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub   = font_title
        font_row   = font_title

    # ── Header ────────────────────────────────────────────────────────────
    draw.rectangle([0, 0, W, 5], fill=ACCENT)
    draw.text((30, 20), "F1 APEX", font=font_title, fill=ACCENT)
    draw.text((170, 26), "× Xtension Labs", font=font_sub, fill=(180, 180, 180))
    draw.text((30, 58), f"{race_name} 2026 — Top {len(predictions)} Prediction", font=font_sub, fill=FG)

    branch_color = (70, 130, 200) if model_branch == "WET_WEATHER" else (100, 200, 100)
    draw.text((30, 80), f"Model: {model_branch}", font=font_sub, fill=branch_color)

    # ── Rows ──────────────────────────────────────────────────────────────
    y = 115
    for idx, p in enumerate(predictions):
        pos   = int(round(p["predicted_position"]))
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, f"P{pos}")

        row_bg = (20, 20, 30) if idx % 2 == 0 else (15, 15, 22)
        if theme == "light":
            row_bg = (235, 235, 240) if idx % 2 == 0 else (225, 225, 232)

        draw.rectangle([0, y, W, y + 55], fill=row_bg)
        draw.text((20, y + 15), medal, font=font_row, fill=FG)
        draw.text((90, y + 15), p["driver"], font=font_row, fill=FG)
        draw.text((310, y + 15), p["team"], font=font_sub, fill=(160, 160, 160))
        draw.text((560, y + 15), f"Win: {p['win_probability']:.0%}", font=font_sub, fill=ACCENT)
        draw.text((670, y + 15), f"DNF: {p['dnf_probability']:.0%}", font=font_sub, fill=(200, 100, 100))

        y += 58

    # ── Footer ────────────────────────────────────────────────────────────
    draw.rectangle([0, H - 35, W, H], fill=(18, 18, 28))
    draw.text((20, H - 26), "f1-apex.xtension.io | @xtension.io", font=font_sub, fill=(120, 120, 120))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
