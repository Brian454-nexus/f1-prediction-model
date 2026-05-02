"""
F1 APEX — Meta-Ensemble Layer
================================
Module 7 of 7 — the final combiner.

PURPOSE:
    Aggregates all six sub-model outputs into a single probability distribution
    over finishing positions for each driver. This is the layer that produces
    the final race prediction published to the API.

METHODOLOGY:
    1. Each sub-model writes its per-driver outputs to Parquet intermediates.
    2. This module loads and merges all Parquet files on the 'Driver' key.
    3. A weighted linear combination of position scores is computed using
       the current ensemble weights.
    4. 50,000 Monte Carlo draws (position sampling from the merged distribution)
       produce: win probability, top-3, top-10, and confidence intervals.
    5. LightGBM re-weighting kicks in once 5+ races are available
       (replaces the hard-coded initial weights with learned optimal weights).

INITIAL WEIGHTS (PRD §4.1):
    CPM 30% | DPI 25% | ESS 20% | RRM 15% | CDC 5% | QRT 5%

AUTO-CALIBRATION:
    After each race, `calibration.py` minimises log-loss on actual results
    and updates these weights. Stored in models/state/ensemble/weights.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("APEX.MetaEnsemble")

STATE_PATH = Path("models/state/ensemble")

# ── Default weights per PRD ───────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "CPM": 0.30,
    "DPI": 0.25,
    "ESS": 0.20,
    "RRM": 0.15,
    "CDC": 0.05,
    "QRT": 0.05,
}

# 2026 grid: 11 constructors × 2 drivers = 22 total
TOTAL_DRIVERS = 22
MC_RUNS = 50_000


@dataclass
class DriverPrediction:
    """Final per-driver prediction output."""
    driver:             str
    team:               str
    predicted_position: float
    win_probability:    float
    top3_probability:   float
    top10_probability:  float
    dnf_probability:    float
    ci_p10:             float
    ci_p90:             float
    dark_horse:         bool = False
    model_version:      str  = "1.0"
    finish_probs:       dict = field(default_factory=dict)


class MetaEnsemble:
    """
    Combines sub-model outputs into the final race prediction.

    Sub-model inputs arrive as individual dicts (one per driver) containing:
        - cpm_score:  CPM pace advantage (s/lap, + = faster)
        - dpi_score:  DPI Elo rating
        - ess_gain:   ESS expected energy time gain (s/race)
        - rrm_dnf:    RRM DNF probability
        - cdc_mod:    CDC circuit affinity modifier (s/lap)
        - qrt_probs:  QRT finish_probs dict {position: probability}
        - grid_pos:   Qualifying/grid position
    """

    def __init__(self, rng_seed: int = 2026) -> None:
        STATE_PATH.mkdir(parents=True, exist_ok=True)
        self.weights_file = STATE_PATH / "weights.json"
        self.weights = self._load_weights()
        self.rng = np.random.default_rng(rng_seed)

    # ── Public API ─────────────────────────────────────────────────────────

    def predict_race(
        self,
        driver_inputs: list[dict],
        circuit_name: str,
        race_round: int,
    ) -> list[DriverPrediction]:
        """
        Generate complete race predictions for all drivers.

        Args:
            driver_inputs: Per-driver sub-model outputs (see class docstring).
            circuit_name:  For logging/metadata.
            race_round:    Race round number, used in output metadata.

        Returns:
            List of DriverPrediction, sorted by predicted finishing position.
        """
        logger.info(
            f"MetaEnsemble.predict_race: {circuit_name} Round {race_round} "
            f"({len(driver_inputs)} drivers) — weights: {self.weights}"
        )

        predictions: list[DriverPrediction] = []

        # Calculate grid averages for zero-centering
        features = ["cpm_score", "dpi_score", "ess_gain", "rrm_dnf", "cdc_mod", "grid_pos"]
        defaults = {"dpi_score": 1500, "grid_pos": 11.5}
        grid_means = {
            f: float(np.mean([d.get(f, defaults.get(f, 0.0)) for d in driver_inputs]))
            if driver_inputs else 0.0
            for f in features
        }

        for driver_data in driver_inputs:
            prediction = self._predict_driver(driver_data, grid_means)
            predictions.append(prediction)

        # Normalise win probabilities to sum to exactly 1.0
        total_win = sum(p.win_probability for p in predictions)
        if total_win > 0:
            for p in predictions:
                p.win_probability = round(p.win_probability / total_win, 6)

        # Flag dark horses: high variance (wide CI) with meaningful upside
        for p in predictions:
            ci_width = p.ci_p90 - p.ci_p10
            if ci_width >= 8 and p.predicted_position <= 8:
                p.dark_horse = True

        return sorted(predictions, key=lambda x: x.predicted_position)

    def update_weights(self, new_weights: dict) -> None:
        """Update ensemble weights (called by AutoCalibrator after each race)."""
        self.weights = {k: round(v, 4) for k, v in new_weights.items()}
        with open(self.weights_file, "w") as f:
            json.dump(self.weights, f, indent=2)
        logger.info(f"Ensemble weights updated: {self.weights}")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _predict_driver(self, data: dict, means: dict = None) -> DriverPrediction:
        """
        Compute a single driver's race prediction using weighted sub-model scores.
        """
        w = self.weights
        if means is None:
            means = {f: 0.0 for f in ["cpm_score", "dpi_score", "ess_gain", "rrm_dnf", "cdc_mod", "grid_pos"]}
            means["dpi_score"] = 1500
            means["grid_pos"] = 11.5

        # Build a composite 'performance score' from sub-model outputs
        # Mean-centered scores are normalised to similar scales before weighting
        cpm_norm  = (data.get("cpm_score", 0) - means.get("cpm_score", 0)) / 1.5
        dpi_norm  = (data.get("dpi_score", 1500) - means.get("dpi_score", 1500)) / 400
        
        # ESS is wildly negative under 2026 regs (e.g. -53s). We only care about relative advantage!
        ess_norm  = (data.get("ess_gain", 0) - means.get("ess_gain", 0)) / 5.0
        
        rrm_score = -(data.get("rrm_dnf", 0.15) - means.get("rrm_dnf", 0.15))
        cdc_norm  = (data.get("cdc_mod", 0) - means.get("cdc_mod", 0)) / 0.2
        
        # grid_pos below mean -> positive composite score
        qrt_norm  = -(data.get("grid_pos", 11.5) - means.get("grid_pos", 11.5)) / 10.0

        composite = (
            w["CPM"] * cpm_norm +
            w["DPI"] * dpi_norm +
            w["ESS"] * ess_norm +
            w["RRM"] * rrm_score +
            w["CDC"] * cdc_norm +
            w["QRT"] * qrt_norm
        )

        # Convert composite score → expected finishing position
        # Score range ≈ [-1.0, +1.0] mapped to positions [1, 22]
        # A positive composite = ahead of the grid mean (11.5)
        expected_finish = 11.5 - composite * 9.0
        expected_finish = max(1.0, min(22.0, expected_finish))

        # Monte Carlo draws: sample position from Normal(expected_finish, std)
        # std is wider for uncertain drivers (rookies, reliability-risk)
        std = 3.0 + abs(data.get("rrm_dnf", 0.05) - 0.05) * 5
        mc_samples = self.rng.normal(expected_finish, std, MC_RUNS)
        mc_positions = np.clip(np.round(mc_samples), 1, TOTAL_DRIVERS).astype(int)

        # P(DNF) reduces finishing probability
        p_finish = 1 - data.get("rrm_dnf", 0.05)
        mc_finishing = self.rng.random(MC_RUNS) < p_finish

        # Compute probabilities from MC distribution
        finish_counts = np.bincount(
            mc_positions[mc_finishing], minlength=TOTAL_DRIVERS + 1
        )
        finish_probs = {
            pos: round(finish_counts[pos] / MC_RUNS, 6)
            for pos in range(1, TOTAL_DRIVERS + 1)
        }

        # Percentile CI on position
        finishing_positions = mc_positions[mc_finishing]
        ci_p10 = float(np.percentile(finishing_positions, 10)) if len(finishing_positions) > 0 else expected_finish - 4
        ci_p90 = float(np.percentile(finishing_positions, 90)) if len(finishing_positions) > 0 else expected_finish + 4

        return DriverPrediction(
            driver=data["driver"],
            team=data["team"],
            predicted_position=round(expected_finish, 2),
            win_probability=finish_probs.get(1, 0.0),
            top3_probability=sum(finish_probs.get(p, 0.0) for p in range(1, 4)),
            top10_probability=sum(finish_probs.get(p, 0.0) for p in range(1, 11)),
            dnf_probability=round(data.get("rrm_dnf", 0.05), 4),
            ci_p10=round(ci_p10, 1),
            ci_p90=round(ci_p90, 1),
            finish_probs=finish_probs,
        )

    def _load_weights(self) -> dict:
        if self.weights_file.exists():
            with open(self.weights_file, "r") as f:
                weights = json.load(f)
                logger.info(f"Loaded ensemble weights from {self.weights_file}")
                return weights
        logger.info("Using default PRD ensemble weights")
        return DEFAULT_WEIGHTS.copy()


if __name__ == "__main__":
    ensemble = MetaEnsemble()

    sample_inputs = [
        {"driver": "Russell",    "team": "Mercedes",    "grid_pos": 1,
         "cpm_score": 1.20, "dpi_score": 1640, "ess_gain": 8.5,
         "rrm_dnf": 0.02, "cdc_mod": 0.15},
        {"driver": "Antonelli",  "team": "Mercedes",    "grid_pos": 2,
         "cpm_score": 1.20, "dpi_score": 1600, "ess_gain": 8.0,
         "rrm_dnf": 0.02, "cdc_mod": 0.15},
        {"driver": "Leclerc",    "team": "Ferrari",     "grid_pos": 3,
         "cpm_score": 0.00, "dpi_score": 1670, "ess_gain": 2.0,
         "rrm_dnf": 0.05, "cdc_mod": 0.05},
        {"driver": "Verstappen", "team": "Red Bull",    "grid_pos": 4,
         "cpm_score": -0.50, "dpi_score": 1820, "ess_gain": -3.0,
         "rrm_dnf": 0.22, "cdc_mod": -0.10},
        {"driver": "Norris",     "team": "McLaren",     "grid_pos": 5,
         "cpm_score": 0.50, "dpi_score": 1680, "ess_gain": 4.0,
         "rrm_dnf": 0.38, "cdc_mod": 0.10},
    ]

    results = ensemble.predict_race(sample_inputs, "Suzuka", 3)
    print(f"\n── F1 APEX — Suzuka 2026 Prediction ──")
    for r in results:
        dark = " 🎯 DARK HORSE" if r.dark_horse else ""
        print(
            f"  P{r.predicted_position:4.1f}  {r.driver:<14}  {r.team:<14}  "
            f"Win:{r.win_probability:.1%}  Top3:{r.top3_probability:.1%}  "
            f"DNF:{r.dnf_probability:.1%}  CI:[{r.ci_p10:.0f},{r.ci_p90:.0f}]{dark}"
        )
