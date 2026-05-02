"""
F1 APEX — Wet Weather Prediction Model
=========================================
Conditional prediction branch triggered when P(rain) > 30% at T-3h.

TRIGGER:
    WeatherRouter returns route_to_wet_model = True.
    The Prefect DAG routes here instead of MetaEnsemble.

METHODOLOGY:
    Uses the same Parquet-serialized sub-model outputs as the standard dry
    MetaEnsemble, but applies wet-specific ensemble weights:
      - DPI weight elevated to 35% (driver skill dominates in wet conditions)
      - ESS weight reduced to 10% (energy strategy less predictable on wet circuits)
      - RRM weight elevated to 20% (reliability risk increases in wet)

    A wet-weather position variance multiplier is applied — increasing the
    spread of outcomes to reflect the higher unpredictability of wet races
    (safety cars, red flags, tyre management divergence).

WET-WEATHER FACTORS:
    1. Safety car probability: Elevated to 0.85 (vs 0.55 dry baseline)
    2. DPI driver-advantage amplified: Verstappen/Hamilton wet-weather
       mastery is rewarded more than in dry races
    3. Reliability risk amplified: McLaren/Aston Martin risk increases further
       under thermal/hydraulic stress from rain
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from utils.logger import get_logger

logger = get_logger("APEX.WetWeather")

# ── Wet Weather Config ────────────────────────────────────────────────────────

WET_ENSEMBLE_WEIGHTS = {
    "CPM": 0.20,
    "DPI": 0.35,    # Elevated — driver skill paramount
    "ESS": 0.10,    # Reduced — energy strategy less predictable
    "RRM": 0.20,    # Elevated — reliability failures more likely
    "CDC": 0.05,
    "QRT": 0.10,
}

# Wet conditions amplify DPI advantage/disadvantage
WET_DPI_MULTIPLIER = 1.25

# Safety car probability under wet conditions
WET_SC_PROBABILITY = 0.85

# Reliability risk amplification factor in wet conditions
WET_RELIABILITY_MULTIPLIER = 1.30

# Notable wet-weather specialists (historical DPI advantage in rain)
WET_SPECIALISTS: dict[str, float] = {
    "Hamilton":   +100,
    "Verstappen": +80,
    "Alonso":     +70,
    "Russell":    +40,
    "Leclerc":    +30,
    "Norris":     +25,
}


@dataclass
class WetWeatherPrediction:
    """Outcome for a single driver in the wet-weather model."""
    driver:              str
    team:                str
    expected_finish:     float
    win_probability:     float
    top3_probability:    float
    top10_probability:   float
    dnf_probability:     float
    wet_specialist_flag: bool
    finish_probs:        dict = field(default_factory=dict)


class WetWeatherModel:
    """
    Wet-weather prediction ensemble.

    Consumes the standard sub-model outputs (same Parquet intermediates),
    re-weights them using wet-specific weights, and applies wet condition
    adjustments before generating the final probability distribution.
    """

    # 2026 grid: 11 constructors × 2 drivers = 22 total
    TOTAL_DRIVERS = 22

    def __init__(self) -> None:
        self.weights = WET_ENSEMBLE_WEIGHTS
        logger.info(f"WetWeatherModel initialised — DPI weight: {self.weights['DPI']}")

    def predict(
        self,
        driver_inputs: list[dict],
        rain_probability: float,
    ) -> list[WetWeatherPrediction]:
        """
        Generate wet-weather race predictions for all drivers.

        Args:
            driver_inputs: List of per-driver dicts, each containing:
                - driver, team, grid_position, dpi_score, dnf_probability,
                  energy_advantage, cpm_pace_advantage, cdc_modifier
            rain_probability: Float 0-1 from WeatherRouter.

        Returns:
            List of WetWeatherPrediction sorted by expected finish position.
        """
        logger.info(
            f"Running WET WEATHER MODEL | P(rain)={rain_probability:.1%} | "
            f"{len(driver_inputs)} drivers"
        )

        predictions: list[WetWeatherPrediction] = []

        for driver_data in driver_inputs:
            pred = self._predict_driver(driver_data, rain_probability)
            predictions.append(pred)

        # Normalize win probabilities to sum to 1
        total_win_prob = sum(p.win_probability for p in predictions)
        if total_win_prob > 0:
            for p in predictions:
                p.win_probability = round(p.win_probability / total_win_prob, 4)

        return sorted(predictions, key=lambda x: x.expected_finish)

    def _predict_driver(self, data: dict, rain_prob: float) -> WetWeatherPrediction:
        """Apply wet-weighted ensemble for a single driver."""
        driver = data["driver"]
        team   = data["team"]

        # Wet specialist DPI bonus
        wet_bonus = WET_SPECIALISTS.get(driver, 0)
        adjusted_dpi = data["dpi_score"] + wet_bonus * (rain_prob / 0.30) * 0.5

        # Elevated RRM under wet conditions
        wet_dnf_prob = min(
            0.95,
            data.get("rrm_dnf", data.get("dnf_probability", 0.10)) * (1 + (rain_prob - 0.30) * WET_RELIABILITY_MULTIPLIER)
        )

        # Weighted ensemble score
        raw_score = (
            self.weights["CPM"] * data.get("cpm_pace_advantage", 0) +
            self.weights["DPI"] * (adjusted_dpi - 1500) / 1500 * 5.0 +
            self.weights["ESS"] * data.get("energy_advantage", 0) +
            self.weights["CDC"] * data.get("cdc_modifier", 0) +
            self.weights["QRT"] * (10 - data.get("grid_position", 10)) * 0.3
        )

        # Map score to expected finish position (higher score → lower position number)
        # Scale: score ≈ [-2, 2] mapped to positions [1, 22]
        # Midpoint of 1-22 grid = 11.5; half-range = 10.5; scale = 10.5/2 = 5.25
        raw_finish = 11.5 - raw_score * 5.25
        expected_finish = max(1.0, min(float(self.TOTAL_DRIVERS), raw_finish))

        finish_probs = self._position_distribution(expected_finish, std=4.5, n_drivers=self.TOTAL_DRIVERS)
        p_finish = 1 - wet_dnf_prob
        finish_probs = {pos: prob * p_finish for pos, prob in finish_probs.items()}

        return WetWeatherPrediction(
            driver=driver,
            team=team,
            expected_finish=round(expected_finish, 1),
            win_probability=finish_probs.get(1, 0.0),
            top3_probability=sum(finish_probs.get(p, 0.0) for p in range(1, 4)),
            top10_probability=sum(finish_probs.get(p, 0.0) for p in range(1, 11)),
            dnf_probability=round(wet_dnf_prob, 4),
            wet_specialist_flag=driver in WET_SPECIALISTS,
            finish_probs=finish_probs,
        )

    @staticmethod
    def _position_distribution(mean: float, std: float, n_drivers: int = 22) -> dict[int, float]:
        """Discretised Normal over positions 1-n_drivers (default: 22 for the 2026 grid)."""
        raw = {p: np.exp(-0.5 * ((p - mean) / std) ** 2) for p in range(1, n_drivers + 1)}
        total = sum(raw.values())
        return {pos: round(v / total, 6) for pos, v in raw.items()}


if __name__ == "__main__":
    model = WetWeatherModel()
    sample_drivers = [
        {"driver": "Russell",     "team": "Mercedes",    "grid_position": 1,
         "dpi_score": 1640, "dnf_probability": 0.02, "energy_advantage": 0.30,
         "cpm_pace_advantage": 1.2, "cdc_modifier": 0.15},
        {"driver": "Verstappen",  "team": "Red Bull",    "grid_position": 5,
         "dpi_score": 1820, "dnf_probability": 0.22, "energy_advantage": -0.30,
         "cpm_pace_advantage": -0.50, "cdc_modifier": -0.10},
        {"driver": "Hamilton",    "team": "Ferrari",     "grid_position": 3,
         "dpi_score": 1750, "dnf_probability": 0.05, "energy_advantage": 0.00,
         "cpm_pace_advantage": 0.00, "cdc_modifier": 0.05},
    ]
    results = model.predict(sample_drivers, rain_probability=0.65)
    print("\n── Wet Weather Predictions ──")
    for r in results:
        flag = " 🌧️ WET SPECIALIST" if r.wet_specialist_flag else ""
        print(f"  {r.driver:<14}  P{r.expected_finish:.1f}  "
              f"Win:{r.win_probability:.1%}  DNF:{r.dnf_probability:.1%}{flag}")
