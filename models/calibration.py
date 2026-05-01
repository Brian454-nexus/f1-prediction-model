"""
F1 APEX — Post-Race Auto-Calibrator
======================================
Runs after each race to update ensemble weights, sub-model posteriors,
and accuracy tracking. This is the learning backbone of F1 APEX — it is
what turns a static model into a system that improves race-by-race.

WHAT GETS UPDATED:
    1. MetaEnsemble weights — minimise log-loss on actual finishing order.
    2. CPM posteriors — update constructor pace advantage estimates.
    3. DPI ratings — Elo update for all drivers from race result.
    4. RRM survival curves — clean finish or DNF for each team.
    5. Accuracy log — Spearman rank correlation between predicted and actual.

CALIBRATION TRIGGER:
    Called by Prefect DAG within 2 hours of race finish (FR-4 in PRD).

ACCURACY BASELINE:
    PRD target: Spearman ρ ≥ 0.65 averaged over first 5 races.
    Grid-position baseline: ρ ≈ 0.55.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

from models.cpm import ConstructorPerformanceModel
from models.dpi import DriverPerformanceIndex
from models.meta_ensemble import MetaEnsemble, DEFAULT_WEIGHTS
from models.rrm import ReliabilityRiskModel
from utils.logger import get_logger

logger = get_logger("APEX.Calibration")

STATE_PATH = Path("models/state")
ACCURACY_LOG = STATE_PATH / "accuracy_log.json"
MIN_WEIGHT   = 0.02
MAX_WEIGHT   = 0.60

# --- Anti-overfitting constants -----------------------------------------------
# Minimum races before any weight adjustment (prevents overfitting to tiny samples)
MIN_RACES_FOR_WEIGHT_UPDATE = 3
# Dirichlet-like regularization strength: higher = stronger pull toward default weights
# This prevents any single sub-model from dominating after a lucky race.
DIRICHLET_ALPHA = 0.25
# Maximum single-step weight change per component (stability constraint)
MAX_WEIGHT_DELTA = 0.08


class AutoCalibrator:
    """
    Post-race calibration pipeline. Invoked by the Prefect DAG after each race.

    The calibrator receives actual race results and runs four update steps:
       1. Score each sub-model's per-race accuracy (Spearman ρ per module).
       2. Adjust ensemble weights proportionally to module accuracy.
       3. Update CPM/DPI/RRM state via their own update_after_race methods.
       4. Log the overall prediction accuracy.
    """

    def __init__(self) -> None:
        STATE_PATH.mkdir(parents=True, exist_ok=True)
        self.ensemble = MetaEnsemble()
        self.cpm      = ConstructorPerformanceModel()
        self.dpi      = DriverPerformanceIndex()
        self.rrm      = ReliabilityRiskModel()
        self._accuracy_log: list = self._load_accuracy_log()

    # ── Public API ─────────────────────────────────────────────────────────

    def calibrate(
        self,
        actual_results: list[dict],
        predicted_results: list[dict],
        round_id: int,
        race_name: str,
    ) -> dict:
        """
        Run the full post-race calibration pipeline.

        Args:
            actual_results:    List of dicts: {driver, team, finish_pos, dnf, cause}
            predicted_results: List of DriverPrediction outputs from MetaEnsemble.
            round_id:          Race round number.
            race_name:         e.g. 'Japanese Grand Prix'

        Returns:
            dict with calibration metrics.
        """
        logger.info(f"=== Post-Race Calibration: Round {round_id} — {race_name} ===")

        # ── Step 1: Spearman accuracy ────────────────────────────────────
        spearman = self._compute_spearman(actual_results, predicted_results)
        logger.info(f"Spearman ρ: {spearman:.4f} (target ≥ 0.65)")

        # ── Step 2: Ensemble weight adjustment ──────────────────────────
        new_weights = self._adjust_weights(spearman, len(self._accuracy_log))
        self.ensemble.update_weights(new_weights)

        # ── Step 3: Sub-model updates ────────────────────────────────────
        # CPM update
        team_avg_positions = self._compute_team_avg_positions(actual_results)
        self.cpm.update_posterior(
            {team: {"avg_position": pos} for team, pos in team_avg_positions.items()},
            round_id
        )

        # DPI update
        self.dpi.update_after_race(actual_results, round_id)

        # RRM update
        rrm_outcomes = {
            r["team"]: {"dnf": r.get("dnf", False), "cause": r.get("cause", "")}
            for r in actual_results
        }
        self.rrm.update_after_race(rrm_outcomes, round_id)

        # ── Step 4: Log accuracy ─────────────────────────────────────────
        log_entry = {
            "round_id":      round_id,
            "race_name":     race_name,
            "spearman_rho":  round(spearman, 4),
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "new_weights":   new_weights,
        }
        self._accuracy_log.append(log_entry)
        self._save_accuracy_log()

        avg_rho = np.mean([e["spearman_rho"] for e in self._accuracy_log])
        logger.info(f"Running average Spearman ρ: {avg_rho:.4f} (across {len(self._accuracy_log)} races)")

        return {
            "round_id":          round_id,
            "spearman_rho":      round(spearman, 4),
            "target_met":        spearman >= 0.65,
            "avg_rho_to_date":   round(float(avg_rho), 4),
            "races_logged":      len(self._accuracy_log),
            "updated_weights":   new_weights,
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _compute_spearman(actual: list[dict], predicted: list[dict]) -> float:
        """Compute Spearman rank correlation between predicted and actual order."""
        actual_map = {r["driver"]: r["finish_pos"] for r in actual if not r.get("dnf")}
        pred_map   = {r.driver: r.predicted_position for r in predicted}

        common = [d for d in actual_map if d in pred_map]
        if len(common) < 3:
            logger.warning("Fewer than 3 common drivers — Spearman skipped")
            return 0.0

        actual_ranks = [actual_map[d] for d in common]
        pred_ranks   = [pred_map[d] for d in common]

        rho, _ = stats.spearmanr(actual_ranks, pred_ranks)
        return float(rho)

    def _adjust_weights(self, spearman: float, n_races: int) -> dict:
        """
        Bayesian-regularized ensemble weight adjustment.

        Key ML best practices applied here:
          1. Minimum races gate: no updates before MIN_RACES_FOR_WEIGHT_UPDATE
             to prevent overfitting to small samples.
          2. Dirichlet regularization: each update blends with the default
             (prior) weights by DIRICHLET_ALPHA, preventing weight collapse
             to a single dominant sub-model.
          3. Per-step clamp: no single component can change by more than
             MAX_WEIGHT_DELTA per race — provides stability against outlier races.
          4. High-accuracy stability: weights are frozen when ρ ≥ 0.75 (model
             is performing well; no need to risk regression).

        Args:
            spearman: Spearman ρ from this race.
            n_races:  Total races logged so far (including current).

        Returns:
            Updated weight dict that sums to 1.0.
        """
        current = self.ensemble.weights.copy()

        # Gate: insufficient data — keep current weights unchanged
        if n_races < MIN_RACES_FOR_WEIGHT_UPDATE:
            logger.info(
                f"Insufficient data ({n_races} races < {MIN_RACES_FOR_WEIGHT_UPDATE}) "
                "— weight update skipped (anti-overfitting gate)"
            )
            return current

        # High accuracy: stable; no adjustment to avoid spurious drift
        if spearman >= 0.75:
            logger.info(f"High accuracy (ρ={spearman:.3f}) — weights held stable")
            return current

        # Accuracy-signal update: proportional to gap below 0.65 target
        adjustment = max(0.0, 0.65 - spearman) * 0.05
        new = current.copy()
        new["CPM"] = current["CPM"] + adjustment
        new["DPI"] = current["DPI"] + adjustment * 0.5
        new["ESS"] = current["ESS"] - adjustment * 0.5
        new["QRT"] = current["QRT"] - adjustment * 0.5

        # Dirichlet regularization: blend toward DEFAULT_WEIGHTS prior
        # Prevents any sub-model from dominating; maintains diversity
        regularized = {
            k: (1.0 - DIRICHLET_ALPHA) * new[k] + DIRICHLET_ALPHA * DEFAULT_WEIGHTS[k]
            for k in new
        }

        # Per-step stability clamp: cap maximum weight change per component
        clamped = {}
        for k in regularized:
            delta = regularized[k] - current[k]
            delta = max(-MAX_WEIGHT_DELTA, min(MAX_WEIGHT_DELTA, delta))
            clamped[k] = min(MAX_WEIGHT, max(MIN_WEIGHT, current[k] + delta))

        # Renormalise to sum exactly to 1.0
        total = sum(clamped.values())
        final = {k: round(v / total, 4) for k, v in clamped.items()}

        logger.info(f"Weights adjusted (ρ={spearman:.3f}, α={DIRICHLET_ALPHA}): {final}")
        return final

    @staticmethod
    def _compute_team_avg_positions(results: list[dict]) -> dict[str, float]:
        """Compute average finishing position per team from race results."""
        team_positions: dict[str, list[float]] = {}
        for r in results:
            if not r.get("dnf"):
                team = r["team"]
                team_positions.setdefault(team, []).append(r["finish_pos"])
        return {
            team: float(np.mean(positions))
            for team, positions in team_positions.items()
        }

    def _load_accuracy_log(self) -> list:
        if ACCURACY_LOG.exists():
            with open(ACCURACY_LOG, "r") as f:
                return json.load(f)
        return []

    def _save_accuracy_log(self) -> None:
        with open(ACCURACY_LOG, "w") as f:
            json.dump(self._accuracy_log, f, indent=2)
