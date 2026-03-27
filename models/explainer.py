"""
F1 APEX — SHAP Explainability Layer
=======================================
Generates human-readable explanations for each prediction output.

PURPOSE:
    The PRD requires that 100% of predictions include a natural language
    rationale. No black-box outputs to end users.
    This module converts SHAP feature importances into the top-3 contributing
    factors expressed in plain language.

    FR-3 requirement: Each prediction includes a top-3 contributing factors
    explanation (e.g. 'Mercedes energy deployment advantage at Suzuka
    estimated at +0.3s/lap').

METHODOLOGY:
    In full production: SHAP TreeExplainer on the fitted LightGBM model.
    In v1.0 (early season, insufficient LightGBM training data): uses a
    rule-based template system that maps the highest-magnitude sub-model
    contributions to pre-written natural language templates.
    This is transparent and explainable without requiring SHAP library
    access on circuits where we have limited 2026 data.
"""

from __future__ import annotations

from dataclasses import dataclass

import sys
from pathlib import Path

# Ensure project root is on path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from utils.logger import get_logger

logger = get_logger("APEX.Explainer")


@dataclass
class ExplanationFactor:
    """A single contributing factor with magnitude and direction."""
    factor_name:  str
    contribution: float    # + = helped this driver | - = hurt this driver
    description:  str      # Human-readable explanation


@dataclass
class PredictionExplanation:
    """Full explanation for a single driver's prediction."""
    driver:         str
    team:           str
    predicted_pos:  float
    top_factors:    list[ExplanationFactor]
    summary:        str    # One-sentence plain English summary


# ── Template library ──────────────────────────────────────────────────────────

def _pace_template(team: str, pace_advantage: float, circuit: str) -> str:
    if pace_advantage > 0.5:
        return (f"{team} car holds a dominant {pace_advantage:+.2f}s/lap pace advantage "
                f"at {circuit} based on 2026 performance data.")
    elif pace_advantage > 0:
        return (f"{team} has a moderate {pace_advantage:+.2f}s/lap pace edge "
                f"at {circuit}.")
    elif pace_advantage < -0.3:
        return (f"{team} is structurally disadvantaged at {circuit} "
                f"({pace_advantage:+.2f}s/lap vs median field).")
    return f"{team} pace is approximately median field at {circuit}."


def _dpi_template(driver: str, dpi_score: float, is_rookie: bool) -> str:
    if is_rookie:
        return (f"{driver} is a 2026 rookie — wide confidence intervals applied "
                f"due to limited data (DPI: {dpi_score:.0f}, CI inflated).")
    if dpi_score > 1750:
        return f"{driver} is an elite-tier driver (DPI {dpi_score:.0f}) — consistently extracts value above their machinery."
    if dpi_score > 1600:
        return f"{driver} is a top-tier driver (DPI {dpi_score:.0f})."
    return f"{driver} is rated at DPI {dpi_score:.0f} — midfield driver performance."


def _reliability_template(team: str, dnf_prob: float, risk_tier: str) -> str:
    if risk_tier == "CRITICAL":
        return (f"{team} reliability is CRITICAL — {dnf_prob:.0%} DNF risk. "
                f"Systemic power unit failures in 2026 R1+R2.")
    if risk_tier == "HIGH":
        return f"{team} carries HIGH mechanical risk ({dnf_prob:.0%} DNF probability)."
    if risk_tier == "LOW":
        return f"{team} has strong reliability ({dnf_prob:.0%} DNF risk) — {4-round(dnf_prob*4)}/4 race finishes."
    return f"{team} reliability is MEDIUM ({dnf_prob:.0%} DNF probability)."


def _energy_template(team: str, energy_gain: float, circuit: str, clipping_prob: float) -> str:
    if energy_gain > 6.0:
        return (f"{team} holds a strong energy deployment advantage at {circuit} "
                f"(~{energy_gain:+.1f}s total gain). Clipping risk: {clipping_prob:.1%}.")
    if energy_gain < -3.0:
        return (f"{team} faces an energy strategy deficit at {circuit} "
                f"({energy_gain:+.1f}s total). Battery clipping risk: {clipping_prob:.1%}.")
    return (f"{team} energy strategy is approximately neutral at {circuit} "
            f"({energy_gain:+.1f}s). Clipping risk: {clipping_prob:.1%}.")


def _circuit_template(team: str, cdc_modifier: float, archetype: str) -> str:
    if cdc_modifier > 0.10:
        return f"{team}'s 2026 car specification has strong affinity for {archetype} circuits (+{cdc_modifier:.2f}s/lap)."
    if cdc_modifier < -0.10:
        return f"{team}'s car is poorly suited to this {archetype} circuit ({cdc_modifier:+.2f}s/lap)."
    return f"{team} has neutral circuit affinity for this {archetype} layout."


# ── Explainer Class ───────────────────────────────────────────────────────────

class ModelExplainer:
    """
    Generates top-3 natural language contributing factors for each prediction.

    In v1.0, uses rule-based templates driven by sub-model magnitudes.
    In v2.0, replaces template selection with SHAP TreeExplainer on LightGBM.
    """

    def explain(
        self,
        driver: str,
        team: str,
        predicted_pos: float,
        cpm_pace_advantage: float,
        dpi_score: float,
        is_rookie: bool,
        dnf_probability: float,
        risk_tier: str,
        ess_gain: float,
        ess_clipping_prob: float,
        cdc_modifier: float,
        circuit_archetype: str,
        circuit_name: str,
    ) -> PredictionExplanation:
        """
        Build explanation for a single driver's prediction.
        The top-3 factors are selected by absolute contribution magnitude.
        """

        # Score each factor by absolute magnitude (higher = more impactful)
        factors_raw = [
            ExplanationFactor(
                factor_name="Constructor Pace (CPM)",
                contribution=cpm_pace_advantage,
                description=_pace_template(team, cpm_pace_advantage, circuit_name),
            ),
            ExplanationFactor(
                factor_name="Driver Skill (DPI)",
                contribution=(dpi_score - 1500) / 100,
                description=_dpi_template(driver, dpi_score, is_rookie),
            ),
            ExplanationFactor(
                factor_name="Reliability Risk (RRM)",
                contribution=-dnf_probability * 5,  # Negative = hurts prediction
                description=_reliability_template(team, dnf_probability, risk_tier),
            ),
            ExplanationFactor(
                factor_name="Energy Strategy (ESS)",
                contribution=ess_gain / 10,
                description=_energy_template(team, ess_gain, circuit_name, ess_clipping_prob),
            ),
            ExplanationFactor(
                factor_name="Circuit DNA (CDC)",
                contribution=cdc_modifier * 3,
                description=_circuit_template(team, cdc_modifier, circuit_archetype),
            ),
        ]

        # Sort by absolute contribution and take top 3
        sorted_factors = sorted(factors_raw, key=lambda f: abs(f.contribution), reverse=True)
        top_3 = sorted_factors[:3]

        # Generate one-sentence summary from the top factor
        top = top_3[0]
        if top.contribution > 0:
            verdict = f"High confidence P{predicted_pos:.0f} based primarily on {top.factor_name}."
        else:
            verdict = f"Headwinds from {top.factor_name} push predicted finish back to P{predicted_pos:.0f}."

        logger.info(f"Explanation for {driver}: top factor = {top.factor_name} ({top.contribution:+.2f})")

        return PredictionExplanation(
            driver=driver,
            team=team,
            predicted_pos=predicted_pos,
            top_factors=top_3,
            summary=verdict,
        )

    def explain_all(self, driver_predictions: list[dict]) -> list[PredictionExplanation]:
        """Generate explanations for a full field of predictions."""
        return [self.explain(**d) for d in driver_predictions]


if __name__ == "__main__":
    explainer = ModelExplainer()
    exp = explainer.explain(
        driver="Russell", team="Mercedes", predicted_pos=1.5,
        cpm_pace_advantage=1.20, dpi_score=1640, is_rookie=False,
        dnf_probability=0.02, risk_tier="LOW",
        ess_gain=8.5, ess_clipping_prob=0.05,
        cdc_modifier=0.15, circuit_archetype="high_speed_low_deg",
        circuit_name="Suzuka",
    )
    print(f"\n── Explanation: {exp.driver} ──")
    print(f"Summary: {exp.summary}\n")
    for i, f in enumerate(exp.top_factors, 1):
        print(f"  {i}. [{f.factor_name}] {f.description}")
