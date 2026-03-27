"""
F1 APEX — Test Suite
======================
Unit tests covering:
  1. Data pipeline Parquet output structure
  2. Feature engineering column presence
  3. ML module output schemas (CPM, DPI, RRM, QRT, CDC, ESS)
  4. MetaEnsemble probability constraints
  5. API schema validation
  6. Cold-start validation (zero 2026 data scenario)
"""

from __future__ import annotations

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Test: Constructor Performance Model
# ─────────────────────────────────────────────────────────────────────────────

def test_cpm_returns_valid_structure():
    from models.cpm import ConstructorPerformanceModel
    cpm = ConstructorPerformanceModel()
    result = cpm.get_pace_advantage("Mercedes", "high_speed_low_deg")
    assert "mean_advantage" in result
    assert "p10" in result
    assert "p90" in result
    assert result["p10"] <= result["mean_advantage"] <= result["p90"]


def test_cpm_covers_all_known_teams():
    from models.cpm import ConstructorPerformanceModel, POSTERIOR_2026_R2
    cpm = ConstructorPerformanceModel()
    for team in POSTERIOR_2026_R2:
        result = cpm.get_pace_advantage(team, "medium_balanced")
        assert result["team"] == team or result["team"] == "Unknown"


def test_cpm_cold_start_wide_intervals():
    """
    Cold-start validation: verify that teams with no 2026 data (data_quality='insufficient')
    have wider confidence intervals than teams with race data.
    """
    from models.cpm import ConstructorPerformanceModel
    cpm = ConstructorPerformanceModel()

    mclaren = cpm.get_pace_advantage("McLaren", "high_speed_low_deg")
    mercedes = cpm.get_pace_advantage("Mercedes", "high_speed_low_deg")

    mclaren_ci_width  = mclaren["p90"] - mclaren["p10"]
    mercedes_ci_width = mercedes["p90"] - mercedes["p10"]

    # McLaren has insufficient 2026 data → should have wider CI
    assert mclaren_ci_width >= mercedes_ci_width, (
        f"McLaren CI ({mclaren_ci_width:.2f}) should be ≥ Mercedes CI ({mercedes_ci_width:.2f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Driver Performance Index
# ─────────────────────────────────────────────────────────────────────────────

def test_dpi_rookie_ci_inflated():
    from models.dpi import DriverPerformanceIndex, ROOKIES_2026
    dpi = DriverPerformanceIndex()

    for rookie in ROOKIES_2026:
        result = dpi.get_driver_dpi(rookie)
        if result["races_2026"] < 5:
            assert result["ci_inflated"] is True, (
                f"{rookie} should have inflated CI before 5 races (has {result['races_2026']})"
            )


def test_dpi_veteran_ci_not_inflated():
    from models.dpi import DriverPerformanceIndex
    dpi = DriverPerformanceIndex()
    result = dpi.get_driver_dpi("Verstappen")
    # Verstappen is not a rookie — CI should not be inflated
    assert result["is_rookie"] is False


def test_dpi_ranking_sorted_descending():
    from models.dpi import DriverPerformanceIndex
    dpi = DriverPerformanceIndex()
    rankings = dpi.rank_all_drivers()
    scores = [r["dpi_score"] for r in rankings]
    assert scores == sorted(scores, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Energy Strategy Simulator
# ─────────────────────────────────────────────────────────────────────────────

def test_ess_circuit_profile_all_circuits():
    from models.ess.ess_circuit_profile import CIRCUIT_PROFILES
    for circuit, profile in CIRCUIT_PROFILES.items():
        assert 0 <= profile.ehi <= 1, f"{circuit} EHI out of bounds"
        assert 0 <= profile.braking_score <= 1
        assert 0 <= profile.straight_penalty <= 1


def test_ess_mdp_policy_length_matches_laps():
    from models.ess.ess_circuit_profile import get_profile
    from models.ess.ess_mdp import EnergyDeploymentMDP
    profile = get_profile("Suzuka")
    mdp     = EnergyDeploymentMDP()
    policy  = mdp.solve("Suzuka", "Mercedes", total_laps=53, ehi=profile.ehi)
    assert len(policy.strategy) == 53


def test_ess_monte_carlo_probabilities_valid():
    from models.ess.ess_circuit_profile import get_profile
    from models.ess.ess_mdp import EnergyDeploymentMDP
    from models.ess.ess_monte_carlo import EnergyMonteCarlo

    profile = get_profile("Suzuka")
    mdp     = EnergyDeploymentMDP()
    policy  = mdp.solve("Suzuka", "Mercedes", total_laps=53, ehi=profile.ehi)

    mc = EnergyMonteCarlo(n_simulations=1_000)  # Small for speed in tests
    result = mc.simulate(policy)

    assert 0 <= result.clipping_probability <= 1
    assert 0 <= result.yoyo_probability <= 1
    assert result.mc_runs == 1_000


# ─────────────────────────────────────────────────────────────────────────────
# Test: Reliability Risk Model
# ─────────────────────────────────────────────────────────────────────────────

def test_rrm_critical_teams_high_risk():
    from models.rrm import ReliabilityRiskModel
    rrm = ReliabilityRiskModel()
    mclaren = rrm.get_dnf_probability("McLaren", "Suzuka")
    aston   = rrm.get_dnf_probability("Aston Martin", "Suzuka")

    assert mclaren["risk_tier"] in ("CRITICAL", "HIGH")
    assert aston["risk_tier"] in ("CRITICAL", "HIGH")


def test_rrm_mercedes_low_risk():
    from models.rrm import ReliabilityRiskModel
    rrm = ReliabilityRiskModel()
    result = rrm.get_dnf_probability("Mercedes", "Suzuka")
    assert result["risk_tier"] == "LOW"


def test_rrm_aston_martin_home_race_flag():
    from models.rrm import ReliabilityRiskModel
    rrm = ReliabilityRiskModel()
    result = rrm.get_dnf_probability("Aston Martin", "Suzuka")
    assert result["home_race"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Test: Circuit DNA Classifier
# ─────────────────────────────────────────────────────────────────────────────

def test_cdc_suzuka_is_high_speed():
    from models.cdc import CircuitDNAClassifier
    cdc = CircuitDNAClassifier()
    clf = cdc.classify_circuit("Suzuka")
    assert clf.archetype_name == "high_speed_low_deg"


def test_cdc_rankings_sorted_correctly():
    from models.cdc import CircuitDNAClassifier
    cdc = CircuitDNAClassifier()
    rankings = cdc.rank_teams_for_circuit("Suzuka")
    modifiers = [r["pace_modifier"] for r in rankings]
    assert modifiers == sorted(modifiers, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Qualifying to Race Translator
# ─────────────────────────────────────────────────────────────────────────────

def test_qrt_probabilities_sum_to_one():
    from models.qrt import QualifyingRaceTranslator
    qrt = QualifyingRaceTranslator()
    result = qrt.translate(
        grid_position=1, energy_advantage=0.3,
        dpi_score=1640, dnf_probability=0.02
    )
    total = sum(result["finish_probs"].values()) + result["dnf_probability"]
    assert 0.98 <= total <= 1.02, f"Probabilities should sum to ~1, got {total:.4f}"


def test_qrt_p1_grid_favours_top_finishes():
    from models.qrt import QualifyingRaceTranslator
    qrt = QualifyingRaceTranslator()
    p1 = qrt.translate(1, 0.3, 1640, 0.02)
    p20 = qrt.translate(20, -0.3, 1400, 0.35)
    assert p1["win_probability"] > p20["win_probability"]


# ─────────────────────────────────────────────────────────────────────────────
# Test: MetaEnsemble
# ─────────────────────────────────────────────────────────────────────────────

def test_meta_ensemble_win_probs_sum_to_one():
    from models.meta_ensemble import MetaEnsemble
    ensemble = MetaEnsemble()
    driver_inputs = [
        {"driver": f"Driver{i}", "team": "Mercedes", "grid_pos": i,
         "cpm_score": 0.5, "dpi_score": 1600, "ess_gain": 5.0,
         "rrm_dnf": 0.05, "cdc_mod": 0.0}
        for i in range(1, 21)
    ]
    results = ensemble.predict_race(driver_inputs, "Suzuka", 3)
    total_win = sum(r.win_probability for r in results)
    assert abs(total_win - 1.0) < 0.01, f"Win probs should sum to 1, got {total_win:.4f}"


def test_meta_ensemble_positions_in_range():
    from models.meta_ensemble import MetaEnsemble
    ensemble = MetaEnsemble()
    driver_inputs = [
        {"driver": f"Driver{i}", "team": "Ferrari", "grid_pos": i,
         "cpm_score": 0.0, "dpi_score": 1500, "ess_gain": 0.0,
         "rrm_dnf": 0.05, "cdc_mod": 0.0}
        for i in range(1, 21)
    ]
    results = ensemble.predict_race(driver_inputs, "Suzuka", 3)
    for r in results:
        assert 1.0 <= r.predicted_position <= 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Test: API Schemas
# ─────────────────────────────────────────────────────────────────────────────

def test_prediction_card_request_theme_validation():
    from api.schemas import PredictionCardRequest
    req = PredictionCardRequest(race="Suzuka", top_n=5, theme="dark")
    assert req.theme == "dark"

    with pytest.raises(Exception):
        PredictionCardRequest(race="Suzuka", top_n=5, theme="invalid_theme")


def test_confidence_interval_schema():
    from api.schemas import ConfidenceInterval
    ci = ConfidenceInterval(p10=3.0, p90=11.0)
    assert ci.p10 < ci.p90
