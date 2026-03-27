"""
F1 APEX — ESS: Energy Deployment MDP
=======================================
Sub-module 2 of the Energy Strategy Simulator.

PURPOSE:
    Models the optimal battery deployment strategy for each team on a given
    circuit using a Markov Decision Process (MDP).

    The 2026 power unit's 50/50 ICE-electric split means every team must
    decide, lap by lap, when to deploy stored energy:
      - Aggressive deployment on straights for overtaking
      - Conservative deployment to ensure completion without 'clipping'
        (battery state-of-charge hitting zero = sudden power loss)

    George Russell's comment in Australia — 'yoyoing with Leclerc due to
    battery management' — is the canonical real-world example of what this
    MDP models.

METHODOLOGY:
    Discrete MDP with:
      - States:   (lap_number, battery_state_of_charge_bucket)
      - Actions:  conservative | neutral | aggressive deployment
      - Rewards:  lap-time gain minus penalty for battery depletion risk
      - Solver:   Simple value iteration (analytic, not sampled)

    The optimal deployment policy is used by ess_monte_carlo.py to
    simulate the position consequences of energy strategy variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import sys
from pathlib import Path

# Ensure project root is on path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from utils.logger import get_logger

logger = get_logger("APEX.ESS.MDP")

# ── Deployment Actions ────────────────────────────────────────────────────────

DeploymentAction = Literal["conservative", "neutral", "aggressive"]

# Energy consumed per lap (% of full charge) per action
ENERGY_COST: dict[DeploymentAction, float] = {
    "conservative": 0.05,
    "neutral":       0.08,
    "aggressive":    0.12,
}

# Lap-time improvement per action (seconds vs neutral)
LAPTIME_IMPROVEMENT: dict[DeploymentAction, float] = {
    "conservative": -0.25,  # Slower — saving energy
    "neutral":       0.00,
    "aggressive":   +0.35,  # Faster — burning energy
}

# Penalty for battery going below 15% SoC (clipping risk region)
CLIPPING_PENALTY_SECONDS = 1.5

# ── MDP State & Value ─────────────────────────────────────────────────────────

@dataclass
class MDPState:
    lap: int
    soc_bucket: int  # 0-9 where 9 = full, 0 = empty


@dataclass
class DeploymentPolicy:
    """Optimal deployment policy per circuit per team."""
    circuit_name:   str
    team:           str
    total_laps:     int
    ehi:            float
    strategy:       list[DeploymentAction]   # One action per lap
    expected_gain:  float                    # Total expected lap-time benefit (seconds)
    clipping_risk:  float                    # Probability of running dangerously low SoC


class EnergyDeploymentMDP:
    """
    Solves the optimal energy deployment strategy for a given team-circuit pair.

    The MDP assumes the team starts each race at full charge (SoC = 1.0)
    and must complete exactly `total_laps` laps. The harvest rate per lap
    is proportional to the circuit EHI.

    Returns a DeploymentPolicy used by the Monte Carlo simulator to sample
    position outcomes under strategy variance.
    """

    # SoC is discretised into 10 buckets (0.0, 0.1, ..., 0.9, 1.0)
    SOC_BUCKETS = 10

    def __init__(self, gamma: float = 0.97) -> None:
        """
        Args:
            gamma: MDP discount factor. Closer to 1 = values future laps equally.
        """
        self.gamma = gamma

    def solve(
        self,
        circuit_name: str,
        team: str,
        total_laps: int,
        ehi: float,
        team_efficiency: float = 1.0,
    ) -> DeploymentPolicy:
        """
        Run value iteration to find the optimal deployment strategy.

        Args:
            circuit_name:     For logging and output labelling.
            team:             Constructor name (used for efficiency modifier).
            total_laps:       Number of laps in the race.
            ehi:              Energy Harvest Index (0-1) from circuit profile.
            team_efficiency:  Per-team ERS efficiency multiplier (1.0 = average).

        Returns:
            DeploymentPolicy with per-lap recommended action.
        """
        logger.info(f"Solving MDP for {team} at {circuit_name} ({total_laps} laps, EHI={ehi:.2f})")

        harvest_rate = ehi * 0.06 * team_efficiency  # SoC % recovered per lap

        # Value table: value[lap][soc_bucket]
        value     = np.zeros((total_laps + 1, self.SOC_BUCKETS + 1))
        policy    = [["neutral"] * (self.SOC_BUCKETS + 1) for _ in range(total_laps)]

        # Terminal value at lap 0 (race over) = 0
        # Backward induction from lap total_laps → 1
        for lap in range(total_laps - 1, -1, -1):
            for soc in range(self.SOC_BUCKETS + 1):
                best_value  = -np.inf
                best_action: DeploymentAction = "neutral"

                for action in ("conservative", "neutral", "aggressive"):
                    cost = ENERGY_COST[action]
                    new_soc_raw = soc / self.SOC_BUCKETS - cost + harvest_rate
                    new_soc_raw = max(0.0, min(1.0, new_soc_raw))
                    new_soc_bucket = round(new_soc_raw * self.SOC_BUCKETS)

                    reward = LAPTIME_IMPROVEMENT[action]

                    # Clipping penalty if SoC falls below 15%
                    if new_soc_bucket < 2:
                        reward -= CLIPPING_PENALTY_SECONDS

                    total_value = reward + self.gamma * value[lap + 1][new_soc_bucket]

                    if total_value > best_value:
                        best_value  = total_value
                        best_action = action

                value[lap][soc]  = best_value
                policy[lap][soc] = best_action

        # Extract strategy: simulate forward from full charge
        strategy: list[DeploymentAction] = []
        soc_bucket = self.SOC_BUCKETS  # Start at full charge
        total_gain = 0.0
        clipping_events = 0

        for lap in range(total_laps):
            action = policy[lap][soc_bucket]
            strategy.append(action)

            cost = ENERGY_COST[action]
            new_soc_raw = soc_bucket / self.SOC_BUCKETS - cost + harvest_rate
            new_soc_raw = max(0.0, min(1.0, new_soc_raw))
            soc_bucket  = round(new_soc_raw * self.SOC_BUCKETS)

            total_gain += LAPTIME_IMPROVEMENT[action]
            if soc_bucket < 2:
                clipping_events += 1

        clipping_risk = clipping_events / total_laps

        logger.info(
            f"Optimal strategy: {strategy[0]} → ... → {strategy[-1]} | "
            f"Expected gain: {total_gain:+.2f}s | Clipping risk: {clipping_risk:.1%}"
        )

        return DeploymentPolicy(
            circuit_name=circuit_name,
            team=team,
            total_laps=total_laps,
            ehi=ehi,
            strategy=strategy,
            expected_gain=round(total_gain, 3),
            clipping_risk=round(clipping_risk, 4),
        )


if __name__ == "__main__":
    from models.ess.ess_circuit_profile import get_profile

    mdp = EnergyDeploymentMDP()
    profile = get_profile("Suzuka")

    policy = mdp.solve(
        circuit_name="Suzuka",
        team="Mercedes",
        total_laps=53,
        ehi=profile.ehi,
        team_efficiency=1.15,  # Mercedes best ERS in field
    )

    print(f"\n── Suzuka Energy Strategy — Mercedes ──")
    print(f"Expected time gain : {policy.expected_gain:+.2f}s")
    print(f"Clipping risk      : {policy.clipping_risk:.1%}")
    print(f"Lap 1 action       : {policy.strategy[0]}")
    print(f"Lap 27 action      : {policy.strategy[26]}")
    print(f"Final lap action   : {policy.strategy[-1]}")
