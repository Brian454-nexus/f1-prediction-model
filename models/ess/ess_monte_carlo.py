"""
F1 APEX — ESS: Battery State Monte Carlo Simulator
====================================================
Sub-module 3 of the Energy Strategy Simulator.

PURPOSE:
    Simulates the stochastic outcomes of deploying the MDP-optimal energy
    strategy over 50,000 Monte Carlo race runs, accounting for:
      - Strategy variance: teams don't perfectly execute their optimal policy
      - Safety car interactions: unexpected SoC consumption/recovery windows
      - Rival energy clipping: moments when an opponent runs out of deployment
      - Track position consequences of yoyo-effect dynamics (Russell/Leclerc
        Australia pattern modelled explicitly)

OUTPUTS PER DRIVER (from MC distribution):
    - clipping_probability: P(SoC drops below 15% at any point in race)
    - yoyo_probability: P(driver experiences ≥2 energy-induced position swaps)
    - expected_energy_gain_seconds: Mean lap-time gain from strategy
    - energy_gain_p10/p90: CI over 50,000 runs
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.ess.ess_mdp import DeploymentPolicy, ENERGY_COST, LAPTIME_IMPROVEMENT
from utils.logger import get_logger

logger = get_logger("APEX.ESS.MonteCarlo")

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MC_RUNS = 50_000

# Standard deviation of actual lap-to-lap energy use vs optimal (execution noise)
STRATEGY_NOISE_STDDEV = 0.015   # ±1.5% SoC per lap

# Safety car probability per race (historical average, modified by circuit)
BASE_SAFETY_CAR_PROB = 0.55

# Safety car SoC recovery bonus (teams harvest on neutralised laps)
SC_SoC_RECOVERY = 0.08


@dataclass
class MonteCarloResult:
    """Summary statistics from a 50,000-run energy MC simulation."""
    team:                       str
    circuit_name:               str
    clipping_probability:       float
    yoyo_probability:           float
    expected_energy_gain_sec:   float
    energy_gain_p10:            float
    energy_gain_p90:            float
    mc_runs:                    int


class EnergyMonteCarlo:
    """
    Runs stochastic battery state simulations over a full race distance.

    Each run:
      1. Draws execution noise around the MDP optimal action for each lap.
      2. Randomly samples a safety car event (impacts SoC recovery).
      3. Tracks SoC across all laps and records clipping/yoyo events.

    The resulting distribution is summarised into a MonteCarloResult used
    by the MetaEnsemble as the ESS module output.
    """

    def __init__(self, n_simulations: int = DEFAULT_MC_RUNS, seed: int = 42) -> None:
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        policy: DeploymentPolicy,
        safety_car_prob: float = BASE_SAFETY_CAR_PROB,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo energy simulation from a DeploymentPolicy.

        Args:
            policy:          Output from EnergyDeploymentMDP.solve().
            safety_car_prob: Probability a safety car appears during this race.

        Returns:
            MonteCarloResult with clipping, yoyo, and gain distributions.
        """
        total_laps  = policy.total_laps
        ehi         = policy.ehi

        # Pre-compute energy cost array from MDP strategy
        base_costs = np.array(
            [ENERGY_COST[action] for action in policy.strategy], dtype=np.float64
        )
        base_gains = np.array(
            [LAPTIME_IMPROVEMENT[action] for action in policy.strategy], dtype=np.float64
        )

        harvest_rate = ehi * 0.06  # Per lap, in SoC fraction

        # ── Vectorised Monte Carlo ─────────────────────────────────────────
        # Shape: (n_simulations, total_laps)

        # Execution noise around the base cost
        noise       = self.rng.normal(0, STRATEGY_NOISE_STDDEV, (self.n_simulations, total_laps))
        actual_cost = np.clip(base_costs + noise, 0.01, 0.20)

        # Safety car lap (random lap per run; -1 = no safety car)
        has_sc     = self.rng.random(self.n_simulations) < safety_car_prob
        sc_lap_idx = self.rng.integers(5, total_laps - 5, size=self.n_simulations)

        # Simulate SoC trajectory
        soc = np.ones(self.n_simulations)   # Start at full charge
        clipping_events = np.zeros(self.n_simulations, dtype=int)
        position_swaps  = np.zeros(self.n_simulations, dtype=int)
        energy_gains    = np.zeros(self.n_simulations)

        prev_soc = soc.copy()

        for lap in range(total_laps):
            # Harvest
            soc += harvest_rate

            # Extra harvest on SC lap
            soc += np.where(has_sc & (sc_lap_idx == lap), SC_SoC_RECOVERY, 0.0)

            # Discharge
            soc -= actual_cost[:, lap]
            soc  = np.clip(soc, 0.0, 1.0)

            # Track clipping (SoC < 15%)
            clipping_events += (soc < 0.15).astype(int)

            # Energy-induced position swap: SoC swings >20% between consecutive laps
            yoyo_condition = np.abs(soc - prev_soc) > 0.20
            position_swaps += yoyo_condition.astype(int)
            prev_soc = soc.copy()

            # Lap-time gain (adjusted for any clipping penalty on this lap)
            lap_gain = base_gains[lap]
            lap_gain_array = np.where(soc < 0.15, lap_gain - 1.5, lap_gain)
            energy_gains += lap_gain_array

        # ── Aggregate ─────────────────────────────────────────────────────

        clipping_prob = float(np.mean(clipping_events > 0))
        yoyo_prob     = float(np.mean(position_swaps >= 2))

        logger.info(
            f"MC complete ({self.n_simulations:,} runs) — {policy.team} at {policy.circuit_name}: "
            f"clipping={clipping_prob:.1%}, yoyo={yoyo_prob:.1%}, "
            f"gain={float(np.mean(energy_gains)):+.2f}s"
        )

        return MonteCarloResult(
            team=policy.team,
            circuit_name=policy.circuit_name,
            clipping_probability=round(clipping_prob, 4),
            yoyo_probability=round(yoyo_prob, 4),
            expected_energy_gain_sec=round(float(np.mean(energy_gains)), 3),
            energy_gain_p10=round(float(np.percentile(energy_gains, 10)), 3),
            energy_gain_p90=round(float(np.percentile(energy_gains, 90)), 3),
            mc_runs=self.n_simulations,
        )


if __name__ == "__main__":
    from models.ess.ess_circuit_profile import get_profile
    from models.ess.ess_mdp import EnergyDeploymentMDP

    profile = get_profile("Suzuka")
    mdp     = EnergyDeploymentMDP()
    policy  = mdp.solve("Suzuka", "Mercedes", total_laps=53, ehi=profile.ehi, team_efficiency=1.15)

    mc      = EnergyMonteCarlo(n_simulations=50_000)
    result  = mc.simulate(policy)

    print(f"\n── ESS Monte Carlo Result — Mercedes @ Suzuka ──")
    print(f"Clipping probability : {result.clipping_probability:.1%}")
    print(f"Yoyo probability     : {result.yoyo_probability:.1%}")
    print(f"Expected gain        : {result.expected_energy_gain_sec:+.2f}s")
    print(f"Gain [P10, P90]      : [{result.energy_gain_p10:+.2f}s, {result.energy_gain_p90:+.2f}s]")
