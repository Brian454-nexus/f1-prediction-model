"""
F1 APEX — Driver Performance Index (DPI)
==========================================
Module 2 of 7 in the prediction ensemble.

PURPOSE:
    Disentangles driver performance from car performance.
    Estimates the 'value-added' a driver brings above and beyond their machinery.

METHODOLOGY:
    Elo-style rating system with temporal decay.
    - 2026 results weighted 5× more than 2025 equivalents.
    - Ratings are updated after each race using a modified Elo K-factor.
    - Qualifying vs race pace are tracked separately.

DRIVER CLASSIFICATIONS (2026-correct):
    ─────────────────────────────────────────────────────────────────────
    TRUE 2026 ROOKIE (only one):
        Lindblad — genuine first F1 season. CI inflated 2.5×. NOT penalised
        downward — uncertainty is symmetric; seeded from F2 relative pace vs
        Racing Bulls car. R1 P8 and R2 P12 now anchor his prior.

    SOPHOMORE DRIVERS (2025 graduates, now Year 2):
        Antonelli  — 2025 debut Mercedes. 2026: 3 races, 2 wins (R2 China + R3 Japan),
                     3 podiums (P2 R1, P1 R2, P1 R3). Championship leader.
                     Treat as established. Remove CI inflation. High Elo prior.
        Hadjar     — 2025 rookie → promoted to Red Bull for 2026. Podium in debut year.
        Bearman    — 2025 debut Haas. 2026: P7 and P5. Remove CI inflation.
        Bortoleto  — 2025 debut Audi. Remove CI inflation.

    RETURNING VETERANS (year-out decay, NOT rookies):
        Perez     — 1 year out (2025). 200+ F1 starts. 0.85× Elo decay for rust.
                    2026: Cadillac R1 P16, R2 P15. Car-limited, not driver-limited.
        Bottas    — 1 year out (2025) but Mercedes reserve driver throughout.
                    0.92× Elo decay (less rust, active testing). Cadillac car-limited.

PHILOSOPHY ON CI INFLATION:
    CI inflation expresses genuine uncertainty — NOT a downward performance bias.
    The CI should be symmetric around the estimated median finishing position.
    A wide-CI driver appears at their expected position with wide error bands,
    not artificially dragged toward the back of the grid.

OUTPUTS:
    - dpi_score: Float rating (reference: 1500 = midfield average)
    - ci_low / ci_high: Symmetric confidence interval bounds
    - classification: 'rookie' | 'sophomore' | 'veteran' | 'returning_veteran'
    - ci_inflated: True only for the genuine 2026 rookie (Lindblad)
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Optional

# Ensure project root is on path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from utils.logger import get_logger

logger = get_logger("APEX.DPI")

STATE_PATH = Path("models/state/dpi")

# ── ELO settings ──────────────────────────────────────────────────────────────
BASE_RATING      = 1500.0
K_FACTOR_2026    = 40.0    # High K = fast adaptation to new 2026 data
K_FACTOR_SPRINT  = 20.0    # Sprint K = half of GP (half-distance, fewer points)
K_FACTOR_PRIOR   = 8.0     # Low K for pre-2026 historical signal
DECAY_PER_RACE   = 0.005   # Each race, historical Elo fades toward base

SPRINT_FORM_WINDOW = 3     # Rolling average over last N sprint results

# ── Rookie CI inflation ───────────────────────────────────────────────────────
# ONLY applied to genuine 2026 first-season drivers.
# Inflation is symmetric — does NOT pull the driver downward.
ROOKIE_CI_MULTIPLIER = 2.5
ROOKIE_CI_RACES_THRESHOLD = 5   # Remove inflation after 5 races

# ── Returning veteran decay factors ──────────────────────────────────────────
RETURNING_VETERAN_DECAY: dict[str, float] = {
    "Perez":   0.85,   # 1 year out; significant rust
    "Bottas":  0.92,   # 1 year out but active Mercedes reserve — less rust
}

# ── Driver behavioural flags (from observed session data) ─────────────────────
# These are persisted in JSON state and consumed by QRT for position-gain calcs.
# start_pattern:     'strong' | 'neutral' | 'poor'
# racecraft_bonus:   'high' | 'neutral'
# Both are updated after each sprint/race observing actual lap-1 position changes.
DRIVER_BEHAVIORAL_FLAGS: dict[str, dict] = {
    "Russell": {
        "start_pattern":    "strong",
        # Australia GP: pole→win (P1→P1). China GP: P2→P2. Japan GP: P4→P4.
        # Consistently holds or gains position at the start.
        "racecraft_bonus":  "neutral",
    },
    "Antonelli": {
        "start_pattern":    "strong",
        # R1 Australia: P2→P2 (podium). R2 China: P1 win (1st career win).
        # R3 Japan: P1 win (back-to-back victories). Championship leader through R3.
        # No start issues observed — strong execution all three races.
        "racecraft_bonus":  "neutral",
    },
    "Lawson": {
        "start_pattern":    "neutral",
        "racecraft_bonus":  "high",
        # P7 R2 China, P9 R3 Japan — consistent mid-pack scorer.
        # Elevated position-gain probability in chaotic/SC-disrupted scenarios.
    },
    "Norris": {
        "start_pattern":    "strong",
        # P5 R1 Australia (Piastri DNS). DNS R2. P5 R3 Japan after McLaren PU fix.
        # Pace is clearly there when car is reliable.
        "racecraft_bonus":  "high",
    },
    "Verstappen": {
        "start_pattern":    "strong",
        # P6 R1 Australia, DNF R2 China (PU), P8 R3 Japan.
        # Red Bull struggling but Verstappen extracting maximum from the car.
        "racecraft_bonus":  "high",
    },
}

# ── Driver classification ─────────────────────────────────────────────────────
# "rookie": true first F1 season (2026) — CI inflated 2.5×
# "sophomore": 2025 debut driver, now in year 2 — no CI inflation
# "returning_veteran": year+ gap, decay applied — no CI inflation
# "veteran": experienced multi-season driver — standard Elo

DRIVER_CLASSIFICATIONS: dict[str, str] = {
    # ── True 2026 rookie (only one) ───────────────────────────────────
    "Lindblad":   "rookie",

    # ── Sophomore (2025 graduates, year 2) ───────────────────────────
    "Antonelli":  "sophomore",
    "Hadjar":     "sophomore",
    "Bearman":    "sophomore",
    "Bortoleto":  "sophomore",
    "Colapinto":  "sophomore",   # Alpine; 2025 Williams mid-season; first points China R2

    # ── Returning veterans ────────────────────────────────────────────
    "Perez":      "returning_veteran",
    "Bottas":     "returning_veteran",

    # ── Established veterans (2026 full grid) ─────────────────────────
    "Verstappen": "veteran",
    "Hamilton":   "veteran",
    "Norris":     "veteran",
    "Leclerc":    "veteran",
    "Piastri":    "veteran",
    "Russell":    "veteran",
    "Alonso":     "veteran",
    "Sainz":      "veteran",
    "Albon":      "veteran",
    "Gasly":      "veteran",
    "Ocon":       "veteran",
    "Stroll":     "veteran",
    "Hulkenberg": "veteran",
    "Lawson":     "veteran",     # Recovered strongly in 2025 post Red Bull demotion
}

# Convenience set: drivers in their genuine first F1 season in 2026.
# Used by tests and any module that needs to identify first-year drivers
# without iterating through DRIVER_CLASSIFICATIONS directly.
ROOKIES_2026: frozenset[str] = frozenset(
    driver for driver, cls in DRIVER_CLASSIFICATIONS.items() if cls == "rookie"
)

# ── Elo seed ratings ──────────────────────────────────────────────────────────
# Sources:
#   Veterans:        2022–2025 career Elo proxy (qualifying + race deltas)
#   Sophomores:      2025 debut performance, then updated by 2026 R1+R2 results
#   Returning vets:  Career Elo × decay factor
#   Lindblad (rookie): Seeded from F2 pace relative to Racing Bulls machinery
#                      R1 P8 and R2 P12 anchor the prior upward from bare minimum
# ── Car-context notes ────────────────────────────────────────────────────────
# These drivers' 2026 race results are SIGNIFICANTLY car-limited.
# DPI calibration must NOT penalise Elo on mechanical DNF/DNS events.
# The note is persisted in the JSON state and checked by AutoCalibrator.
CAR_LIMITED_CONTEXT: dict[str, str] = {
    "Norris":  "McLaren PU crisis: DNS R2. P5 R1, P5 R3 when car ran. "
               "Honda fix confirmed by R3; DNF/DNS events do not update Elo negatively.",
    "Piastri": "McLaren PU crisis: DNS R1 and DNS R2. P2 R3 Japan after PU fix. "
               "Pre-fix DNS events do not update Elo negatively.",
    "Alonso":  "Aston Martin structural underperformance: DNF R1 and DNF R2. "
               "All failures confirmed mechanical. Elo protected.",
    "Stroll":  "Aston Martin structural underperformance: DNF R2 and DNF R3. "
               "Car-dependent driver; DNFs not driver-attributed.",
}

PRIOR_RATINGS: dict[str, float] = {
    # ── Veterans — updated through R3 (Japan) ──────────────────────
    # Ratings reflect cumulative 2026 performance plus multi-year career Elo.
    "Norris":      1800.0,   # P5 R1, DNS R2, P5 R3; career pace unquestioned
    "Alonso":      1755.0,   # Aston Martin reliability issues; no Elo gain R1-R3
    "Leclerc":     1790.0,   # P3 R1, P4 R2, P3 R3 — consistently third-fastest
    "Hamilton":    1755.0,   # P4 R1, P3 R2, P6 R3 — 41pts, adapting to Ferrari well
    "Piastri":     1720.0,   # P2 R3 Japan shows true pace; DNS R1+R2 car-limited
    "Verstappen":  1800.0,   # P6 R1, DNF R2 (PU), P8 R3 — car limited but racecraft intact
    "Russell":     1665.0,   # Win R1 Australia; P2 R2, P4 R3 — 63pts, strong season
    "Sainz":       1710.0,   # Strong tyre management; Williams consistent points scorer
    "Hulkenberg":  1638.0,   # Audi car-limited; DNS R1 (Hulkenberg) then improving
    "Albon":       1668.0,   # Williams pace extraction; above-car results
    "Lawson":      1628.0,   # P7 R2, P9 R3 — consistent Racing Bulls scorer
    "Gasly":       1652.0,   # P10 R1, P6 R2, P7 R3 — reliable points scorer
    "Ocon":        1612.0,   # P10 R3; solid Haas team-mate
    "Stroll":      1478.0,   # Car-dependent; AM reliability crisis ongoing

    # ── Returning veterans — career Elo × decay factor ────────────────
    "Perez":       1530.0 * RETURNING_VETERAN_DECAY["Perez"],   # 0.85× ≈ 1301
    "Bottas":      1560.0 * RETURNING_VETERAN_DECAY["Bottas"],  # 0.92× ≈ 1435

    # ── Sophomores — 2025 debut, updated by 2026 R1-R3 ───────────────
    "Antonelli":   1725.0,   # P2 R1, Win R2 (1st career win), Win R3 — championship leader
    "Hadjar":      1530.0,   # Red Bull promotion; DNF R1, improving thereafter
    "Bearman":     1542.0,   # R1 P7, R2 P5, DNF R3; Haas overperformance confirmed
    "Bortoleto":   1508.0,   # P9 R1; Audi car-limited; strong qualifying relative to car
    "Colapinto":   1530.0,   # P10 R2; Alpine; building consistency

    # ── True 2026 rookie ─────────────────────────────────────────────
    # R1 P8 confirmed. Wide symmetric CI maintained (races_2026 < 5 threshold not cleared).
    "Lindblad":    1528.0,
}

# ── Within-team baseline pairs 2026 (full grid) ──────────────────────────────
TEAMMATE_PAIRS_2026: dict[str, list[str]] = {
    "Mercedes":     ["Russell",    "Antonelli"],
    "Ferrari":      ["Leclerc",    "Hamilton"],
    "McLaren":      ["Norris",     "Piastri"],
    "Red Bull":     ["Verstappen", "Hadjar"],
    "Aston Martin": ["Alonso",     "Stroll"],
    "Williams":     ["Albon",      "Sainz"],
    "Racing Bulls": ["Lawson",     "Lindblad"],
    "Alpine":       ["Gasly",      "Colapinto"],
    "Haas":         ["Bearman",    "Ocon"],
    "Cadillac":     ["Perez",      "Bottas"],
    "Audi":         ["Hulkenberg", "Bortoleto"],
}


class DriverPerformanceIndex:
    """
    Maintains and updates per-driver Elo-style ratings.

    Key design decisions:
    1. Only Lindblad receives CI inflation in 2026 — the sole genuine first-season driver.
    2. Antonelli, Hadjar, Bearman, Bortoleto are sophomores: normal CI, full Elo updates.
    3. Perez and Bottas receive career-Elo × decay priors; modelled as car-limited at Cadillac.
    4. CI inflation is strictly symmetric — it does NOT pull predicted positions downward.
    """

    def __init__(self) -> None:
        STATE_PATH.mkdir(parents=True, exist_ok=True)
        self.state_file = STATE_PATH / "dpi_ratings.json"
        self._ratings: dict = self._load_state()

    # ── Public API ─────────────────────────────────────────────────────────

    def get_driver_dpi(self, driver: str) -> dict:
        """
        Return current DPI rating and confidence interval for a driver.

        CI inflation is ONLY applied to Lindblad (true 2026 rookie, < 5 races).
        The inflation is symmetric — equal bands above and below the median.
        """
        if driver not in self._ratings:
            logger.warning(f"Unknown driver {driver!r} — using base rating")
            return self._unknown_entry(driver)

        entry = self._ratings[driver]
        rating  = entry["rating"]
        stddev  = entry["stddev"]
        classification = DRIVER_CLASSIFICATIONS.get(driver, "veteran")
        races_done = entry.get("races_2026", 0)

        # Inflate CI ONLY for genuine 2026 rookie with < 5 races completed
        is_rookie = (classification == "rookie")
        ci_inflated = is_rookie and (races_done < ROOKIE_CI_RACES_THRESHOLD)
        ci_mult = ROOKIE_CI_MULTIPLIER if ci_inflated else 1.0

        half_ci = 1.28 * stddev * ci_mult   # 80% interval
        # Symmetric: ci_low and ci_high equally spaced around rating
        ci_low  = round(rating - half_ci, 1)
        ci_high = round(rating + half_ci, 1)

        return {
            "driver":          driver,
            "dpi_score":       round(rating, 1),
            "ci_low":          ci_low,
            "ci_high":         ci_high,
            "classification":  classification,
            "is_rookie":       is_rookie,
            "ci_inflated":     ci_inflated,
            "races_2026":      races_done,
            "rating_quality":  "reliable" if races_done >= ROOKIE_CI_RACES_THRESHOLD else "early",
        }

    def update_after_race(self, race_results: list[dict], round_id: int) -> None:
        """
        Update all driver ratings given a completed race result.

        Args:
            race_results: List of {driver, team, finish_pos, qualifying_pos, dnf}
            round_id:     Race round number.

        DNF handling: If DNF is caused by car (power unit, hydraulics etc), the
        driver's Elo is NOT penalised. Only driver-error DNFs update Elo negatively.
        """
        logger.info(f"Updating DPI ratings after Round {round_id}")
        total_drivers = len(race_results)

        for result in race_results:
            driver = result["driver"]
            if driver not in self._ratings:
                self._ratings[driver] = self._new_entry(driver)

            entry = self._ratings[driver]

            # DNF check — only penalise driver-fault DNFs
            dnf = result.get("dnf", False)
            dnf_cause = result.get("cause", "").lower()
            driver_fault = dnf and any(
                kw in dnf_cause for kw in ["crash", "spin", "collision", "driver"]
            )

            if dnf and not driver_fault:
                # Car DNF (PU, hydraulics, etc) — burn race counter, don't penalise
                entry["races_2026"] = entry.get("races_2026", 0) + 1
                logger.debug(f"  {driver}: Car DNF ({dnf_cause}) — Elo unchanged")
                continue

            if dnf and driver_fault:
                # Driver-at-fault DNF — treat as last place
                finish_pos = total_drivers
            else:
                finish_pos = result["finish_pos"]

            expected = self._expected_finish(entry["rating"], total_drivers)
            actual   = total_drivers - finish_pos + 1   # Inverted score (P1 = highest)
            delta    = K_FACTOR_2026 * (actual - expected)

            entry["rating"] = max(900.0, entry["rating"] + delta)
            entry["races_2026"] = entry.get("races_2026", 0) + 1

            # Narrow stddev as observations accumulate (more races → more confidence)
            entry["stddev"] = max(18.0, entry["stddev"] * 0.95)

            logger.debug(
                f"  {driver}: P{finish_pos} | Δ={delta:+.1f} | "
                f"Rating now {entry['rating']:.0f}"
            )

        self._save_state()
        logger.info(f"DPI ratings updated and saved to {self.state_file}")

    def get_teammate_rating_gap(self, team: str) -> Optional[float]:
        """
        Rating gap between the two listed teammates of a team.
        Positive = driver-1 is rated higher.
        """
        pair = TEAMMATE_PAIRS_2026.get(team)
        if not pair or len(pair) < 2:
            return None
        r1 = self._ratings.get(pair[0], {}).get("rating", BASE_RATING)
        r2 = self._ratings.get(pair[1], {}).get("rating", BASE_RATING)
        return round(r1 - r2, 1)

    def rank_all_drivers(self) -> list[dict]:
        """Return all drivers sorted by DPI score (highest first)."""
        return sorted(
            [self.get_driver_dpi(d) for d in self._ratings],
            key=lambda x: -x["dpi_score"],
        )

    def update_after_sprint(self, sprint_results: list[dict], round_id: int) -> None:
        """
        Apply Elo updates from a sprint race result.

        Sprint K-factor = 20 (half of full GP K=40), reflecting:
          - Sprint is half-distance
          - Sprint awards fewer championship points
          - Sprint is less representative of full race pace

        Technical DNFs (same as GP): Elo is NOT updated. Race counter incremented.
        Driver-fault DNFs: treated as last place (same as GP).

        Args:
            sprint_results: List of {driver, team, finish_pos, dnf, cause}
            round_id:       Race round number (used for logging).
        """
        logger.info(f"Updating DPI ratings after Round {round_id} SPRINT")
        total_drivers = len([r for r in sprint_results if not r.get("dns", False)])

        for result in sprint_results:
            driver = result["driver"]
            if result.get("dns", False):
                continue

            if driver not in self._ratings:
                self._ratings[driver] = self._new_entry(driver)

            entry = self._ratings[driver]

            # DNF classification: technical vs driver fault (same logic as GP)
            dnf       = result.get("dnf", False)
            dnf_cause = result.get("cause", "").lower()
            driver_fault = dnf and any(
                kw in dnf_cause for kw in ["crash", "spin", "collision", "driver"]
            )

            if dnf and not driver_fault:
                # Technical DNF — increment count, no Elo change
                entry["races_2026"] = entry.get("races_2026", 0) + 1
                # Still record the DNF in sprint_history for sprint_form calc
                history = entry.setdefault("sprint_history", [])
                history.append(None)   # None = DNF/excluded
                entry["sprint_history"] = history[-SPRINT_FORM_WINDOW:]
                logger.debug(f"  {driver}: Sprint technical DNF — Elo unchanged")
                continue

            finish_pos = total_drivers if (dnf and driver_fault) else result["finish_pos"]

            expected = self._expected_finish(entry["rating"], total_drivers)
            actual   = total_drivers - finish_pos + 1
            delta    = K_FACTOR_SPRINT * (actual - expected)  # K=20 for sprint

            entry["rating"]    = max(900.0, entry["rating"] + delta)
            entry["races_2026"] = entry.get("races_2026", 0) + 1
            entry["stddev"]    = max(18.0, entry.get("stddev", 50.0) * 0.97)

            # Update rolling sprint_history (window = SPRINT_FORM_WINDOW)
            history = entry.setdefault("sprint_history", [])
            history.append(finish_pos)
            entry["sprint_history"] = history[-SPRINT_FORM_WINDOW:]

            logger.debug(
                f"  {driver}: Sprint P{finish_pos} | K=20 | Δ={delta:+.1f} | "
                f"Rating now {entry['rating']:.0f}"
            )

        self._save_state()
        logger.info(f"DPI sprint ratings updated — {self.state_file}")

    def get_sprint_form(self, driver: str) -> Optional[float]:
        """
        Return rolling average finish position from last SPRINT_FORM_WINDOW sprints.
        None values (technical DNFs) are excluded from the average.
        Returns None if no sprint history exists yet.
        """
        entry = self._ratings.get(driver)
        if not entry:
            return None
        history = [p for p in entry.get("sprint_history", []) if p is not None]
        if not history:
            return None
        return round(sum(history) / len(history), 2)

    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded DPI ratings from {self.state_file}")
            # Merge any new behavioral flags into existing state
            # (handles upgrades when DRIVER_BEHAVIORAL_FLAGS extends post-deploy)
            for driver, flags in DRIVER_BEHAVIORAL_FLAGS.items():
                if driver in data:
                    for key, val in flags.items():
                        if key not in data[driver]:
                            data[driver][key] = val
                            logger.debug(f"Merged flag {key}={val!r} for {driver}")
            return data

        logger.info("Initialising DPI ratings from 2026-corrected priors")
        initial_state = self._build_initial_state()
        # Persist immediately — GCS sync and pipeline resumability require the
        # checkpoint file to exist after first initialisation.
        with open(self.state_file, "w") as f:
            json.dump(initial_state, f, indent=2)
        logger.info(f"Initial DPI state saved to {self.state_file}")
        return initial_state

    def _save_state(self) -> None:
        with open(self.state_file, "w") as f:
            json.dump(self._ratings, f, indent=2)

    def _build_initial_state(self) -> dict:
        """
        Build the initial state from PRIOR_RATINGS with classification-aware stddevs.
        Car-context notes are persisted for drivers flagged in CAR_LIMITED_CONTEXT
        so AutoCalibrator can reference them when deciding Elo penalty exclusions.

        Seeded at races_2026=3 (through Japan R3) for all drivers.
        """
        state = {}
        for driver, rating in PRIOR_RATINGS.items():
            classification = DRIVER_CLASSIFICATIONS.get(driver, "veteran")

            if classification == "rookie":
                # Lindblad only. stddev = 80 here, CI multiplied by 2.5× in get_driver_dpi.
                # Wide bands are symmetric — not a downward drag.
                stddev = 70.0    # Moderately narrowed after 3 races of data
                races_2026 = 3

            elif classification == "sophomore":
                stddev = 50.0    # Narrowed from 60 — 3 races observed
                if driver == "Antonelli":
                    stddev = 32.0    # 2 wins + 3 podiums; championship leader, highest sophomore confidence
                elif driver == "Colapinto":
                    stddev = 55.0    # Still building consistency
                races_2026 = 3

            elif classification == "returning_veteran":
                stddev = 60.0        # Decay baked into rating; std reflects rust
                races_2026 = 3

            else:
                # Veteran: most historical data → tightest stddev
                stddev = 45.0    # Moderately narrowed after 3 races
                races_2026 = 3

            entry: dict = {
                "rating":         round(rating, 1),
                "stddev":         stddev,
                "races_2026":     races_2026,
                "classification": classification,
                "sprint_history": [],   # Populated incrementally by update_after_sprint
            }

            # Attach car-context note for mechanically-limited drivers
            if driver in CAR_LIMITED_CONTEXT:
                entry["car_context_note"] = CAR_LIMITED_CONTEXT[driver]

            # Seed observed behavioural flags from session data
            if driver in DRIVER_BEHAVIORAL_FLAGS:
                for key, val in DRIVER_BEHAVIORAL_FLAGS[driver].items():
                    entry[key] = val

            state[driver] = entry

        return state

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _expected_finish(rating: float, total_drivers: int) -> float:
        """Logistic expected score mapped to [1, total_drivers]."""
        exp_prob = 1 / (1 + 10 ** ((BASE_RATING - rating) / 400))
        return total_drivers * exp_prob

    @staticmethod
    def _new_entry(driver: str) -> dict:
        classification = DRIVER_CLASSIFICATIONS.get(driver, "veteran")
        return {
            "rating":         BASE_RATING,
            "stddev":         80.0 if classification == "rookie" else 60.0,
            "races_2026":     0,
            "classification": classification,
        }

    @staticmethod
    def _unknown_entry(driver: str) -> dict:
        return {
            "driver":         driver,
            "dpi_score":      BASE_RATING,
            "ci_low":         BASE_RATING - 200,
            "ci_high":        BASE_RATING + 200,
            "classification": "unknown",
            "is_rookie":      False,
            "ci_inflated":    True,
            "races_2026":     0,
            "rating_quality": "no_data",
        }


# ── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dpi = DriverPerformanceIndex()
    car_flags = set(CAR_LIMITED_CONTEXT.keys())

    print("\n--- DPI 2026 Full Grid Rankings (22 drivers) ---")
    print(f"{'Pos':<4} {'Driver':<14} {'Rating':>6}  {'CI':<20}  {'Class':<20}  {'Flags'}")
    print("-" * 100)
    for pos, entry in enumerate(dpi.rank_all_drivers(), 1):
        flags = []
        if entry["ci_inflated"]:
            flags.append("WIDE-CI")
        if entry["classification"] == "returning_veteran":
            decay = RETURNING_VETERAN_DECAY.get(entry["driver"], 1.0)
            flags.append(f"DECAY-x{decay}")
        if entry["classification"] == "sophomore":
            flags.append("SOPHOMORE")
        if entry["driver"] in car_flags:
            flags.append("CAR-LIMITED")
        flag_str = " | ".join(flags) if flags else ""
        print(
            f"  {pos:<3} {entry['driver']:<14} {entry['dpi_score']:>6.0f}  "
            f"[{entry['ci_low']:.0f}, {entry['ci_high']:.0f}]  "
            f"{entry['classification']:<20}  {flag_str}"
        )
    print(f"\nTotal drivers in state: {len(dpi._ratings)}")
