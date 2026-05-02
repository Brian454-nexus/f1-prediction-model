"""
F1 APEX — ESS: Circuit Energy Profile
========================================
Sub-module 1 of the Energy Strategy Simulator (ESS, Module 3).

PURPOSE:
    Provides the Energy Harvest Index (EHI) for each circuit —
    a 0-to-1 score representing how much electrical energy can be recovered
    per lap through regenerative braking and KERS.

    High EHI circuits (Monaco, Shanghai) offer many heavy-braking zones
    where the MGU-K captures kinetic energy. Low EHI circuits (Monza, Jeddah)
    are fast-flowing with minimal braking, leaving batteries drained.

METHODOLOGY:
    EHI is computed from three sub-scores:
      1. Braking Intensity Score  ─ number × severity of braking zones
      2. Corner Energy Score      ─ energy recoverable through medium-speed corners
      3. Straight Penalty         ─ penalises circuits where regen can't keep up with discharge

    In v1.0 these are statically calibrated from reference data.
    In v2.0 they will auto-update from FastF1 telemetry speed deltas.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger("APEX.ESS.CircuitProfile")


# ── Circuit Profile Data ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class CircuitEnergyProfile:
    """Immutable energy profile for a specific circuit."""
    circuit_name:    str
    ehi:             float   # Energy Harvest Index 0-1
    braking_score:   float   # 0-1: intensity of braking zones
    corner_score:    float   # 0-1: energy recovery from corners
    straight_penalty: float  # 0-1: energy loss on straights (discharge)
    notes:           str     = ""

    @property
    def net_energy_balance(self) -> float:
        """Positive = net energy surplus per lap; negative = net drain."""
        return round(self.braking_score * 0.5 + self.corner_score * 0.3
                     - self.straight_penalty * 0.2, 4)


# Pre-calibrated profiles for all 2026 circuits
CIRCUIT_PROFILES: dict[str, CircuitEnergyProfile] = {
    "Albert Park": CircuitEnergyProfile(
        "Albert Park", ehi=0.55, braking_score=0.50, corner_score=0.55, straight_penalty=0.30,
        notes="Mixed layout. Turns 1-3 braking moderate. Safety car common."
    ),
    "Shanghai": CircuitEnergyProfile(
        "Shanghai", ehi=0.85, braking_score=0.80, corner_score=0.65, straight_penalty=0.45,
        notes="Long back straight + heavy T6 hairpin = maximum regen. Yoyo effect documented R2."
    ),
    "Suzuka": CircuitEnergyProfile(
        "Suzuka", ehi=0.65, braking_score=0.60, corner_score=0.70, straight_penalty=0.40,
        notes="S1 high-speed (low regen). Chicane + hairpin = moderate regen. Spoon harvesting good."
    ),
    "Bahrain": CircuitEnergyProfile(
        "Bahrain", ehi=0.70, braking_score=0.75, corner_score=0.55, straight_penalty=0.45,
        notes="Heavy T1 braking. Long straights discharge battery on DRS zones."
    ),
    "Jeddah": CircuitEnergyProfile(
        "Jeddah", ehi=0.40, braking_score=0.30, corner_score=0.40, straight_penalty=0.60,
        notes="High-speed street. Minimal braking. Net energy drain circuit."
    ),
    "Miami": CircuitEnergyProfile(
        "Miami", ehi=0.60, braking_score=0.55, corner_score=0.50, straight_penalty=0.40,
        notes="Semi-permanent circuit. Heavy T1 + T11 braking zones deliver moderate regen. "
              "Back straight (T10-T11) significant discharge. S2 technical complex aids harvest. "
              "Net profile: balanced — above Jeddah, below Bahrain."
    ),
    "Imola": CircuitEnergyProfile(
        "Imola", ehi=0.55, braking_score=0.55, corner_score=0.50, straight_penalty=0.35,
        notes="Tamburello + Acque Minerale = good regen. Shorter lap = more frequent cycles."
    ),
    "Monaco": CircuitEnergyProfile(
        "Monaco", ehi=0.90, braking_score=0.90, corner_score=0.75, straight_penalty=0.10,
        notes="Maximum regen. Almost no straights. Slowest circuit = continuous braking."
    ),
    "Catalunya": CircuitEnergyProfile(
        "Catalunya", ehi=0.60, braking_score=0.55, corner_score=0.55, straight_penalty=0.40,
    ),
    "Montreal": CircuitEnergyProfile(
        "Montreal", ehi=0.70, braking_score=0.75, corner_score=0.50, straight_penalty=0.45,
        notes="Hairpin at end of main straight = maximum braking regen."
    ),
    "Spielberg": CircuitEnergyProfile(
        "Spielberg", ehi=0.65, braking_score=0.65, corner_score=0.60, straight_penalty=0.35,
    ),
    "Silverstone": CircuitEnergyProfile(
        "Silverstone", ehi=0.50, braking_score=0.45, corner_score=0.65, straight_penalty=0.35,
        notes="High-speed flowing corners. Limited braking = low regen."
    ),
    "Budapest": CircuitEnergyProfile(
        "Budapest", ehi=0.75, braking_score=0.70, corner_score=0.70, straight_penalty=0.30,
        notes="Heavy T1 + many slow corners. Good harvesting throughput."
    ),
    "Spa": CircuitEnergyProfile(
        "Spa", ehi=0.45, braking_score=0.40, corner_score=0.50, straight_penalty=0.50,
        notes="Kemmel straight = significant discharge. Raidillon recovery limited."
    ),
    "Zandvoort": CircuitEnergyProfile(
        "Zandvoort", ehi=0.55, braking_score=0.55, corner_score=0.60, straight_penalty=0.35,
    ),
    "Monza": CircuitEnergyProfile(
        "Monza", ehi=0.35, braking_score=0.45, corner_score=0.20, straight_penalty=0.70,
        notes="Temple of speed. Minimal regen. Pure ICE circuit. Lowest EHI on calendar."
    ),
    "Baku": CircuitEnergyProfile(
        "Baku", ehi=0.60, braking_score=0.60, corner_score=0.50, straight_penalty=0.50,
        notes="Castle section creates regen. Long main straight discharges heavily."
    ),
    "Singapore": CircuitEnergyProfile(
        "Singapore", ehi=0.80, braking_score=0.75, corner_score=0.75, straight_penalty=0.20,
        notes="Street circuit with no fast sections. Continuous braking zones."
    ),
    "Austin": CircuitEnergyProfile(
        "Austin", ehi=0.60, braking_score=0.60, corner_score=0.55, straight_penalty=0.40,
    ),
    "Mexico City": CircuitEnergyProfile(
        "Mexico City", ehi=0.65, braking_score=0.65, corner_score=0.55, straight_penalty=0.40,
        notes="Altitude affects combustion; ERS relatively more powerful here."
    ),
    "Sao Paulo": CircuitEnergyProfile(
        "Sao Paulo", ehi=0.70, braking_score=0.70, corner_score=0.60, straight_penalty=0.40,
    ),
    "Las Vegas": CircuitEnergyProfile(
        "Las Vegas", ehi=0.50, braking_score=0.50, corner_score=0.40, straight_penalty=0.50,
        notes="Long straights on the strip. Mixed regen profile."
    ),
    "Lusail": CircuitEnergyProfile(
        "Lusail", ehi=0.65, braking_score=0.65, corner_score=0.60, straight_penalty=0.40,
    ),
    "Yas Marina": CircuitEnergyProfile(
        "Yas Marina", ehi=0.55, braking_score=0.55, corner_score=0.55, straight_penalty=0.45,
    ),
}


def get_profile(circuit_name: str) -> CircuitEnergyProfile:
    """
    Return the energy profile for a named circuit.
    Falls back to a median profile if circuit not found.
    """
    if circuit_name in CIRCUIT_PROFILES:
        return CIRCUIT_PROFILES[circuit_name]
    logger.warning(f"Circuit {circuit_name!r} not in profile table — using median profile")
    return CircuitEnergyProfile(
        circuit_name=circuit_name,
        ehi=0.58,
        braking_score=0.55,
        corner_score=0.55,
        straight_penalty=0.40,
        notes="Fallback median profile"
    )


if __name__ == "__main__":
    profile = get_profile("Suzuka")
    print(f"Suzuka EHI: {profile.ehi}")
    print(f"Net energy balance: {profile.net_energy_balance:+.4f}")
    print(f"Notes: {profile.notes}")
