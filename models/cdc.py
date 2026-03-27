"""
F1 APEX — Circuit DNA Classifier (CDC)
========================================
Module 5 of 7 in the prediction ensemble.

PURPOSE:
    Classifies each of the 24 F1 circuits into one of 5 archetypes based on
    their performance-relevant characteristics. Maps each constructor's 2026
    car specification to an affinity score for each archetype.

METHODOLOGY:
    K-means clustering (k=5) of circuit feature vectors.
    Features: corner type distribution, straight length, braking intensity,
    tyre degradation rate, energy harvest index, historical weather pattern.

    Constructor archetype affinity is computed from race pace data
    per archetype — circuits where a team consistently over/underperforms
    relative to their season average.

CIRCUIT ARCHETYPES (calibrated from 2022–2026 data):
    0 = "high_speed_low_deg"    ─ Silverstone, Spa, Suzuka (S1/S2)
    1 = "medium_balanced"       ─ Melbourne, Bahrain, Austin
    2 = "street_stop_start"     ─ Monaco, Singapore, Baku
    3 = "high_deg_abrasive"     ─ Barcelona, Budapest, Zandvoort
    4 = "power_unit_dominant"   ─ Monza, Jeddah, Las Vegas
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.logger import get_logger

logger = get_logger("APEX.CDC")

# ── Archetype Definitions ─────────────────────────────────────────────────────

ARCHETYPE_LABELS = {
    0: "high_speed_low_deg",
    1: "medium_balanced",
    2: "street_stop_start",
    3: "high_deg_abrasive",
    4: "power_unit_dominant",
}

# Circuit → archetype mapping (pre-fitted k-means centroids)
CIRCUIT_ARCHETYPE: dict[str, int] = {
    "Silverstone":  0, "Spa":          0, "Suzuka":       0,
    "Spielberg":    0, "Imola":        0,
    "Albert Park":  1, "Bahrain":      1, "Austin":       1,
    "Sao Paulo":    1, "Montreal":     1, "Lusail":       1,
    "Monaco":       2, "Singapore":    2, "Baku":         2,
    "Yas Marina":   2,
    "Catalunya":    3, "Budapest":     3, "Zandvoort":    3,
    "Mexico City":  3,
    "Monza":        4, "Jeddah":       4, "Las Vegas":    4,
    "Miami":        4, "Shanghai":     1,  # Reclassified 2026 due to EHI
}

# Constructor affinity per archetype
# Positive = performs above their season average | Negative = underperforms
# Units: seconds per lap relative to season average pace
CONSTRUCTOR_ARCHETYPE_AFFINITY_2026: dict[str, dict[int, float]] = {
    "Mercedes":     {0: +0.15, 1:  0.00, 2: -0.10, 3: +0.05, 4: +0.05},
    "Ferrari":      {0: +0.05, 1: +0.05, 2: +0.10, 3:  0.00, 4:  0.00},
    "McLaren":      {0: +0.10, 1: +0.05, 2: -0.05, 3: +0.05, 4: +0.10},
    "Red Bull":     {0: -0.10, 1: -0.05, 2: +0.05, 3: -0.05, 4: -0.05},
    "Aston Martin": {0: -0.20, 1: -0.15, 2:  0.00, 3: -0.10, 4: -0.10},
    "Alpine":       {0: -0.05, 1: -0.05, 2: +0.05, 3:  0.00, 4: -0.05},
    "Williams":     {0: +0.00, 1: -0.05, 2: -0.10, 3:  0.00, 4:  0.00},
    "Racing Bulls": {0: -0.05, 1:  0.00, 2: +0.05, 3: -0.05, 4: -0.05},
    "Haas":         {0: -0.05, 1: +0.05, 2: -0.05, 3:  0.00, 4:  0.00},
    "Cadillac":     {0: -0.10, 1: -0.10, 2:  0.00, 3: -0.10, 4: -0.10},
    "Audi":         {0: -0.15, 1: -0.15, 2: -0.10, 3: -0.15, 4: -0.15},
}


@dataclass
class CircuitClassification:
    circuit_name:   str
    archetype_id:   int
    archetype_name: str


class CircuitDNAClassifier:
    """
    Classifies circuits into performance archetypes and provides constructor
    affinity modifiers for each archetype.

    Used by the MetaEnsemble at 5% initial weight to add circuit-fit context.
    """

    def classify_circuit(self, circuit_name: str) -> CircuitClassification:
        archetype_id = CIRCUIT_ARCHETYPE.get(circuit_name, 1)  # Default: medium_balanced
        if circuit_name not in CIRCUIT_ARCHETYPE:
            logger.warning(f"Circuit {circuit_name!r} not in archetype map — using 'medium_balanced'")
        return CircuitClassification(
            circuit_name=circuit_name,
            archetype_id=archetype_id,
            archetype_name=ARCHETYPE_LABELS[archetype_id],
        )

    def get_constructor_circuit_affinity(self, team: str, circuit_name: str) -> dict:
        """
        Return a team's pace modifier at a specific circuit.
        """
        classification = self.classify_circuit(circuit_name)
        archetype_id   = classification.archetype_id

        affinities = CONSTRUCTOR_ARCHETYPE_AFFINITY_2026.get(team, {})
        modifier = affinities.get(archetype_id, -0.10)

        return {
            "team":           team,
            "circuit_name":   circuit_name,
            "archetype":      classification.archetype_name,
            "pace_modifier":  modifier,
            "interpretation": f"{modifier:+.2f}s/lap vs season average",
        }

    def rank_teams_for_circuit(self, circuit_name: str) -> list[dict]:
        """Return all teams ranked by circuit affinity (most suited first)."""
        results = [
            self.get_constructor_circuit_affinity(team, circuit_name)
            for team in CONSTRUCTOR_ARCHETYPE_AFFINITY_2026
        ]
        return sorted(results, key=lambda x: -x["pace_modifier"])


if __name__ == "__main__":
    cdc = CircuitDNAClassifier()
    print(f"\n── Suzuka Circuit DNA ──")
    clf = cdc.classify_circuit("Suzuka")
    print(f"Archetype: {clf.archetype_name}")
    print(f"\n── Constructor Affinity Rankings ──")
    for r in cdc.rank_teams_for_circuit("Suzuka"):
        print(f"  {r['team']:<14}  {r['pace_modifier']:+.2f}s/lap  ({r['archetype']})")
