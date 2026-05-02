"""
F1 APEX — Feature Engineering
================================
Computes the six 2026-specific predictive features used across all ML modules.
All features are derived from the Parquet-serialized lap data produced by
`data_pipeline.py` and applied in-memory before being written back to a
processed Parquet file.

2026 Feature Definitions (from PRD §5.2):
──────────────────────────────────────────
BER  Battery Efficiency Ratio        ─ proxy for electrical deployment efficiency
EHI  Energy Harvest Index            ─ per-circuit energy recovery score (static table)
RHS  Reliability Hazard Score        ─ per-constructor rolling DNF/DNS probability
NED  New Era Delta                   ─ performance swing 2025 final → 2026 current
TGI  Teammate Gap Index              ─ delta between teammates (car vs driver isolation)
AASP Active Aero Setting Proxy       ─ inferred downforce from speed-trap vs sector time

All per-session features are added as new columns on the lap DataFrame so
that the meta-ensemble can join any combination of them.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("APEX.FeatureEngineering")


# ── Static tables ─────────────────────────────────────────────────────────────

# Energy Harvest Index per circuit (0–1 scale, higher = more regen opportunity).
# Derived from braking zone intensity and corner count.
CIRCUIT_EHI: dict[str, float] = {
    "Albert Park":  0.55,
    "Shanghai":     0.85,   # Long straights + hairpin braking
    "Suzuka":       0.65,   # Chicane + spoon
    "Bahrain":      0.70,
    "Jeddah":       0.40,   # High-speed street circuit, limited braking
    "Miami":        0.60,
    "Imola":        0.55,
    "Monaco":       0.90,   # Constant braking — maximum regen
    "Catalunya":    0.60,
    "Montreal":     0.70,
    "Spielberg":    0.65,
    "Silverstone":  0.50,   # Fast flowing — less regen
    "Budapest":     0.75,
    "Spa":          0.45,
    "Zandvoort":    0.55,
    "Monza":        0.35,   # Ultra-high-speed — minimal braking
    "Baku":         0.60,
    "Singapore":    0.80,
    "Austin":       0.60,
    "Mexico City":  0.65,
    "Sao Paulo":    0.70,
    "Las Vegas":    0.50,
    "Lusail":       0.65,
    "Yas Marina":   0.55,
}

# 2025 final WCC standings (points as proxy for baseline performance pre-2026)
STANDINGS_2025_WCC: dict[str, float] = {
    "McLaren":      666.0,
    "Ferrari":      584.0,
    "Red Bull":     589.0,
    "Mercedes":     468.0,
    "Aston Martin": 94.0,
    "Alpine":       65.0,
    "Haas":         58.0,
    "Racing Bulls": 46.0,
    "Williams":     17.0,
    "Sauber":       4.0,
}

# 2026 constructor race pace rank through R3 Japan (lower = quicker; 1 = fastest)
# Updated from R1-R3 classified finishing performance and sector time analysis.
# Bahrain and Saudi Arabia cancelled — season reduced to 22 races, 3 completed.
PACE_RANK_2026_R3: dict[str, float] = {
    "Mercedes":     1.0,
    "Ferrari":      2.0,
    "McLaren":      3.0,   # Both cars finished R3 Japan; true pace emerging post-PU fix
    "Haas":         4.0,   # Bearman P7 R1, P5 R2 — consistently above expectation
    "Racing Bulls": 5.0,   # Lawson P7 R2, P9 R3 — extracting solid results
    "Alpine":       6.0,   # Stable midfield; Gasly points every race
    "Williams":     7.0,   # Albon/Sainz above-car extraction
    "Cadillac":     8.0,
    "Red Bull":     9.0,   # Verstappen P6/DNF/P8 — struggling despite upgrades
    "Aston Martin": 10.0,
    "Audi":         11.0,
}

# Legacy aliases for backward compatibility
PACE_RANK_2026_R5    = PACE_RANK_2026_R3
PACE_RANK_2026_EARLY = PACE_RANK_2026_R3


class FeatureEngineering:
    """
    Transforms raw FastF1 lap DataFrames into the feature-enriched DataFrames
    consumed by the ML models.

    Usage:
        eng = FeatureEngineering()
        processed_path = eng.process_session("data/raw/telemetry/laps_2026_R03_Q.parquet",
                                             circuit_name="Suzuka")
    """

    def __init__(self) -> None:
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def process_session(self, raw_parquet_path: str, circuit_name: str) -> Path:
        """
        Load a raw Parquet lap file, compute all features, and save a processed
        Parquet file to data/processed/.

        Args:
            raw_parquet_path: Path produced by F1DataPipeline.
            circuit_name:     Used for static EHI lookup.

        Returns:
            Path to the processed Parquet file.
        """
        logger.info(f"Processing features for: {raw_parquet_path}")
        df = pd.read_parquet(raw_parquet_path, engine="pyarrow")

        df = self._add_ber(df)
        df = self._add_ehi(df, circuit_name)
        df = self._add_ned(df)
        df = self._add_tgi(df)
        df = self._add_aasp(df)
        # RHS is computed by the RRM module incrementally — not added here

        out_path = self.processed_dir / Path(raw_parquet_path).name.replace("laps_", "features_")
        df.to_parquet(out_path, engine="pyarrow", index=False)
        logger.info(f"Saved {len(df)} rows with features to {out_path}")
        return out_path

    # ── Feature Computations ───────────────────────────────────────────────

    @staticmethod
    def _add_ber(df: pd.DataFrame) -> pd.DataFrame:
        """
        Battery Efficiency Ratio (BER).
        Estimated from the ratio of each driver's fastest clean lap in a stint
        to the median clean lap — a high BER implies consistent high deployment.

        BER = 1 - (lap_time / session_median_lap_time)
        Values near 0 = median pace | positive = faster than median.
        """
        if "LapTime" not in df.columns:
            logger.warning("LapTime column absent — skipping BER")
            df["BER"] = np.nan
            return df

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use only accurate (non-pit, non-safety-car) laps
            clean = df[df.get("IsAccurate", pd.Series([True] * len(df), index=df.index))]
            median_laptime = clean["LapTime"].median()

            if pd.isna(median_laptime) or median_laptime == 0:
                df["BER"] = np.nan
            else:
                df["BER"] = 1 - (df["LapTime"] / median_laptime)

        return df

    @staticmethod
    def _add_ehi(df: pd.DataFrame, circuit_name: str) -> pd.DataFrame:
        """
        Energy Harvest Index (EHI).
        Static per-circuit score — same for all drivers at a given circuit.
        """
        ehi = CIRCUIT_EHI.get(circuit_name, 0.60)
        df["EHI"] = ehi
        return df

    @staticmethod
    def _add_ned(df: pd.DataFrame) -> pd.DataFrame:
        """
        New Era Delta (NED).
        Captures how much each constructor gained/lost in the regulation reset.
        Positive = improved vs 2025; negative = worse.
        NED = normalised(2025_WCC_points) - normalised(2026_pace_rank_inverted)
        """
        max_2025 = max(STANDINGS_2025_WCC.values())
        max_rank  = max(PACE_RANK_2026_EARLY.values())

        def _ned_for_team(team: str) -> float:
            pts_2025 = STANDINGS_2025_WCC.get(team, 0.0)
            rank_2026 = PACE_RANK_2026_EARLY.get(team, max_rank)
            score_2025 = pts_2025 / max_2025
            score_2026 = 1 - (rank_2026 / max_rank)  # Invert so higher = better
            return round(score_2026 - score_2025, 4)

        if "Team" in df.columns:
            df["NED"] = df["Team"].map(_ned_for_team)
        else:
            df["NED"] = np.nan

        return df

    @staticmethod
    def _add_tgi(df: pd.DataFrame) -> pd.DataFrame:
        """
        Teammate Gap Index (TGI).
        For each lap, compute the delta between a driver's lap time and their
        teammate's median lap time in the same session.
        Positive TGI = driver is faster than their teammate this session.
        """
        if "LapTime" not in df.columns or "Driver" not in df.columns or "Team" not in df.columns:
            df["TGI"] = np.nan
            return df

        team_medians: dict[str, float] = (
            df.groupby("Team")["LapTime"].median().to_dict()
        )

        def _tgi(row: pd.Series) -> float:
            team_med = team_medians.get(row.get("Team"))
            lap_t = row.get("LapTime")
            if pd.isna(team_med) or pd.isna(lap_t) or team_med == 0:
                return np.nan
            return float(team_med - lap_t)  # Positive = driver faster than team avg

        df["TGI"] = df.apply(_tgi, axis=1)
        return df

    @staticmethod
    def _add_aasp(df: pd.DataFrame) -> pd.DataFrame:
        """
        Active Aero Setting Proxy (AASP).
        Infers downforce level from speed-trap vs sector-time ratio.
        High speed-trap + slow sector 2 time = low-downforce (low drag) setup.
        Low speed-trap + fast sector 2 time = high-downforce (high grip) setup.

        AASP is normalised per session: 1 = highest downforce, 0 = lowest.
        """
        speed_col = next(
            (c for c in ["SpeedST", "SpeedFL", "SpeedI1"] if c in df.columns), None
        )
        sector2_col = "Sector2Time" if "Sector2Time" in df.columns else None

        if speed_col is None or sector2_col is None:
            logger.warning("Speed/sector columns absent — skipping AASP")
            df["AASP"] = np.nan
            return df

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            speed_norm = (df[speed_col] - df[speed_col].min()) / (
                df[speed_col].max() - df[speed_col].min() + 1e-9
            )
            # Invert speed (higher speed-trap = lower downforce)
            df["AASP"] = 1 - speed_norm

        return df
