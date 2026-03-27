"""
F1 APEX — Data Pipeline
=======================
Ingests official FOM telemetry via the FastF1 API and serializes each
session's lap data to Parquet files under data/raw/telemetry/.

This module is the entry-point for the entire data layer. Downstream
modules (feature_engineering, models) consume the Parquet output rather
than calling FastF1 directly, ensuring a clear separation of concerns and
making the pipeline fully resumable from disk at any stage.

Design choices:
  - FastF1 offline cache is always enabled to limit network calls.
  - Each session is serialized immediately upon load — if a later pipeline
    step fails, the ingested data is already persisted and doesn't need
    re-downloading.
  - Parquet is chosen over CSV for columnar storage efficiency and type
    preservation (timedeltas, categoricals).
"""

from __future__ import annotations

import os
from pathlib import Path

import fastf1 as ff1
import pandas as pd

from utils.logger import get_logger

logger = get_logger("APEX.DataPipeline")

# ── Constants ────────────────────────────────────────────────────────────────

FF1_CACHE_DIR: str = os.getenv("FF1_CACHE_DIR", "cache/fastf1")

# Sessions in race-weekend chronological order
SESSION_ORDER: list[str] = ["FP1", "FP2", "FP3", "Q", "R"]

# Sprint weekend session order (FP1 replaced by SQ + Sprint block)
SESSION_ORDER_SPRINT: list[str] = ["FP1", "SQ", "S", "Q", "R"]

# 2026 calendar sprint weekends (round numbers)
SPRINT_WEEKENDS_2026: frozenset[int] = frozenset([
    2,   # China (Shanghai)
    5,   # Saudi Arabia (Jeddah)
    10,  # Canada (Montreal)
    14,  # Belgium (Spa)
    19,  # United States (Austin)
    21,  # Brazil (Sao Paulo)
])

# Mapping from Ergast/FastF1 round number to short circuit identifier
CIRCUIT_MAP: dict[int, dict] = {
    1: {"name": "Albert Park",  "coords": (-37.8497, 144.9680)},
    2: {"name": "Shanghai",     "coords": (31.3389, 121.2194)},
    3: {"name": "Suzuka",       "coords": (34.8431, 136.5411)},
    4: {"name": "Bahrain",      "coords": (26.0325, 50.5106)},
    5: {"name": "Jeddah",       "coords": (21.6319, 39.1044)},
    6: {"name": "Miami",        "coords": (25.9581, -80.2389)},
    7: {"name": "Imola",        "coords": (44.3439, 11.7167)},
    8: {"name": "Monaco",       "coords": (43.7347, 7.4206)},
    9: {"name": "Catalunya",    "coords": (41.5700, 2.2611)},
    10: {"name": "Montreal",    "coords": (45.5017, -73.5224)},
    11: {"name": "Spielberg",   "coords": (47.2197, 14.7647)},
    12: {"name": "Silverstone", "coords": (52.0786, -1.0169)},
    13: {"name": "Budapest",    "coords": (47.5789, 19.2486)},
    14: {"name": "Spa",         "coords": (50.4372, 5.9717)},
    15: {"name": "Zandvoort",   "coords": (52.3888, 4.5409)},
    16: {"name": "Monza",       "coords": (45.6156, 9.2811)},
    17: {"name": "Baku",        "coords": (40.3724, 49.8533)},
    18: {"name": "Singapore",   "coords": (1.2914, 103.8640)},
    19: {"name": "Austin",      "coords": (30.1328, -97.6411)},
    20: {"name": "Mexico City", "coords": (19.4042, -99.0907)},
    21: {"name": "Sao Paulo",   "coords": (-23.7036, -46.6997)},
    22: {"name": "Las Vegas",   "coords": (36.2679, -115.0125)},
    23: {"name": "Lusail",      "coords": (25.4900, 51.4542)},
    24: {"name": "Yas Marina",  "coords": (24.4672, 54.6031)},
}


# ── Pipeline Class ────────────────────────────────────────────────────────────

class F1DataPipeline:
    """
    Manages FastF1 data ingestion for a single F1 season.

    Usage:
        pipeline = F1DataPipeline(year=2026)
        parquet_path = pipeline.fetch_session(round_id=3, session_id="Q")
    """

    def __init__(self, year: int = 2026) -> None:
        self.year = year
        self.raw_dir = Path("data/raw/telemetry")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Enable FastF1 offline cache
        cache_path = Path(FF1_CACHE_DIR)
        cache_path.mkdir(parents=True, exist_ok=True)
        ff1.Cache.enable_cache(str(cache_path))
        logger.info(f"FastF1 cache enabled at: {cache_path.resolve()}")

    # ── Public API ─────────────────────────────────────────────────────────

    def fetch_session(self, round_id: int, session_id: str) -> Path:
        """
        Load a single session and serialize laps to Parquet.

        Args:
            round_id:   Race round number (1-based, matches Ergast convention).
            session_id: One of FP1 | FP2 | FP3 | Q | R.

        Returns:
            Path to the written Parquet file.
        """
        output_path = self._parquet_path(round_id, session_id)

        if output_path.exists():
            logger.info(f"Cache hit — using existing Parquet: {output_path}")
            return output_path

        logger.info(f"Fetching {self.year} R{round_id} {session_id} from FastF1 ...")

        session = ff1.get_session(self.year, round_id, session_id)
        session.load(telemetry=True, laps=True, weather=True)

        laps = session.laps.copy()
        laps = self._sanitize_laps(laps, round_id, session_id)

        laps.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info(f"Saved {len(laps)} laps to {output_path}")
        return output_path

    def fetch_full_weekend(self, round_id: int) -> dict[str, Path]:
        """
        Sequentially fetch all sessions for a race weekend.

        Returns:
            Mapping of session_id -> Parquet path.
        """
        session_order = (
            SESSION_ORDER_SPRINT if self.is_sprint_weekend(round_id) else SESSION_ORDER
        )
        results: dict[str, Path] = {}
        for session_id in session_order:
            try:
                if session_id in ("S", "SQ"):
                    results[session_id] = self.fetch_sprint(round_id, session_id)
                else:
                    results[session_id] = self.fetch_session(round_id, session_id)
            except Exception as exc:
                logger.warning(f"Could not fetch {session_id} (Round {round_id}): {exc}")
        return results

    def get_circuit_coords(self, round_id: int) -> tuple[float, float]:
        """Return (lat, lon) for a given round, used by the weather router."""
        if round_id in CIRCUIT_MAP:
            return CIRCUIT_MAP[round_id]["coords"]
        raise ValueError(f"Round {round_id} not found in CIRCUIT_MAP")

    def get_circuit_name(self, round_id: int) -> str:
        """Return the human-readable circuit name for a given round."""
        if round_id in CIRCUIT_MAP:
            return CIRCUIT_MAP[round_id]["name"]
        raise ValueError(f"Round {round_id} not found in CIRCUIT_MAP")

    def is_sprint_weekend(self, round_id: int) -> bool:
        """Return True if the given round is a sprint weekend."""
        return round_id in SPRINT_WEEKENDS_2026

    def fetch_sprint(self, round_id: int, session_type: str = "S") -> Path:
        """
        Fetch and cache a sprint session.
        FastF1 mapping: 'S' -> 'Sprint', 'SQ' -> 'Sprint Qualifying'.

        Sprint weekends only — raises ValueError for non-sprint rounds.
        """
        ff1_name   = "Sprint" if session_type == "S" else "Sprint Qualifying"
        output_path = self._parquet_path(round_id, session_type)

        if output_path.exists():
            logger.info(f"Sprint cache hit: {output_path}")
            return output_path

        if not self.is_sprint_weekend(round_id):
            raise ValueError(f"Round {round_id} is not a sprint weekend")

        logger.info(f"Fetching {self.year} R{round_id} {ff1_name} from FastF1 ...")

        session = ff1.get_session(self.year, round_id, ff1_name)
        session.load(telemetry=False, laps=True, weather=True)

        laps = session.laps.copy()
        laps = self._sanitize_laps(laps, round_id, session_type)
        laps.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info(f"Saved {len(laps)} sprint laps to {output_path}")
        return output_path

    def fetch_sprint_results(self, round_id: int) -> list[dict]:
        """
        Extract structured sprint-race finishing results for DPI update_after_sprint().
        DNF drivers flagged from FastF1 Status column. Technical DNFs are flagged
        but their Elo is protected by update_after_sprint car-DNF exclusion logic.

        Returns:
            List of {driver, team, finish_pos, dnf, cause} dicts.
        """
        sprint_path = self._parquet_path(round_id, "S")
        if not sprint_path.exists():
            self.fetch_sprint(round_id, "S")

        laps = pd.read_parquet(sprint_path)

        if "Driver" not in laps.columns:
            logger.warning("Sprint Parquet missing 'Driver' column")
            return []

        # One row per driver: take their final lap (highest LapNumber)
        driver_laps = (
            laps.sort_values("LapNumber", ascending=False)
            .groupby("Driver")
            .first()
            .reset_index()
        )

        results = []
        finish_pos = 1
        for _, row in driver_laps.sort_values("Position", na_position="last").iterrows():
            status = str(row.get("Status", "")).lower().strip()
            dnf    = status not in ("", "finished", "1 lap", "+1 lap")
            cause  = status if dnf else ""
            results.append({
                "driver":     row["Driver"],
                "team":       row.get("Team", ""),
                "finish_pos": finish_pos,
                "dnf":        dnf,
                "cause":      cause,
            })
            if not dnf:
                finish_pos += 1

        logger.info(f"Sprint results: {len(results)} drivers extracted (Round {round_id})")
        return results

    # ── Internal Helpers ───────────────────────────────────────────────────

    def _parquet_path(self, round_id: int, session_id: str) -> Path:
        return self.raw_dir / f"laps_{self.year}_R{round_id:02d}_{session_id}.parquet"

    @staticmethod
    def _sanitize_laps(laps: pd.DataFrame, round_id: int, session_id: str) -> pd.DataFrame:
        """
        Adds metadata columns and converts timedelta columns to float seconds
        so that Parquet can serialize them without type issues.
        """
        # Attach round and session metadata
        laps["RoundId"] = round_id
        laps["SessionId"] = session_id

        # Convert timedelta columns to float seconds for Parquet compatibility
        timedelta_cols = laps.select_dtypes(include=["timedelta64[ns]"]).columns
        for col in timedelta_cols:
            laps[col] = laps[col].dt.total_seconds()

        return laps


# ── CLI helper ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(description="F1 APEX -- Data Pipeline CLI")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--round", type=int, required=True, dest="round_id")
    parser.add_argument(
        "--session", type=str, default="Q",
        help="Session to fetch: FP1/FP2/FP3/Q/R/S/SQ",
    )
    parser.add_argument(
        "--sprint-results", action="store_true",
        help="Print structured sprint results JSON for DPI seeding",
    )
    args = parser.parse_args()

    pipeline = F1DataPipeline(year=args.year)
    print(f"Round {args.round_id} | Sprint weekend: {pipeline.is_sprint_weekend(args.round_id)}")

    if args.sprint_results:
        results = pipeline.fetch_sprint_results(args.round_id)
        print(_json.dumps(results, indent=2))
    elif args.session in ("S", "SQ"):
        path = pipeline.fetch_sprint(args.round_id, args.session)
        print(f"Sprint Parquet saved to: {path}")
    else:
        path = pipeline.fetch_session(args.round_id, args.session)
        print(f"Parquet saved to: {path}")
