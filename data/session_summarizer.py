"""
F1 APEX — Session Summarizer (Token-Optimized)
==============================================
Condenses thousands of raw telemetry rows into a high-density JSON summary
specifically for LLM consumption (protecting the token budget).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("APEX.SessionSummarizer")

def summarize_session(parquet_path: str, session_id: str) -> dict:
    """
    Transforms a session Parquet into a concise dictionary of key stats.
    """
    if not Path(parquet_path).exists():
        logger.warning(f"File not found: {parquet_path}")
        return {}

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.error(f"Error reading Parquet: {e}")
        return {}

    # Basic session info
    summary = {
        "session": session_id,
        "total_drivers": df["Driver"].nunique(),
        "track_status": "Dry" if df.get("TrackStatus", pd.Series(["1"])).iloc[0] == "1" else "Wet/Mixed",
        "drivers": {}
    }

    # Aggregate performance by driver
    for driver, group in df.groupby("Driver"):
        # Accurate laps only
        accurate = group[group["IsAccurate"] == True]
        if accurate.empty: continue

        best_lap = accurate["LapTime"].min()
        median_lap = accurate["LapTime"].median()
        
        # Identify "long runs" (>5 laps) to check consistency
        stints = accurate.groupby("Stint")
        consistency_score = None
        for stint, s_laps in stints:
            if len(s_laps) >= 5:
                # Standard deviation of lap times — lower is more consistent
                std = s_laps["LapTime"].std()
                if consistency_score is None or std < consistency_score:
                    consistency_score = round(float(std), 3)

        # Top speed (SpeedST or SpeedFL)
        top_speed = float(group[["SpeedST", "SpeedFL"]].max().max())

        summary["drivers"][driver] = {
            "team": group["Team"].iloc[0],
            "best_lap": round(float(best_lap), 3),
            "race_pace_median": round(float(median_lap), 3),
            "consistency_delta": consistency_score,
            "top_speed": int(top_speed) if not np.isnan(top_speed) else None
        }

    return summary
