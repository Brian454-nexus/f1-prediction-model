"""
F1 APEX — Free Practice Analysis
================================
Extracts representative long-run pace from FP2 telemetry to augment the
Constructor Performance Model prior to Sunday's GP.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("APEX.FPAnalysis")

def extract_long_runs(parquet_path: str, min_laps: int = 5) -> dict[str, float]:
    """
    Parses FP2 telemetry to find long runs on identical tire compounds.
    Returns a dictionary of { team_name: pace_advantage_seconds }.
    Negative values indicate a constructor is faster than the field median.
    """
    logger.info(f"Extracting long-run pace from {parquet_path}")
    
    if not Path(parquet_path).exists():
        logger.warning(f"Parquet file {parquet_path} does not exist. Skipping FP2 analysis.")
        return {}

    try:
        df = pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as e:
        logger.error(f"Could not read FP Parquet: {e}")
        return {}

    # Validate required columns
    required_cols = ["Driver", "Team", "Compound", "LapTime", "Stint", "IsAccurate"]
    if not all(c in df.columns for c in required_cols):
        logger.warning("Missing required lap columns for long-run analysis. Returning empty deltas.")
        return {}

    # Filter for accurate laps only (no pit in/out or VSC)
    accurate = df[df["IsAccurate"] == True].copy()
    
    if accurate.empty:
        logger.warning("No accurate laps found in dataset.")
        return {}

    # Identify continuous long runs grouped by Driver, Team, Stint, and Compound
    stints = accurate.groupby(["Driver", "Team", "Stint", "Compound"])
    
    team_best_medians = {}
    
    for (driver, team, stint, compound), group in stints:
        if len(group) >= min_laps:
            median_lap = group["LapTime"].median()
            
            # Record the best long-run median if they did multiple stints
            if team not in team_best_medians or median_lap < team_best_medians[team]:
                team_best_medians[team] = median_lap

    if not team_best_medians:
        logger.warning(f"No valid long runs (>={min_laps} laps) found. Session may be wet or red-flagged.")
        return {}

    # Compute deltas versus field median to standardise pace advantage
    field_median = np.median(list(team_best_medians.values()))
    
    deltas = {}
    for team, pace in team_best_medians.items():
        # Delta = Team Median - Field Median
        # Negative delta means the team was inherently faster than average
        deltas[team] = round(float(pace - field_median), 4)
        
    logger.info(f"FP2 Analysis complete. {len(deltas)} teams logged viable long runs.")
    return deltas
