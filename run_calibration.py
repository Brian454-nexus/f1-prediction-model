import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

import fastf1
from utils.logger import get_logger
from orchestration.race_weekend_dag import race_weekend_flow
from models.calibration import AutoCalibrator
from models.meta_ensemble import DriverPrediction

logger = get_logger("Calibration_Runner")

def run():
    # 1. Generate predictions for China
    logger.info("Re-running DAG to fetch predictions for China 2026...")
    dag_result = race_weekend_flow(
        round_id=2, circuit_name="Shanghai", total_laps=56, race_utc="2026-03-15T07:00:00Z"
    )
    predicted_dicts = dag_result["predictions"]
    
    # 2. Fetch actual results from FastF1
    logger.info("Fetching actual results from FastF1...")
    session = fastf1.get_session(2026, 2, 'R')
    session.load(telemetry=False, weather=False, laps=False)
    results_df = session.results
    
    actual_results = []
    for _, row in results_df.iterrows():
        status = str(row.get("Status", "")).lower().strip()
        dnf = status not in ("", "finished", "1 lap", "+1 lap", "2 laps", "+2 laps")
        driver_name = row.get("LastName", row.get("BroadcastName", ""))
        actual_results.append({
            "driver": driver_name,
            "team": row.get("TeamName", ""),
            "finish_pos": row["Position"],
            "dnf": dnf,
            "cause": status if dnf else ""
        })

    # 3. Create objects for calibration
    pred_objs = [DriverPrediction(**p) for p in predicted_dicts]

    # 4. Trigger Calibration
    calibrator = AutoCalibrator()
    calib_metrics = calibrator.calibrate(actual_results, pred_objs, 2, "Shanghai")
    logger.info(f"Final Calibration Summary: {calib_metrics}")

if __name__ == "__main__":
    run()
