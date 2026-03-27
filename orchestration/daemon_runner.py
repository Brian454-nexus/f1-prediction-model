"""
F1 APEX — Live Session Ingestion Daemon
=========================================
Polls and automatically triggers Prefect ingestion tasks for live race weekend 
sessions as soon as the FIA/FastF1 telemetry datastream is assembled.
"""

import time
import schedule
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from orchestration.race_weekend_dag import task_ingest_session

logger = get_logger("APEX.Daemon")

def trigger_ingestion(round_id: int, session_id: str):
    logger.info(f"Triggering Auto-Ingestion for Round {round_id} - {session_id} 🚀")
    try:
        # Utilizing the Prefect task `.fn()` to bypass the backend UI requirement and execute locally
        # The task inherently has 3 retries w/ 30s delays in case FastF1 CDN is out of sync.
        task_ingest_session.fn(round_id, session_id)
        logger.info(f"✅ Auto-Ingestion for {session_id} successfully parsed and cached.")
    except Exception as e:
        logger.error(f"❌ Failed to parse {session_id} telemetry: {e}")


# ── Japan GP 2026 (Suzuka) Ingestion Schedule ───────────────────────────────
# FP1 normally starts 02:30 UTC (05:30 EAT). We trigger ingestion at 07:00 EAT to ensure parsed CDN availability.
schedule.every().friday.at("07:00").do(trigger_ingestion, round_id=3, session_id="FP1")

# FP2 normally starts 06:00 UTC (09:00 EAT). We trigger at 10:30 EAT.
schedule.every().friday.at("10:30").do(trigger_ingestion, round_id=3, session_id="FP2")

if __name__ == "__main__":
    logger.info("📡 F1 APEX Ingestion Daemon Online")
    logger.info("Listening for Suzuka (Japan GP) FP1 and FP2 session closures...")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.warning("Daemon terminated by user.")
