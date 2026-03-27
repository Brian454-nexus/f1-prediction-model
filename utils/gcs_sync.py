"""
F1 APEX — GCS State Sync Utility
===================================
Synchronises model state and Parquet intermediates between local disk
and Google Cloud Storage to ensure Cloud Run container restarts
do not lose accumulated race posteriors.

STATE LIFECYCLE:
    1. Container starts (Cloud Run) → `restore_from_gcs()` called first.
    2. Pipeline runs → sub-models update local files under models/state/.
    3. Pipeline ends → `sync_to_gcs()` called to persist everything.

WHAT IS SYNCED:
    - models/state/**   (CPM posteriors, DPI ratings, RRM survival state,
                         ensemble weights, accuracy log)
    - data/processed/   (Feature-engineered Parquet files)
    - data/raw/telemetry/ (Raw FastF1 Parquet cache — avoids re-downloading)

RUN LOCALLY:
    python utils/gcs_sync.py --action restore
    python utils/gcs_sync.py --action push
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Load GCP_PROJECT and GOOGLE_APPLICATION_CREDENTIALS from .env

from utils.logger import get_logger

logger = get_logger("APEX.GCSSync")

GCS_BUCKET       = os.getenv("GCS_BUCKET", "f1-apex-state")
GCP_PROJECT      = os.getenv("GCP_PROJECT", "")

# Local directories to sync
SYNC_DIRS = [
    Path("models/state"),
    Path("data/processed"),
    Path("data/raw/telemetry"),
]


def _get_gcs_client():
    """Lazy import of google-cloud-storage to avoid import errors in offline mode."""
    try:
        from google.cloud import storage
        return storage.Client(project=GCP_PROJECT) if GCP_PROJECT else storage.Client()
    except ImportError:
        logger.error("google-cloud-storage not installed. Run: pip install google-cloud-storage")
        raise
    except Exception as exc:
        logger.error(f"GCS client init failed: {exc}")
        raise


def sync_to_gcs(dry_run: bool = False) -> int:
    """
    Upload all local state files to GCS.

    Args:
        dry_run: If True, log what would be uploaded without uploading.

    Returns:
        Number of files uploaded (0 for dry runs).
    """
    logger.info(f"Syncing state to gs://{GCS_BUCKET}/ {'(DRY RUN)' if dry_run else ''}")
    uploaded = 0

    if not dry_run:
        client = _get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)

    for sync_dir in SYNC_DIRS:
        if not sync_dir.exists():
            continue
        for local_file in sync_dir.rglob("*"):
            if local_file.is_file():
                blob_name = str(local_file).replace("\\", "/")
                if dry_run:
                    logger.info(f"  [DRY] Would upload: {local_file} → gs://{GCS_BUCKET}/{blob_name}")
                else:
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(local_file))
                    logger.debug(f"  Uploaded: {blob_name}")
                    uploaded += 1

    logger.info(f"Sync complete — {uploaded} files uploaded")
    return uploaded


def restore_from_gcs(dry_run: bool = False) -> int:
    """
    Download all state files from GCS to local disk.

    Args:
        dry_run: If True, log what would be downloaded without downloading.

    Returns:
        Number of files restored (0 for dry runs).
    """
    logger.info(f"Restoring state from gs://{GCS_BUCKET}/ {'(DRY RUN)' if dry_run else ''}")
    restored = 0

    if not dry_run:
        client = _get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blobs  = list(bucket.list_blobs())
    else:
        blobs = []

    for blob in blobs:
        local_path = Path(blob.name)
        if dry_run:
            logger.info(f"  [DRY] Would restore: gs://{GCS_BUCKET}/{blob.name} → {local_path}")
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))
            logger.debug(f"  Restored: {local_path}")
            restored += 1

    logger.info(f"Restore complete — {restored} files restored")
    return restored


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="F1 APEX — GCS Sync CLI")
    parser.add_argument("--action", choices=["push", "restore"], required=True)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    if args.action == "push":
        n = sync_to_gcs(dry_run=args.dry_run)
    else:
        n = restore_from_gcs(dry_run=args.dry_run)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}{args.action}: {n} files {'would be ' if args.dry_run else ''}processed")
