import os
import logging
from google.cloud import storage

# ───── Logging Setup ─────
os.makedirs("/tmp/logs", exist_ok=True)
log_file = "/tmp/logs/cleanup.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def clean_gcs_folders(request):
    logging.info("=== Starting GCS Cleanup ===")
    
    bucket_name = "clinical-trial-data-pipeline"
    prefixes = ["embeddings/", "processed/", "raw/"]
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for prefix in prefixes:
        try:
            blobs = list(bucket.list_blobs(prefix=prefix))
            if not blobs:
                logging.info(f"No files found under '{prefix}' to delete.")
                continue

            for blob in blobs:
                blob.delete()
                logging.info(f"Deleted: {blob.name}")

            logging.info(f"All files under '{prefix}' deleted successfully.")

        except Exception as e:
            logging.error(f"Error deleting files under '{prefix}': {e}")

    # ───── Upload Log File ─────
    try:
        log_blob = bucket.blob("logs/cleanup.log")
        log_blob.upload_from_filename(log_file)
        logging.info("Uploaded cleanup log to GCS.")
    except Exception as e:
        logging.error(f"Failed to upload cleanup log: {e}")
        return "Cleanup complete, but log upload failed", 500

    return "GCS folder cleanup complete", 200
