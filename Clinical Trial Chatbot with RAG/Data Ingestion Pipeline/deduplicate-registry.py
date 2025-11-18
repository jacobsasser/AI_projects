from google.cloud import storage
import logging
import os

# Setup local logging
os.makedirs("/tmp/logs", exist_ok=True)
log_file = "/tmp/logs/deduplication.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# GCS Configuration
BUCKET_NAME = "clinical-trial-data-pipeline"
REGISTRY_PREFIX = "registry/"

def read_gcs_textfile(client, bucket_name, blob_name):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if blob.exists():
        content = blob.download_as_text()
        return set(line.strip() for line in content.splitlines() if line.strip())
    else:
        logging.warning(f"Blob {blob_name} not found.")
        return set()

def write_gcs_textfile(client, bucket_name, blob_name, lines):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = "\n".join(sorted(lines))
    blob.upload_from_string(content, content_type="text/plain")
    logging.info(f"Wrote {len(lines)} lines to {blob_name}")

def deduplicate_registry(request):
    client = storage.Client()

    # Step 1: Load current and active IDs
    current_ids = read_gcs_textfile(client, BUCKET_NAME, f"{REGISTRY_PREFIX}current_ids.txt")
    active_ids = read_gcs_textfile(client, BUCKET_NAME, f"{REGISTRY_PREFIX}active_ids.txt")

    logging.info(f"Loaded {len(current_ids)} current IDs and {len(active_ids)} active IDs.")

    # Step 2: Compute differences
    to_insert = current_ids - active_ids
    to_delete = active_ids - current_ids

    logging.info(f"New IDs to insert: {len(to_insert)}")
    logging.info(f"Old IDs to delete: {len(to_delete)}")

    # Step 3: Write results back to GCS
    write_gcs_textfile(client, BUCKET_NAME, f"{REGISTRY_PREFIX}to_insert.txt", to_insert)
    write_gcs_textfile(client, BUCKET_NAME, f"{REGISTRY_PREFIX}to_delete.txt", to_delete)

    # Step 4: (Optional) Update active_ids to reflect current registry
    write_gcs_textfile(client, BUCKET_NAME, f"{REGISTRY_PREFIX}active_ids.txt", current_ids)

    # Upload log
    bucket = client.bucket(BUCKET_NAME)
    bucket.blob("logs/deduplication.log").upload_from_filename(log_file)

    return f"Deduplication complete: {len(to_insert)} to insert, {len(to_delete)} to delete."
