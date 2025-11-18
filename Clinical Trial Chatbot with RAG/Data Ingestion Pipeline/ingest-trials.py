import requests
import json
import time
import os
import logging
from datetime import datetime
from google.cloud import storage

# CONFIG
BUCKET_NAME = "clinical-trial-data-pipeline"
GCS_FOLDER = "raw"
LOG_FOLDER = "logs"
REGISTRY_FOLDER = "registry"
LOG_FILE = "clinical_trials_fetch.log"
PAGE_SIZE = 100
MAX_RETRIES = 5
RETRY_BACKOFF_FACTOR = 2
BUFFER_LIMIT = 10000  # records per file

FIELDS = [
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.identificationModule.officialTitle",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.statusModule.startDateStruct.date",
    "protocolSection.statusModule.completionDateStruct.date",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.descriptionModule.briefSummary",
    "protocolSection.descriptionModule.detailedDescription",
    "protocolSection.eligibilityModule.eligibilityCriteria",
    "protocolSection.eligibilityModule.sex",
    "protocolSection.eligibilityModule.minimumAge",
    "protocolSection.eligibilityModule.maximumAge",
    "protocolSection.designModule.studyType",
    "protocolSection.designModule.designInfo.primaryPurpose",
    "protocolSection.designModule.designInfo.interventionModel",
    "protocolSection.designModule.enrollmentInfo.count",
    "protocolSection.armsInterventionsModule.interventions",
    "protocolSection.contactsLocationsModule.locations",
    "protocolSection.outcomesModule.primaryOutcomes",
    "protocolSection.outcomesModule.secondaryOutcomes"
]

# Logging setup
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def fetch_data(params, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            response = requests.get("https://clinicaltrials.gov/api/v2/studies", params=params)
            if response.status_code == 200:
                return response.json()
            else:
                logging.warning(f"Status {response.status_code}: {response.text}")
        except Exception as e:
            logging.error(f"Request error: {e}")
        time.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))
    logging.critical("Failed to fetch data after retries.")
    return None

def upload_to_gcs(blob_name, content, content_type="application/jsonl"):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content, content_type=content_type)
    logging.info(f"Uploaded to gs://{BUCKET_NAME}/{blob_name}")

def run(request):
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    params = {
        "format": "json",
        "pageSize": PAGE_SIZE,
        "fields": ",".join(FIELDS),
        "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING,ENROLLING_BY_INVITATION"
    }

    total = 0
    page = 1
    chunk_num = 1
    buffer = []
    nct_ids = set()

    logging.info("Starting paginated fetch...")

    while True:
        logging.info(f"Fetching page {page}...")
        data = fetch_data(params)
        if not data:
            logging.error("Stopping ingestion due to fetch failure.")
            break

        studies = data.get("studies", [])
        if not studies:
            logging.info("No more studies returned. Ending.")
            break

        for r in studies:
            nct_id = r.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            if nct_id:
                nct_ids.add(nct_id)
            buffer.append(r)

        total += len(studies)
        logging.info(f"Page {page}: {len(studies)} studies (buffer size: {len(buffer)})")

        if len(buffer) >= BUFFER_LIMIT:
            jsonl_str = "\n".join(json.dumps(r) for r in buffer)
            blob_name = f"{GCS_FOLDER}/clinical_trials_part_{chunk_num}_{timestamp}.jsonl"
            upload_to_gcs(blob_name, jsonl_str)
            logging.info(f"Flushed chunk {chunk_num} with {len(buffer)} records")
            buffer = []
            chunk_num += 1

        next_token = data.get("nextPageToken")
        if not next_token:
            logging.info("Reached end of pages.")
            break

        params["pageToken"] = next_token
        page += 1
        time.sleep(0.5)

    # Final flush
    if buffer:
        jsonl_str = "\n".join(json.dumps(r) for r in buffer)
        blob_name = f"{GCS_FOLDER}/clinical_trials_part_{chunk_num}_{timestamp}.jsonl"
        upload_to_gcs(blob_name, jsonl_str)
        logging.info(f"Flushed final chunk {chunk_num} with {len(buffer)} records")

    # Save current_ids registry file
    registry_blob = f"{REGISTRY_FOLDER}/current_ids.txt"
    upload_to_gcs(registry_blob, "\n".join(sorted(nct_ids)), content_type="text/plain")

    # Upload log file
    log_blob_path = f"{LOG_FOLDER}/clinical_trials_fetch_{timestamp}.log"
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            log_contents = f.read()
        upload_to_gcs(log_blob_path, log_contents, content_type="text/plain")
        logging.info(f"Uploaded log file to {log_blob_path}")
    else:
        logging.warning("Log file not found during upload.")

    return f"Uploaded {total} studies in {chunk_num} chunks and {len(nct_ids)} unique IDs with logs and registry."

