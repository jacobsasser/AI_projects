import os
import json
import logging
from tqdm import tqdm
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from google.cloud import storage

# ───── Load Environment Variables ─────
load_dotenv()

# ───── Config ─────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "capstone")
BUCKET_NAME = os.getenv("BUCKET_NAME", "clinical-trial-data-pipeline")
EMBEDDING_PREFIX = os.getenv("EMBEDDING_PREFIX", "embeddings/")

# ───── Logging ─────────────────────────────
os.makedirs("/tmp/logs", exist_ok=True)
log_file = "/tmp/logs/pinecone_sync.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def upload_embeddings_to_pinecone(request):
    logging.info("=== Starting Pinecone Sync (ALL EMBEDDINGS) ===")

    # ───── Pinecone Init ─────
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        logging.info(f"Connected to Pinecone index '{PINECONE_INDEX}'.")
    except Exception as e:
        logging.exception("Pinecone init failed")
        return f"Pinecone init failed: {e}", 500

    # ───── GCS Init ─────
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix=EMBEDDING_PREFIX))
        jsonl_blobs = [b for b in blobs if b.name.endswith(".jsonl")]
        logging.info(f"Found {len(jsonl_blobs)} .jsonl files in {EMBEDDING_PREFIX}")
    except Exception as e:
        logging.exception("Failed to access GCS")
        return "Failed to access embeddings in GCS", 500

    # ───── Upload All Embedding Files ─────
    for blob in jsonl_blobs:
        filename = os.path.basename(blob.name)
        local_path = f"/tmp/{filename}"

        try:
            blob.download_to_filename(local_path)
            logging.info(f"Downloaded {blob.name} to {local_path}")
        except Exception as e:
            logging.error(f"Failed to download {blob.name}: {e}")
            continue

        try:
            with open(local_path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            logging.error(f"Failed to read {filename}: {e}")
            continue

        vectors = []
        for line in tqdm(lines, desc=f"Uploading {filename}"):
            try:
                record = json.loads(line)
                vector_id = f"{record['nct_id']}_{record['chunk_id']}"
                embedding = record["embedding"]
                metadata = {
                    "nct_id": record["nct_id"],
                    "chunk_id": record["chunk_id"],
                    "condition": record.get("condition", ""),
                    "text": record.get("text", "")
                }
                vectors.append((vector_id, embedding, metadata))
            except Exception as e:
                logging.warning(f"Skipping bad line in {filename}: {e}")

        for i in range(0, len(vectors), 100):
            batch = vectors[i:i+100]
            try:
                index.upsert(vectors=batch)
                logging.info(f"Uploaded batch {i}-{i+len(batch)} from {filename}")
            except Exception as e:
                logging.error(f"Failed to upsert batch from {filename}: {e}")

    # ───── Deletion Logic ─────
    try:
        delete_blob = bucket.blob("registry/to_delete.txt")
        delete_path = "/tmp/to_delete.txt"
        delete_blob.download_to_filename(delete_path)
        logging.info("Downloaded registry/to_delete.txt")

        with open(delete_path, "r") as f:
            nct_ids = [line.strip() for line in f if line.strip()]

        for nct_id in nct_ids:
            try:
                index.delete(filter={"nct_id": {"$eq": nct_id}})
                logging.info(f"Deleted vectors with nct_id = {nct_id}")
            except Exception as e:
                logging.warning(f"Failed to delete vectors for {nct_id}: {e}")
    except Exception as e:
        logging.warning(f"No delete file found or failed to process deletions: {e}")

    # ───── Upload Logs ─────
    try:
        log_blob = bucket.blob("logs/pinecone_sync.log")
        log_blob.upload_from_filename(log_file)
        logging.info("Uploaded sync log to GCS")
    except Exception as e:
        logging.error(f"Failed to upload log to GCS: {e}")
        return "Upload completed, but log upload failed", 500

    logging.info("Pinecone Sync Completed.")
    return "Full embedding sync completed", 200