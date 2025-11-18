import os
import json
import torch
import logging
import pandas as pd
import numpy as np
from google.cloud import storage
from transformers import AutoTokenizer, AutoModel

# ───── Config ─────────────────────────────
BUCKET_NAME = "clinical-trial-data-pipeline"
PROCESSED_FOLDER = "processed/"
REGISTRY_PREFIX = "registry/"
EMBEDDINGS_FOLDER = "embeddings"
LOG_PATH = "logs/embed.log"

MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
CHUNK_SIZE = 512

# ───── Logging ─────────────────────────────
os.makedirs("/tmp/logs", exist_ok=True)
log_file = "/tmp/logs/embed.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ───── Model Setup ─────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ───── GCS Setup ───────────────────────────
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

def read_to_insert_ids():
    blob = bucket.blob(f"{REGISTRY_PREFIX}to_insert.txt")
    if not blob.exists():
        logging.warning("No to_insert.txt found.")
        return set()
    ids = blob.download_as_text().splitlines()
    return set(i.strip() for i in ids if i.strip())

def chunk_text(text, tokenizer, chunk_size=512, overlap=0.2):
    tokens = tokenizer.encode(text, truncation=False)
    stride = int(chunk_size * (1 - overlap))
    if stride <= 0:
        raise ValueError("Overlap too high.")

    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), stride) if i + chunk_size <= len(tokens)]
    if len(tokens) > chunk_size and (len(tokens) - chunk_size) % stride != 0:
        chunks.append(tokens[-chunk_size:])
    return chunks

def embed_chunk(chunk_tokens):
    with torch.no_grad():
        input_ids = torch.tensor([chunk_tokens]).to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = output.last_hidden_state[:, 0, :]  # CLS token
        return embeddings.squeeze().cpu().numpy()

def generate_embeddings(request):
    logging.info("=== Starting Embedding Generation ===")
    try:
        to_insert = read_to_insert_ids()
        logging.info(f"Loaded {len(to_insert)} NCT IDs from to_insert.txt")
    except Exception as e:
        logging.error(f"Failed to read to_insert.txt: {e}")
        return "Failed to read insertion list", 500

    if not to_insert:
        logging.info("No new trials to embed.")
        return "No new trials to embed", 200

    try:
        blobs = list(bucket.list_blobs(prefix=PROCESSED_FOLDER))
        logging.info(f"Found {len(blobs)} files in {PROCESSED_FOLDER}")
    except Exception as e:
        logging.error(f"Failed to list blobs: {e}")
        return "Failed to list blobs", 500


    for blob in blobs:
        if not blob.name.endswith(".parquet"):
            continue

        try:
            local_path = f"/tmp/{os.path.basename(blob.name)}"
            blob.download_to_filename(local_path)
            logging.info(f"Downloaded {blob.name} to {local_path}")
        except Exception as e:
            logging.error(f"Failed to download {blob.name}: {e}")
            return f"Failed to download {blob.name}", 500

        try:
            df = pd.read_parquet(local_path, engine="pyarrow")
            logging.info(f"Loaded DataFrame with {len(df)} records from {blob.name}")
        except Exception as e:
            logging.error(f"Failed to load parquet file: {e}")
            return f"Failed to load parquet file {e}", 500

        try:
            df = df[df["nct_id"].isin(to_insert)]
            logging.info(f"Filtered DataFrame to {len(df)} records in to_insert")
        except Exception as e:
            logging.error(f"Filtering by NCT ID failed: {e}")
            return "NCT ID filtering failed", 500

        if df.empty:
            logging.info("No relevant records to embed in this file.")
            continue

        try:
            logging.info("Generating embedding text fields...")
            df["text"] = (
                "Title: " + df["official_title"].fillna("") + "\n"
                + "Summary: " + df["summary"].fillna("") + "\n"
                + "Description: " + df["description"].fillna("") + "\n"
                + "Interventions: " + df["interventions"].fillna("") + "\n"
                + "Primary Outcomes: " + df["primary_outcomes"].fillna("") + "\n"
                + "Eligibility Criteria: " + df["eligibility_inclusion"].fillna("") + "\n"
                + "Exclusions: " + df["eligibility_exclusion"].fillna("") + "\n"
                + "Locations: " + df["locations"].fillna("")
            )
        except Exception as e:
            logging.error(f"Failed to construct text field: {e}")
            return "Text field creation failed", 500

        records = []
        try:
            logging.info("Starting chunking and embedding loop...")
            for _, row in df.iterrows():
                try:
                    chunks = chunk_text(row["text"], tokenizer, CHUNK_SIZE)
                    for i, chunk in enumerate(chunks):
                        embedding = embed_chunk(chunk)
                        records.append({
                            "nct_id": row["nct_id"],
                            "chunk_id": i,
                            "embedding": embedding.tolist(),
                            "condition": row["conditions"],
                            "text": row["text"]
                        })
                except Exception as inner_e:
                    logging.warning(f"Failed to process row {row.get('nct_id', 'unknown')}: {inner_e}")
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}")
            return "Chunking/Embedding failed", 500

        try:
            json_path = "/tmp/embeddings.jsonl"
            with open(json_path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            logging.info(f"Wrote {len(records)} embeddings to {json_path}")
        except Exception as e:
            logging.error(f"Failed to write JSONL: {e}")
            return "Failed to write embeddings file", 500

        try:
            embed_blob = f"{EMBEDDINGS_FOLDER}/embeddings_{os.path.basename(blob.name).replace('.parquet', '')}.jsonl"
            bucket.blob(embed_blob).upload_from_filename(json_path)
            logging.info(f"Uploaded embeddings to {embed_blob}")
        except Exception as e:
            logging.error(f"Failed to upload embeddings: {e}")
            return "Upload to GCS failed", 500

    try:
        bucket.blob(LOG_PATH).upload_from_filename(log_file)
        logging.info(f"Uploaded log file to {LOG_PATH}")
    except Exception as e:
        logging.error(f"Failed to upload log file: {e}")
        return "Log file upload failed", 500

    return "Embedding generation completed.", 200

