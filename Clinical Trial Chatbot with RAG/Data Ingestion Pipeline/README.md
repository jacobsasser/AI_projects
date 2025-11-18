# Clinical Trial Data Ingestion Pipeline

This folder contains a set of Python scripts used to maintain an up-to-date, semantically searchable repository of clinical trial records for the Clinical Trial Discovery chatbot. The scripts are designed to be deployed as **Google Cloud Run jobs** and executed on a schedule via **Google Cloud Scheduler**.

The pipeline ingests, cleans, embeds, and synchronizes clinical trial data from the ClinicalTrials.gov REST API (v2), ensuring the Pinecone vector database and Firebase Firestore remain fully aligned with the latest available records.

---

## Pipeline Overview

The ingestion workflow consists of **six stages**, each implemented as its own Cloud Run function:

### 1. **Ingest Trials**
- Downloads the latest ClinicalTrials.gov dataset (JSON via REST API v2)
- Stores raw records in `gs://<bucket>/raw/`  
- Ensures the system always begins with the newest dataset

### 2. **Process Trials**
- Cleans & normalizes structured fields  
- Applies preprocessing to unstructured text (HTML stripping, boilerplate removal, segmentation, etc.)  
- Exports optimized columnar Parquet files to `gs://<bucket>/processed/`

### 3. **Deduplicate Registry**
Compares the current ingestion with the existing Pinecone registry:

Files maintained in GCS:
- `active_ids.txt` — IDs currently in Pinecone  
- `current_ids.txt` — IDs found in newest ingestion  
- `to_insert.txt` — new trials needing embeddings  
- `to_delete.txt` — trials removed or no longer active  

This step determines exactly **what to add** and **what to remove**, preventing full reprocessing.

### 4. **Embedding Creation**
- Generates BioBERT embeddings for each newly identified trial  
- Saves embeddings in JSONL format to `gs://<bucket>/embeddings/`

### 5. **Pinecone Sync**
- Upserts new embeddings into Pinecone  
- Deletes inactive vectors  
- Uploads the corresponding full metadata to **Firebase Firestore**, ensuring the frontend chatbot can resolve vector IDs to complete trial summaries

### 6. **Cleanup**
- Removes obsolete files from `raw/`, `processed/`, and `embeddings/`  
- Prevents GCS clutter and unnecessary storage cost

---

## Scheduling & Automation

The entire sequence is orchestrated by **Google Cloud Scheduler**, which triggers each stage in order.  
This ensures that the system stays synchronized with ClinicalTrials.gov’s update cadence (typically Monday–Friday at ~9 a.m. EST), while minimizing compute and avoiding redundant processing.

---

## Folder Contents

Each Python script in this directory corresponds to one stage of the pipeline. Typical structure:
1. ingest_trials.py
2. process_trials.py
3. deduplicate_registry.py
4. create_embeddings.py
5. pinecone_sync.py
6. cleanup_old.py

Each script:
- Runs independently as a Cloud Run job  
- Logs progress to a shared GCS logging directory  
- Communicates with the next stage via files in GCS

---
## Purpose of This Pipeline

This ingestion system provides:

- A **continuously updated clinical trial knowledge base**
- A **consistent preprocessing pipeline** for unstructured biomedical text  
- **Efficient vector database synchronization** without full reindexing  
- **Minimal compute cost** through incremental updates  
- **High reliability** for a production-grade clinical trial discovery assistant

---
