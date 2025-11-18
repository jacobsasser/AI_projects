import os
import re
import json
import pandas as pd
import numpy as np
import logging
from google.cloud import storage

# Cloud Run config
INPUT_BUCKET = "clinical-trial-data-pipeline"
INPUT_PREFIX = "raw/"
OUTPUT_PREFIX = "processed/"
LOG_PATH = "logs/preprocessing.log"

# Initialize GCS client
client = storage.Client()

# Logging setup
os.makedirs("/tmp/logs", exist_ok=True)
log_file = "/tmp/logs/preprocessing.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.info("=== Starting Cloud Run preprocessing - FLATTEN ONLY ===")

import pandas as pd
import re
import numpy as np
import logging

def preprocess_trials(df):
    # ───── Flatten Nested Fields ─────────────────────────────
    def safe_flatten(val):
        if isinstance(val, list):
            if all(isinstance(item, str) for item in val):
                return "; ".join(val)
            elif all(isinstance(item, dict) for item in val):
                return "; ".join(
                    [", ".join(f"{k}: {str(v)}" for k, v in item.items() if v is not None) for item in val]
                )
            else:
                return str(val)
        elif isinstance(val, dict):
            return ", ".join(f"{k}: {str(v)}" for k, v in val.items() if v is not None)
        return val

    nested_cols = [
        "protocolSection.conditionsModule.conditions",
        "protocolSection.armsInterventionsModule.interventions",
        "protocolSection.outcomesModule.primaryOutcomes",
        "protocolSection.outcomesModule.secondaryOutcomes"
    ]

    for col in nested_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_flatten)

    df.replace(["None", "NA", "", "null", "N/A"], np.nan, inplace=True)

    def patch_partial_dates(series):
        return series.apply(lambda x: f"{x}-01" if isinstance(x, str) and re.fullmatch(r"\d{4}-\d{2}", x) else x)

    date_cols = [
        "protocolSection.statusModule.startDateStruct.date",
        "protocolSection.statusModule.completionDateStruct.date"
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = patch_partial_dates(df[col])
            df[col] = pd.to_datetime(df[col], errors="coerce")

    def parse_age(age_str):
        if pd.isna(age_str): return np.nan
        try:
            num, unit = age_str.split()
            num = float(num)
            if "month" in unit.lower(): return num / 12
            elif "week" in unit.lower(): return num / 52
            elif "day" in unit.lower(): return num / 365
            return num
        except:
            return np.nan

    age_cols = [
        "protocolSection.eligibilityModule.minimumAge",
        "protocolSection.eligibilityModule.maximumAge"
    ]
    for col in age_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_age)

    if "protocolSection.identificationModule.officialTitle" in df.columns:
        df["protocolSection.identificationModule.officialTitle"] = df["protocolSection.identificationModule.officialTitle"].fillna(
            df["protocolSection.identificationModule.briefTitle"]
        )

    if "protocolSection.conditionsModule.conditions" in df.columns:
        df = df[df["protocolSection.conditionsModule.conditions"].notna()]

    if "protocolSection.outcomesModule.primaryOutcomes" in df.columns and "protocolSection.outcomesModule.secondaryOutcomes" in df.columns:
        mask = df["protocolSection.outcomesModule.primaryOutcomes"].isna() & df["protocolSection.outcomesModule.secondaryOutcomes"].notna()
        df.loc[mask, "protocolSection.outcomesModule.primaryOutcomes"] = df.loc[mask, "protocolSection.outcomesModule.secondaryOutcomes"]
        df.loc[mask, "protocolSection.outcomesModule.secondaryOutcomes"] = None
        df = df[df["protocolSection.outcomesModule.primaryOutcomes"].notna()]

    df["protocolSection.eligibilityModule.minimumAge"] = df["protocolSection.eligibilityModule.minimumAge"].fillna(0)
    df["protocolSection.eligibilityModule.maximumAge"] = df["protocolSection.eligibilityModule.maximumAge"].fillna(120)

    if "protocolSection.descriptionModule.detailedDescription" in df.columns:
        df["protocolSection.descriptionModule.detailedDescription"] = df["protocolSection.descriptionModule.detailedDescription"].fillna(
            df["protocolSection.descriptionModule.briefSummary"]
        )

    df["protocolSection.designModule.enrollmentInfo.count"] = df["protocolSection.designModule.enrollmentInfo.count"].clip(upper=20000)
    df["protocolSection.eligibilityModule.minimumAge"] = df["protocolSection.eligibilityModule.minimumAge"].clip(upper=65)
    df["protocolSection.eligibilityModule.maximumAge"] = df["protocolSection.eligibilityModule.maximumAge"].clip(upper=120)

    def normalize_whitespace(text):
        text = str(text).replace('≥', '>=').replace('≤', '<=').replace('±', '+/-')
        text = re.sub(r'[\r\n\t•–—]', ' ', text)
        text = re.sub(r'\s*\*\s*', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r' +([.,;:?!])', r'\1', text)
        return text.strip()

    def remove_boilerplate(text):
        patterns = [
            r'for more information.*?visit.*?clinicaltrials\.gov.*',
            r'please contact.*?[\w\.\-]+@[\w\.\-]+\.\w+',
            r'last updated.*?\d{4}',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()

    def split_inclusion_exclusion(text):
        if pd.isna(text): return "", ""
        match = re.search(r'(Inclusion Criteria[:\-]?\s*)(.*?)(Exclusion Criteria[:\-]?\s*)(.*)', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(2).strip(), match.group(4).strip()
        return re.sub(r'Inclusion Criteria[:\-]?\s*', '', text, flags=re.IGNORECASE).strip(), ""

    def clean(text, typ='general'):
        if pd.isna(text): return "" if typ != 'eligibility' else ("", "")
        text = normalize_whitespace(text)
        text = remove_boilerplate(text)
        return split_inclusion_exclusion(text) if typ == 'eligibility' else text

    df["summary"] = df["protocolSection.descriptionModule.briefSummary"].apply(clean)
    df["description"] = df["protocolSection.descriptionModule.detailedDescription"].apply(clean)
    inc_exc = df["protocolSection.eligibilityModule.eligibilityCriteria"].apply(lambda x: clean(x, typ="eligibility"))
    df["eligibility_inclusion"], df["eligibility_exclusion"] = zip(*inc_exc)

    def summarize_locations(locations):
        if not isinstance(locations, list): return ""
        loc_strings = []
        for loc in locations:
            try:
                city = loc.get("city", "")
                state = loc.get("state", "")
                country = loc.get("country", "")
                status = loc.get("status", "")
                parts = [city, state, country]
                summary = ", ".join(filter(None, parts))
                if status:
                    summary += f" ({status})"
                loc_strings.append(summary)
            except Exception as e:
                logging.warning(f"[Location Summary] Failed: {e}")
        return "; ".join(loc_strings)

    df["locations"] = df["protocolSection.contactsLocationsModule.locations"].apply(summarize_locations)

    rename_cols = {
        "protocolSection.identificationModule.nctId": "nct_id",
        "protocolSection.identificationModule.briefTitle": "title",
        "protocolSection.identificationModule.officialTitle": "official_title",
        "protocolSection.statusModule.overallStatus": "status",
        "protocolSection.statusModule.startDateStruct.date": "start_date",
        "protocolSection.statusModule.completionDateStruct.date": "completion_date",
        "protocolSection.conditionsModule.conditions": "conditions",
        "protocolSection.designModule.studyType": "study_type",
        "protocolSection.designModule.enrollmentInfo.count": "enrollment_count",
        "protocolSection.eligibilityModule.sex": "eligible_sex",
        "protocolSection.eligibilityModule.minimumAge": "minimum_age",
        "protocolSection.eligibilityModule.maximumAge": "maximum_age",
        "protocolSection.armsInterventionsModule.interventions": "interventions",
        "protocolSection.outcomesModule.primaryOutcomes": "primary_outcomes",
        "protocolSection.outcomesModule.secondaryOutcomes": "secondary_outcomes"
    }
    df.rename(columns=rename_cols, inplace=True)

    drop_cols = [
        "protocolSection.descriptionModule.briefSummary",
        "protocolSection.descriptionModule.detailedDescription",
        "protocolSection.eligibilityModule.eligibilityCriteria",
        "protocolSection.contactsLocationsModule.locations"
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    return df

def run(request):
    bucket = client.bucket(INPUT_BUCKET)
    blobs = list(bucket.list_blobs(prefix=INPUT_PREFIX))

    processed_count = 0

    for blob in blobs:
        if not blob.name.endswith(".jsonl"):
            continue

        try:
            logging.info(f"Reading: {blob.name}")
            lines = blob.download_as_text().splitlines()
            records = [json.loads(line) for line in lines if line.strip()]
            df = pd.json_normalize(records)
            logging.info(f"Loaded {len(df)} records from {blob.name}")

            # Preprocess this blob
            df_cleaned = preprocess_trials(df)

            # Generate clean output name
            base_name = os.path.basename(blob.name).replace(".jsonl", ".parquet")
            output_blob_path = os.path.join(OUTPUT_PREFIX, base_name)
            local_parquet_path = f"/tmp/{base_name}"

            # Save locally and upload
            df_cleaned.to_parquet(local_parquet_path, index=False)
            bucket.blob(output_blob_path).upload_from_filename(local_parquet_path)
            logging.info(f"Uploaded cleaned file to: {output_blob_path}")

            processed_count += 1

        except Exception as e:
            logging.error(f"Failed to process blob {blob.name}: {e}")

    # Upload logs
    bucket.blob(LOG_PATH).upload_from_filename(log_file)

    return f"Processed {processed_count} files into individual Parquet files. Logs uploaded.", 200