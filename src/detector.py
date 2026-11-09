import re
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider


# --------------------------------------------------
# 1. Presidio setup
# --------------------------------------------------

nlp_config = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_sm"}
    ],
}

provider = NlpEngineProvider(nlp_configuration=nlp_config)
nlp_engine = provider.create_engine()

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    supported_languages=["en"]
)


# --------------------------------------------------
# 2. Custom PHI patterns
# --------------------------------------------------

CUSTOM_PATTERNS = {
    "DATE": r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    # Only ages above 89; do not flag age 88.
    "AGE_OVER_89": r"\b(?:age\s*[:=]?\s*(?:9[0-9]|1[0-9]{2})|(?:9[0-9]|1[0-9]{2})\s*(?:years?\s*old|yrs?\s*old|y/o))\b",
    "MRN": r"\b(?:MRN|Medical Record Number)\s*[:#-]?\s*[A-Z0-9-]{5,}\b",
    "HEALTH_PLAN_ID": r"\b(?:Member ID|Health Plan ID|Insurance ID)\s*[:#-]?\s*[A-Z0-9-]{5,}\b",
    "ACCOUNT_ID": r"\b(?:Account Number|Account No\.?)\s*[:#-]?\s*[A-Z0-9-]{5,}\b",
    "LICENSE_ID": r"\b(?:License Number|Licence Number|Driver'?s License)\s*[:#-]?\s*[A-Z0-9-]{5,}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    # Ten-digit phone numbers.
    "PHONE_NUMBER": r"(?<!\d)\d{10}(?!\d)",
    # Street addresses.
    "LOCATION": r"\b\d{1,5}[A-Za-z]?\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr)\b",
    # Hospital and medical facility names.
    "FACILITY": r"\b[A-Z][a-zA-Z&'-]*(?:\s+[A-Z][a-zA-Z&'-]*){1,3}\s+Hospital\b",
    "URL": r"\b(?:https?://|www\.)[^\s]+",
}


# --------------------------------------------------
# 3. Presidio detection
# --------------------------------------------------

def detect_with_presidio(text):
    results = analyzer.analyze(text=text, language="en")
    findings = []

    for result in results:
        detected_text = text[result.start:result.end]

        if result.entity_type == "DATE_TIME":
            if re.search(
                r"\bage\s*[:=]?\s*\d{1,3}\b",
                detected_text,
                re.IGNORECASE
            ):
                continue

            if re.search(
                r"\b(?:in|for|within|after|before|since)\s+\d+\s+"
                r"(?:days?|weeks?|months?|hours?|years?)\b",
                detected_text,
                re.IGNORECASE
            ):
                continue

            if re.fullmatch(
                r"\d+\s+(?:days?|weeks?|months?|hours?|years?)",
                detected_text.strip(),
                re.IGNORECASE
            ):
                continue

        findings.append({
            "start": result.start,
            "end": result.end,
            "entity_type": result.entity_type,
            "text": detected_text
        })

    return findings


# --------------------------------------------------
# 4. Regex detection
# --------------------------------------------------

def detect_custom(text):
    findings = []

    for entity_type, pattern in CUSTOM_PATTERNS.items():
        for match in re.finditer(
            pattern, text, flags=re.IGNORECASE
        ):
            findings.append({
                "start": match.start(),
                "end": match.end(),
                "entity_type": entity_type,
                "text": match.group()
            })

    return findings


# --------------------------------------------------
# 5. Load trained DistilBERT model
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "distilbert_phi"

tokenizer = None
model = None
id2label = None


def get_model_source():
    # Prefer the locally trained model during local development.
    if (MODEL_DIR / "config.json").exists():
        return str(MODEL_DIR)

    # On Streamlit Cloud, read the Hugging Face repository ID.
    model_id = os.getenv("HF_MODEL_ID")
    if not model_id:
        try:
            import streamlit as st
            model_id = st.secrets.get("HF_MODEL_ID")
        except Exception:
            model_id = None

    if not model_id:
        raise RuntimeError(
            "Trained PHI model not found. Set HF_MODEL_ID "
            "to your Hugging Face model repository."
        )
    return model_id


def load_phi_model():
    global tokenizer, model, id2label

    if model is not None:
        return

    model_source = get_model_source()
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForTokenClassification.from_pretrained(model_source)
    model.eval()

    id2label = {
        int(k): v
        for k, v in model.config.id2label.items()
    }


# --------------------------------------------------
# 6. DistilBERT inference
# --------------------------------------------------

def detect_with_model(text):
    load_phi_model()

    if model is None:
        return []

    word_matches = list(re.finditer(r"\w+|[^\w\s]", text))

    words = [match.group() for match in word_matches]

    if not words:
        return []

    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=192
    )

    word_ids = encoded.word_ids()

    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]
        )

    predictions = outputs.logits.argmax(dim=-1)[0].tolist()

    word_predictions = {}
    for token_index, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in word_predictions:
            word_predictions[word_id] = id2label[
                predictions[token_index]
            ]

    findings = []
    active_start = None
    active_end = None
    active_type = None

    def save_entity():
        if active_start is not None:
            findings.append({
                "start": active_start,
                "end": active_end,
                "entity_type": active_type,
                "text": text[active_start:active_end]
            })

    label_mapping = {
        "DATE": "DATE_TIME",
        "EMAIL": "EMAIL_ADDRESS",
        "PHONE": "PHONE_NUMBER",
        "FAX": "PHONE_NUMBER"
    }

    for word_index, match in enumerate(word_matches):
        tag = word_predictions.get(word_index, "O")

        if tag.startswith("B-"):
            save_entity()

            active_start = match.start()
            active_end = match.end()
            raw_type = tag[2:]
            active_type = label_mapping.get(raw_type, raw_type)

        elif tag.startswith("I-") and active_start is not None:
            raw_type = tag[2:]
            current_type = label_mapping.get(raw_type, raw_type)

            if current_type == active_type:
                active_end = match.end()
            else:
                save_entity()
                active_start = match.start()
                active_end = match.end()
                active_type = current_type

        else:
            save_entity()
            active_start = None
            active_end = None
            active_type = None

    save_entity()

    return findings


# --------------------------------------------------
# 7. Combined detection
# --------------------------------------------------

def detect_phi(text):
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if not text.strip():
        return []

    findings = detect_custom(text)
    findings.extend(detect_with_model(text))
    findings.extend(detect_with_presidio(text))

    cleaned = []
    for item in findings:
        value = item["text"].strip()
        entity = item["entity_type"]
        # Ignore punctuation-only detections.
        if not any(char.isalnum() for char in value):
            continue
        # Facilities must be complete hospital names.
        if entity == "FACILITY":
            if not re.fullmatch(
                r"[A-Z][A-Za-z&'-]*(?:\s+[A-Z][A-Za-z&'-]*){1,3}\s+Hospital",
                value
            ):
                continue
        cleaned.append(item)
    findings = cleaned

    filtered = []
    for item in findings:
        value = item["text"].strip()
        entity = item["entity_type"]
        # Do not mask ages of 89 or below.
        if entity == "AGE_OVER_89":
            numbers = re.findall(r"\d+", value)
            if not numbers or int(numbers[-1]) <= 89:
                continue
        # Reject generic words incorrectly classified as PHI.
        if value.casefold() in {"fax", "ssn", "member", "account", "driver"}:
            continue
        filtered.append(item)
    findings = filtered

    # Priority: custom rules > trained model > Presidio.
    priority = {
        "PHONE_NUMBER": 3,
        "AGE_OVER_89": 3,
        "MRN": 3,
        "HEALTH_PLAN_ID": 3,
        "ACCOUNT_ID": 3,
        "LICENSE_ID": 3,
        "SSN": 3,
        "IP_ADDRESS": 3,
        "LOCATION": 3,
        "FACILITY": 3,
        "DATE": 3,
        "URL": 3,
    }

    findings.sort(
        key=lambda item: (
            item["start"],
            -(item["end"] - item["start"]),
            -priority.get(item["entity_type"], 1)
        )
    )

    cleaned = []

    for item in findings:
        if not cleaned or item["start"] >= cleaned[-1]["end"]:
            cleaned.append(item)
    return cleaned


if __name__ == "__main__":
    sample = (
        "Patient John Smith, age 92, visited on 12/05/2025. "
        "Contact john@example.com. MRN: ABC123456. "
        "Follow-up required in 3 days."
    )

    for finding in detect_phi(sample):
        print(finding)
