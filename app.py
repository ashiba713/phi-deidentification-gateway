
import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st

# Make modules inside src/ importable
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from deidentify import deidentify
from rehydrate import rehydrate


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="PHI De-identification Gateway",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 PHI De-identification Gateway")
st.caption("Privacy-preserving clinical text processing using AI")

st.warning(
    "Use synthetic notes for testing. Do not enter real patient information. "
    "Only masked text is sent to the external LLM."
)


# ---------------- GEMINI API ----------------

def call_gemini(masked_text, api_key, task):
    """Send masked text to Gemini with automatic retry."""
    model = "gemini-2.5-flash"

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent"
    )

    prompt = f"""You are assisting with clinical text processing.
Task:{task}
Preserve placeholders such as [PERSON_1] exactly.
Do not guess the identities behind placeholders.
Masked clinical note:{masked_text}"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024
        }
    }

    retry_delays = (4, 8)
    for attempt in range(len(retry_delays) + 1):
        response = requests.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=60
        )

        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt < len(retry_delays):
                time.sleep(retry_delays[attempt])
                continue
        break

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API error ({response.status_code}): "
            f"{response.text[:500]}"
        )

    result = response.json()

    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("The LLM returned an unexpected response format.")


# ---------------- SIDEBAR ----------------

st.sidebar.header("Configuration")

api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    help="Your API key is used for this request and is not saved by this application."
)

task = st.sidebar.selectbox(
    "Choose LLM task",
    [
        "Summarize the clinical note in 3 clear bullet points.",
        "Extract the main symptoms, diagnosis and treatment.",
        "Generate a concise clinical handover summary.",
        "Identify the important follow-up actions."
    ]
)

st.sidebar.info(
    "The placeholder-to-original mapping stays inside this application. "
    "Only the masked note is sent to Gemini."
)


# ---------------- NOTE INPUT ----------------

st.subheader("1. Enter Clinical Note")

default_note = (
    "Patient John Smith, aged 92, visited Green Valley Hospital "
    "on 12/05/2025. Contact: 9876543210. "
    "Email: john@example.com. MRN: AB12345678. "
    "The patient reported chest pain for 3 days."
)

original_note = st.text_area(
    "Clinical note",
    value=default_note,
    height=180,
    placeholder="Enter a synthetic clinical note..."
)


# ---------------- PROCESSING ----------------

if st.button("🔍 De-identify and Process", type="primary", use_container_width=True):

    if not original_note.strip():
        st.error("Please enter a clinical note.")
        st.stop()

    if not api_key.strip():
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()

    try:
        with st.spinner("Detecting and masking identifiers..."):

            # Step 1: Mask PHI locally
            masked_text, mapping = deidentify(original_note)

        st.success("PHI detection and masking completed.")

        st.subheader("2. De-identified Clinical Note")
        st.text_area(
            "Masked text",
            value=masked_text,
            height=180,
            disabled=True
        )

        with st.expander("View local placeholder mapping"):
            st.json(mapping)
            st.caption(
                "This mapping is kept locally and is never included "
                "in the Gemini request."
            )

        # Step 2: Send masked text only to Gemini
        with st.spinner("Sending masked text to Gemini..."):

            llm_response = call_gemini(
                masked_text=masked_text,
                api_key=api_key,
                task=task
            )

        st.success("LLM processing completed.")

        st.subheader("3. LLM Response (Masked)")
        st.text_area(
            "LLM output before restoration",
            value=llm_response,
            height=180,
            disabled=True
        )

        # Step 3: Restore identifiers locally
        restored_response = rehydrate(
            llm_response,
            mapping
        )

        st.subheader("4. Rehydrated Response")

        st.text_area(
            "Restored output",
            value=restored_response,
            height=180,
            disabled=True
        )

        st.success("Round-trip processing completed successfully.")

        st.download_button(
            "Download restored response",
            data=restored_response,
            file_name="restored_clinical_response.txt",
            mime="text/plain"
        )

    except requests.exceptions.Timeout:
        st.error("The Gemini request timed out. Please try again.")

    except requests.exceptions.RequestException as error:
        st.error(f"Network error: {error}")

    except Exception as error:
        st.error(f"Processing failed: {error}")


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "PHI De-identification Gateway | "
    "DistilBERT + Presidio + Regex | "
    "Local mapping and rehydration"
)
