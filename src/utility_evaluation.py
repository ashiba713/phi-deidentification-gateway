
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deidentify import deidentify
from rehydrate import rehydrate

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"

INPUT_FILE = ROOT / "data" / "synthetic_test.json"
OUTPUT_FILE = ROOT / "results" / "utility_evaluation.json"

TASK = (
    "Summarize the clinical note in 3 concise bullet points. "
    "Preserve all clinically relevant information, including symptoms, "
    "diagnoses, treatments and follow-up details. Do not invent facts."
)


def call_gemini(text):
    """Call Gemini with the supplied note text."""

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{MODEL}:generateContent"
    )

    prompt = f"""
You are assisting with clinical text summarization.

{TASK}

If the note contains placeholders such as [PERSON_1] or [DATE_1],
preserve them exactly. Do not guess their original values.

Clinical note:
{text}
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024
        }
    }

    for attempt in range(3):
        response = requests.post(
            url,
            params={"key": API_KEY},
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                raise RuntimeError("Unexpected Gemini response format.")

        if response.status_code in (429, 500, 503) and attempt < 2:
            time.sleep(2 ** (attempt + 1))
            continue

        raise RuntimeError(
            f"Gemini API error {response.status_code}: "
            f"{response.text[:400]}"
        )

    raise RuntimeError("Gemini remained unavailable after retries.")


def main():
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it in PowerShell first."
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        notes = json.load(file)

    selected_notes = notes[:5]
    results = []

    print(f"Running utility evaluation on {len(selected_notes)} synthetic notes.")

    for index, note in enumerate(selected_notes, start=1):
        original_text = note["text"]

        print(f"\nProcessing note {index}/5 (ID: {note['id']})")

        # Mask identifiers locally
        masked_text, mapping = deidentify(original_text)

        print("  Calling Gemini with original synthetic note...")
        original_answer = call_gemini(original_text)

        print("  Calling Gemini with masked note...")
        masked_answer = call_gemini(masked_text)

        # Restore identifiers locally
        restored_answer = rehydrate(masked_answer, mapping)

        results.append({
            "note_id": note["id"],
            "original_note": original_text,
            "masked_note": masked_text,
            "original_answer": original_answer,
            "masked_answer": masked_answer,
            "restored_answer": restored_answer,
            "placeholder_mapping": mapping
        })

        print("  Completed.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\nUTILITY EVALUATION COMPLETED")
    print(f"Notes evaluated: {len(results)}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
