import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detector import detect_phi, detect_with_presidio


def normalize_type(entity_type):
    aliases = {
        "DATE_TIME": "DATE",
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
    }
    return aliases.get(entity_type, entity_type)


def evaluate_detector(records, detector):
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    leaked_notes = 0
    errors = []

    for record in records:
        text = record["text"]
        expected = {
            (e["start"], e["end"], e["entity_type"])
            for e in record["entities"]
        }

        predicted = {
            (
                item["start"],
                item["end"],
                normalize_type(item["entity_type"])
            )
            for item in detector(text)
        }

        missing = expected - predicted
        extra = predicted - expected
        if missing or extra:
            errors.append({
                "note_id": record["id"],
                "missing": [
                    {
                        "text": text[start:end],
                        "type": entity_type
                    }
                    for start, end, entity_type in sorted(missing)
                ],
                "false_positives": [
                    {
                        "text": text[start:end],
                        "type": entity_type
                    }
                    for start, end, entity_type in sorted(extra)
                ]
            })

        for entity in expected | predicted:
            entity_type = entity[2]

            if entity in expected and entity in predicted:
                counts[entity_type]["tp"] += 1
            elif entity in predicted:
                counts[entity_type]["fp"] += 1
            else:
                counts[entity_type]["fn"] += 1

        if expected - predicted:
            leaked_notes += 1

    total_tp = sum(x["tp"] for x in counts.values())
    total_fp = sum(x["fp"] for x in counts.values())
    total_fn = sum(x["fn"] for x in counts.values())

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    report = {}

    for entity_type, values in sorted(counts.items()):
        tp, fp, fn = values["tp"], values["fp"], values["fn"]

        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        score = 2 * p * r / (p + r) if p + r else 0

        report[entity_type] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(score, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

    return {
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "leaked_notes": leaked_notes,
            "total_notes": len(records),
            "leak_rate": round(leaked_notes / len(records), 4)
        },
        "per_entity": report,
        "errors": errors
    }


def main():
    with open(ROOT / "data" / "synthetic_test.json", encoding="utf-8") as f:
        records = json.load(f)

    print("Evaluating Presidio baseline...")
    baseline = evaluate_detector(records, detect_with_presidio)

    print("Evaluating combined detector...")
    hybrid = evaluate_detector(records, detect_phi)

    results = {
        "test_notes": len(records),
        "presidio_baseline": baseline,
        "hybrid_detector": hybrid
    }

    output = ROOT / "results" / "metrics.json"
    output.parent.mkdir(exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nPRESIDIO BASELINE")
    print(json.dumps(baseline, indent=2))

    print("\nHYBRID DETECTOR")
    print(json.dumps(hybrid, indent=2))

    print(f"\nMetrics saved to {output}")


if __name__ == "__main__":
    main()
