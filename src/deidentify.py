from detector import detect_phi


def deidentify(text):
    """
    Replace detected identifiers with consistent placeholders.
    Returns masked text and a local mapping.
    """
    findings = detect_phi(text)

    mapping = {}
    counters = {}
    replacements = []

    for item in findings:
        original = item["text"]
        entity = item["entity_type"].upper()

        # Reuse the same placeholder for repeated values.
        key = (entity, original.casefold())

        if key not in mapping:
            counters[entity] = counters.get(entity, 0) + 1
            placeholder = f"[{entity}_{counters[entity]}]"
            mapping[key] = {
                "placeholder": placeholder,
                "original": original
            }

        placeholder = mapping[key]["placeholder"]

        replacements.append({
            "start": item["start"],
            "end": item["end"],
            "placeholder": placeholder
        })

    masked_text = text

    # Replace from right to left to preserve character positions.
    for item in sorted(replacements, key=lambda x: x["start"], reverse=True):
        masked_text = (
            masked_text[:item["start"]]
            + item["placeholder"]
            + masked_text[item["end"]:]
        )

    # Convert mapping into a serializable dictionary.
    serializable_mapping = {
        value["placeholder"]: value["original"]
        for value in mapping.values()
    }

    return masked_text, serializable_mapping