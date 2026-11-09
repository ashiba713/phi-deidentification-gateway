from deidentify import deidentify
from rehydrate import rehydrate


def process_note(note, llm_function=None):
    """
    De-identify a note, optionally send masked text to an LLM,
    and restore identifiers in the response.
    """

    masked_note, mapping = deidentify(note)

    if llm_function is None:
        return {
            "original_note": note,
            "masked_note": masked_note,
            "mapping": mapping,
            "llm_response": None,
            "restored_response": None
        }

    # Only masked text is sent to the LLM.
    llm_response = llm_function(masked_note)

    restored_response = rehydrate(llm_response, mapping)

    return {
        "original_note": note,
        "masked_note": masked_note,
        "mapping": mapping,
        "llm_response": llm_response,
        "restored_response": restored_response
    }


if __name__ == "__main__":
    note = (
        "Patient John Smith, age 92, visited on 12/05/2025. "
        "Contact john@example.com. MRN: ABC123456."
    )

    result = process_note(note)

    print("\nORIGINAL:\n", result["original_note"])
    print("\nMASKED:\n", result["masked_note"])
    print("\nLOCAL MAPPING:\n", result["mapping"])