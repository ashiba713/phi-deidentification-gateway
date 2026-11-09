def rehydrate(response, mapping):
    """
    Restore original identifiers in an LLM response.
    """
    if not isinstance(response, str):
        raise TypeError("Response must be a string.")

    restored = response

    for placeholder, original in sorted(
        mapping.items(),
        key=lambda item: len(item[0]),
        reverse=True
    ):
        restored = restored.replace(placeholder, original)

    return restored