def normalize_language(language: str | None) -> str | None:
    return language.lower().strip() if language else None
