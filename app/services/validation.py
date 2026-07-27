"""Input validation helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationError:
    message: str
    code: str = "validation_error"


def validate_news_text(
    text: object,
    *,
    min_length: int,
    max_length: int,
) -> tuple[str | None, ValidationError | None]:
    if text is None:
        return None, ValidationError("Field 'text' is required.")
    if not isinstance(text, str):
        return None, ValidationError("Field 'text' must be a string.")

    cleaned = text.strip()
    if not cleaned:
        return None, ValidationError("Text cannot be empty.")
    if len(cleaned) < min_length:
        return None, ValidationError(
            f"Text is too short. Provide at least {min_length} characters "
            "(a full headline or short paragraph works best)."
        )
    if len(cleaned) > max_length:
        return None, ValidationError(
            f"Text exceeds the {max_length:,}-character limit."
        )
    return cleaned, None
