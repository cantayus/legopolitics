from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from legopolitics.exceptions import StructuredOutputError

T = TypeVar("T", bound=BaseModel)


def extract_json(text: str) -> Any:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        if not match:
            raise StructuredOutputError("No JSON object found in model response") from None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(f"Invalid JSON model response: {exc}") from exc


def parse_model(text: str, schema: type[T]) -> T:
    try:
        return schema.model_validate(extract_json(text))
    except ValidationError as exc:
        raise StructuredOutputError(str(exc)) from exc
