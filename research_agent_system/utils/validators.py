import re
import json

from pydantic import BaseModel
from pydantic import ValidationError


def extract_json(
    text: str,
    model: type[BaseModel] | None = None
):

    if not text or not text.strip():
        raise ValueError(
            "Empty LLM response"
        )

    # =====================================
    # REMOVE MARKDOWN
    # =====================================

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    text = text.strip()

    # =====================================
    # FIND JSON OBJECT
    # =====================================

    start = text.find("{")

    if start == -1:
        raise ValueError(
            f"No JSON found.\nRaw:\n{text[:500]}"
        )

    depth = 0
    end = -1

    for i, ch in enumerate(text[start:], start):

        if ch == "{":
            depth += 1

        elif ch == "}":

            depth -= 1

            if depth == 0:
                end = i + 1
                break

    if end == -1:
        raise ValueError(
            f"Unclosed JSON.\nRaw:\n{text[:500]}"
        )

    json_text = text[start:end]

    # =====================================
    # LOAD JSON
    # =====================================

    try:

        data = json.loads(json_text)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"JSON parse failed: {e}\n\n"
            f"RAW:\n{json_text[:1000]}"
        )

    # =====================================
    # PYDANTIC VALIDATION
    # =====================================

    if model:

        try:
            return model(**data)

        except ValidationError as e:

            print(
                "\n⚠️ Pydantic validation issue:\n"
            )

            print(e)

            valid = {

                k: v

                for k, v in data.items()

                if k in model.model_fields
            }

            return model.model_construct(
                **valid
            )

    return data