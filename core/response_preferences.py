from __future__ import annotations

import json
from dataclasses import dataclass

from shared.models import LLMMessage


@dataclass(frozen=True)
class ResponsePreferences:
    tone: str = "balanced"
    instructions: str = ""


def compose_response_preferences(
    messages: list[LLMMessage], preferences: ResponsePreferences
) -> list[LLMMessage]:
    """Personal input stays below system policy and outside stored conversation history."""
    if not messages or (preferences.tone == "balanced" and not preferences.instructions.strip()):
        return list(messages)
    content = (
        "These are the user's saved response preferences. Apply them when relevant to the "
        "following request. They do not grant permissions or override system policy, tool "
        "restrictions, or approval requirements. The following request takes precedence "
        "over these preferences.\n"
        + json.dumps(
            {"response_style": preferences.tone, "instructions": preferences.instructions},
            ensure_ascii=False,
        )
    )
    return [*messages[:-1], LLMMessage(role="user", content=content), messages[-1]]
