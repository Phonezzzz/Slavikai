from llm.types import LLMUsage
from shared.models import JSONValue


def parse_reported_usage(raw: JSONValue) -> LLMUsage | None:
    """Полные сообщённые counters либо unknown; это не проверка billing или quota."""
    if not isinstance(raw, dict):
        return None
    counts: list[int] = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        counts.append(value)
    return LLMUsage(prompt_tokens=counts[0], completion_tokens=counts[1], total_tokens=counts[2])
