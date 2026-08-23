from __future__ import annotations


def normalize_email(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    if not normalized or "@" not in normalized:
        return None
    local, domain = normalized.rsplit("@", 1)
    if not local or not domain:
        return None
    return normalized


def principal_id_for_email(email: str) -> str:
    normalized = normalize_email(email)
    if normalized is None:
        raise ValueError("email must be valid")
    return f"email:{normalized}"
