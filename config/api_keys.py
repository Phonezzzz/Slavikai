from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

DEFAULT_API_KEYS_PATH = Path(__file__).resolve().parent / "api_keys.json"


class ApiKeyStoreError(RuntimeError):
    """Raised when the on-disk API key store cannot be read safely."""


def load_api_keys(*, path: Path = DEFAULT_API_KEYS_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiKeyStoreError(f"Не удалось прочитать хранилище API-ключей: {exc}") from exc
    if not isinstance(raw, dict):
        raise ApiKeyStoreError("Хранилище API-ключей должно содержать JSON-объект.")

    api_keys: dict[str, str] = {}
    for provider, value in raw.items():
        if not isinstance(provider, str) or not isinstance(value, str):
            raise ApiKeyStoreError("Хранилище API-ключей содержит некорректную запись.")
        normalized_provider = provider.strip().lower()
        normalized_key = value.strip()
        if normalized_provider and normalized_key:
            api_keys[normalized_provider] = normalized_key
    return api_keys


def save_api_keys(api_keys: dict[str, str], *, path: Path = DEFAULT_API_KEYS_PATH) -> None:
    normalized = {
        provider.strip().lower(): value.strip()
        for provider, value in api_keys.items()
        if provider.strip() and value.strip()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            json.dump(normalized, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    except OSError as exc:
        raise ApiKeyStoreError(f"Не удалось сохранить API-ключи: {exc}") from exc
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
