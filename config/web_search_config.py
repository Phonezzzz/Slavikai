from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class WebSearchConfig:
    provider: Literal["serper"] = "serper"
    api_key: str | None = None
    top_k: int = 5
    timeout: int = 10
    max_bytes: int = 200_000
