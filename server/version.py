from __future__ import annotations

import os
from typing import Final

VERSION: Final[str] = "0.1.0-beta.1"


def build_sha() -> str | None:
    raw = os.getenv("SLAVIK_BUILD_SHA", "").strip()
    return raw or None
