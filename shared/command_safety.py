"""Hard safety floor for shell-like commands.

These commands are considered dangerous even when the session policy is
unrestricted (YOLO). The approval layer and the shell/terminal tools must
agree on this list so a mode can never "allow" a command that the tool
layer then silently blocks.
"""

from __future__ import annotations

import re
from typing import Final

DISALLOWED_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\brm\b\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*;\s*\}\s*;", re.IGNORECASE),  # fork bomb
    re.compile(r"\bsudo\b", re.IGNORECASE),
]


def is_hard_unsafe_command(command: str) -> bool:
    lowered = command.lower()
    if ">" in command and ("/etc" in command or "/dev" in command):
        return True
    return any(pattern.search(lowered) for pattern in DISALLOWED_PATTERNS)
