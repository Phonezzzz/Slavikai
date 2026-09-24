"""Hard safety floor for shell-like commands.

These commands are considered dangerous even when the session policy is
unrestricted (YOLO). The approval layer and the shell/terminal tools must
agree on this list so a mode can never "allow" a command that the tool
layer then silently blocks.
"""

from __future__ import annotations

import re
import shlex
from typing import Final

DISALLOWED_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*;\s*\}\s*;", re.IGNORECASE),  # fork bomb
    re.compile(r"\bsudo\b", re.IGNORECASE),
]


def _rm_is_recursive_force(args: list[str]) -> bool:
    if not args:
        return False
    if args[0].lower() != "rm":
        return False
    recursive = False
    force = False
    for arg in args[1:]:
        if arg == "--":
            continue
        if arg.startswith("--"):
            name = arg[2:].split("=", 1)[0].lower()
            if name in {"recursive", "r"}:
                recursive = True
            elif name == "force":
                force = True
            continue
        if arg.startswith("-") and arg != "-":
            flags = arg[1:].lower()
            if "r" in flags:
                recursive = True
            if "f" in flags:
                force = True
    return recursive and force


def is_hard_unsafe_command(command: str) -> bool:
    lowered = command.lower()
    if ">" in command and ("/etc" in command or "/dev" in command):
        return True
    if any(pattern.search(lowered) for pattern in DISALLOWED_PATTERNS):
        return True
    try:
        args = shlex.split(command)
    except ValueError:
        return False
    return _rm_is_recursive_force(args)
