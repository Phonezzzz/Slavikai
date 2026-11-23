"""Single-file unified hunk patch application (shared by sandbox and bridge lanes).

Extracted from ``tools/workspace_tools.py`` in E03-T04 so the sandbox
``workspace_patch`` tool and the Owner Host Bridge ``host.fs.patch``
capability share one patch implementation (ADR-0013 §12: one host
boundary, one implementation — no copy, no daemon→tools import).

Contract: a single-file unified hunk patch **without** ``diff --git`` /
``---`` / ``+++`` headers, with at least one ``@@`` hunk. Context and
removal lines must match the original exactly; any mismatch raises
``ValueError`` and nothing is applied.
"""

from __future__ import annotations

from collections.abc import Iterator


def validate_patch_contract(diff_text: str) -> str | None:
    """Return a contract-violation message, or None when the patch is well-formed."""
    lines = diff_text.splitlines()
    if any(line.startswith(prefix) for line in lines for prefix in ("diff --git ", "--- ", "+++ ")):
        return (
            "workspace_patch поддерживает только single-file hunk patch "
            "(без diff --git/---/+++ заголовков)."
        )
    if not any(line.startswith("@@") for line in lines):
        return "Patch должен содержать хотя бы один hunk (@@ ... @@)."
    return None


def apply_unified_patch(original: str, diff_text: str) -> str:
    """
    Простейшее применение unified diff к одному файлу.
    Без поддержки переименований/бинарных файлов.
    """
    lines = original.splitlines(keepends=True)
    diff_lines = diff_text.splitlines()

    def parse_hunk_header(header: str) -> tuple[int, int]:
        # @@ -l,s +l,s @@
        parts = header.strip("@ ").split()
        minus = parts[0]  # -l,s
        plus = parts[1]  # +l,s
        start_old = int(minus.split(",")[0].lstrip("-"))
        start_new = int(plus.split(",")[0].lstrip("+"))
        return start_old, start_new

    new_lines: list[str] = []
    idx = 0  # current position in original
    it: Iterator[str] = iter(diff_lines)
    for line in it:
        if not line.startswith("@@"):
            continue
        start_old, _ = parse_hunk_header(line)
        # copy unchanged section before hunk
        target_index = start_old - 1
        if target_index < idx or target_index > len(lines):
            raise ValueError("Hunk не соответствует файлу.")
        new_lines.extend(lines[idx:target_index])
        idx = target_index
        # process hunk lines
        for hunk_line in it:
            if hunk_line.startswith("@@"):
                # next hunk, push back by restarting loop
                # but we already consumed line; use recursion-like approach
                # easiest: process by resetting iterator with current line
                it = _prepend(hunk_line, it)
                break
            if not hunk_line:
                continue
            prefix = hunk_line[0]
            content = hunk_line[1:]
            if prefix == " ":
                if idx >= len(lines) or lines[idx].rstrip("\n") != content.rstrip("\n"):
                    raise ValueError("Hunk не совпадает с оригиналом.")
                new_lines.append(lines[idx])
                idx += 1
            elif prefix == "-":
                if idx >= len(lines) or lines[idx].rstrip("\n") != content.rstrip("\n"):
                    raise ValueError("Hunk не совпадает с оригиналом.")
                idx += 1  # skip (delete)
            elif prefix == "+":
                new_lines.append(content + ("\n" if content and not content.endswith("\n") else ""))
            elif prefix == "\\":
                # "\ No newline" -- ignore
                continue
            else:
                raise ValueError("Неизвестный префикс патча.")
    # append remaining original
    new_lines.extend(lines[idx:])
    return "".join(new_lines)


def _prepend(first: str, iterator: Iterator[str]) -> Iterator[str]:
    yield first
    yield from iterator
