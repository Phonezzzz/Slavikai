#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

ROUTE_PATTERN = re.compile(
    r'add_(?P<method>get|post|put|patch|delete)\(\s*"(?P<path>/ui/api/[^"]+)"\s*,\s*[A-Za-z0-9_]+\.(?P<handler>[A-Za-z0-9_]+)',
)
UI_PATH_PATTERN = re.compile(r"/ui/api/[A-Za-z0-9_./${}-]+")
PLACEHOLDER_PATTERN = re.compile(r"\$\{[^}]+\}|\{[^}]+\}")


def normalize(path: str) -> str:
    value = path.strip().split("?", 1)[0]
    value = PLACEHOLDER_PATTERN.sub("{var}", value)
    value = re.sub(r"/{2,}", "/", value)
    if value.endswith("/") and value != "/":
        value = value[:-1]
    return value


def route_map() -> dict[str, dict[str, str]]:
    text = Path("server/http/routes.py").read_text(encoding="utf-8")
    mapping: dict[str, dict[str, str]] = {}
    for match in ROUTE_PATTERN.finditer(text):
        path = normalize(match.group("path"))
        mapping[path] = {
            "method": match.group("method").upper(),
            "handler": match.group("handler"),
        }
    return mapping


def consumers() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    roots = [Path("ui/src"), Path("tests/ui_api")]
    for root in roots:
        for file_path in root.rglob("*"):
            if not file_path.is_file() or file_path.suffix not in {".ts", ".tsx", ".py"}:
                continue
            text = file_path.read_text(encoding="utf-8")
            for match in UI_PATH_PATTERN.finditer(text):
                path = normalize(match.group(0))
                found.setdefault(path, []).append(str(file_path))
    for path in list(found):
        found[path] = sorted(set(found[path]))
    return found


def main() -> None:
    routes = route_map()
    refs = consumers()
    payload: dict[str, dict[str, object]] = {}
    for path, meta in sorted(routes.items()):
        payload[path] = {
            "method": meta["method"],
            "handler": meta["handler"],
            "consumers": refs.get(path, []),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
