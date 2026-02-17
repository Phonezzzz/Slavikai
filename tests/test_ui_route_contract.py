from __future__ import annotations

import re
from pathlib import Path

UI_PATH_PATTERN = re.compile(r"""/ui/api[\s\S]*?(?=['"`])""")
PLACEHOLDER_PATTERN = re.compile(r"\$\{[\s\S]*?\}|\{[^}]+\}")
ROUTE_REG_PATTERN = re.compile(
    r'add_(?:get|post|put|patch|delete)\(\s*"(?P<path>/ui/api/[^"]+)"',
)

LEGACY_FORBIDDEN_PATHS = {
    "/ui/api/session/policy",
    "/ui/api/workspace/settings",
    "/ui/api/project/command",
}


def _normalize_path(raw: str) -> str:
    path = re.sub(r"\s+", "", raw).split("?", 1)[0]
    path = PLACEHOLDER_PATTERN.sub("{var}", path)
    path = re.sub(r"/{2,}", "/", path)
    if path.endswith("/") and path != "/":
        path = path[:-1]
    return path


def _extract_ui_paths_from_dir(root: Path) -> set[str]:
    paths: set[str] = set()
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix not in {".ts", ".tsx", ".py"}:
            continue
        text = file_path.read_text(encoding="utf-8")
        for match in UI_PATH_PATTERN.finditer(text):
            paths.add(_normalize_path(match.group(0)))
    return paths


def _extract_server_routes() -> set[str]:
    sources = [
        Path("server/http/routes.py"),
        Path("server/http_api.py"),
    ]
    routes: set[str] = set()
    for source in sources:
        if not source.exists():
            continue
        text = source.read_text(encoding="utf-8")
        for match in ROUTE_REG_PATTERN.finditer(text):
            routes.add(_normalize_path(match.group("path")))
    return routes


def test_ui_endpoints_are_registered_in_server_router() -> None:
    ui_paths = _extract_ui_paths_from_dir(Path("ui/src"))
    server_routes = _extract_server_routes()
    missing = sorted(path for path in ui_paths if path not in server_routes)
    assert missing == [], f"UI uses unregistered API routes: {missing}"


def test_legacy_ui_api_paths_are_not_used_in_ui_or_ui_tests() -> None:
    sources = [
        Path("ui/src"),
        Path("tests/ui_api"),
    ]
    found: dict[str, list[str]] = {}
    for root in sources:
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in {".ts", ".tsx", ".py"}:
                continue
            text = file_path.read_text(encoding="utf-8")
            for legacy_path in LEGACY_FORBIDDEN_PATHS:
                if legacy_path in text:
                    found.setdefault(legacy_path, []).append(str(file_path))
    assert found == {}, f"Found forbidden legacy API paths: {found}"
