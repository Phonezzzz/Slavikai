from __future__ import annotations

import importlib
import importlib.util
import os
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from shared.models import JSONValue, ToolRequest, ToolResult

MAX_ACCESSIBLE_NODES = 500
MAX_ACCESSIBLE_DEPTH = 8
MAX_OCR_ITEMS = 300
MAX_SCREENSHOTS = 30


class DesktopGuiTool:
    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        artifact_root: Path | None = None,
        runner: Callable[[Sequence[str], int], subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self._env = dict(environ if environ is not None else os.environ)
        self._artifact_root = (
            artifact_root or Path.home() / ".cache" / "slavikai" / "gui"
        ).expanduser()
        self._runner = runner or _run
        self._atspi = PyAtspiAdapter() if PyAtspiAdapter.available() else None

    def handle(self, request: ToolRequest) -> ToolResult:
        operation = _operation(request, default="observe")
        try:
            if operation == "capabilities":
                return ToolResult.success(
                    {"output": "Detected GUI capabilities.", **self._capabilities()}
                )
            if operation == "windows":
                windows = self._windows()
                return ToolResult.success(
                    {
                        "output": f"Returned {len(windows)} GUI window(s).",
                        "operation": operation,
                        "items": windows,
                        "truncated": len(windows) >= MAX_ACCESSIBLE_NODES,
                    }
                )
            if operation == "active_window":
                active = self._active_window()
                if active is None:
                    return ToolResult.failure("Active window information is unavailable.")
                return ToolResult.success(
                    {"output": "Active GUI window detected.", "operation": operation, **active}
                )
            if operation == "observe":
                visual = request.args.get("visual") is True
                if not visual and self._atspi is not None:
                    tree, truncated = self._atspi.observe()
                    return ToolResult.success(
                        {
                            "output": "Semantic AT-SPI observation.",
                            "operation": operation,
                            "mechanism": "at-spi",
                            "tree": tree,
                            "truncated": truncated,
                        }
                    )
                return self._visual_observation(operation=operation)
            if operation == "screenshot":
                return self._visual_observation(operation=operation)
            if operation == "focus":
                accessible_path = _optional_string(request, "accessible_path")
                window_id = _optional_string(request, "window_id")
                if accessible_path is not None and self._atspi is not None:
                    self._atspi.focus(accessible_path)
                    return self._post_action(request, "focus", "at-spi")
                if window_id is not None and self._env.get("DISPLAY") and shutil.which("wmctrl"):
                    completed = self._runner(("wmctrl", "-ia", window_id), 10)
                    if completed.returncode != 0:
                        return ToolResult.failure(
                            completed.stderr.strip() or "Window focus failed."
                        )
                    return self._post_action(request, "focus", "x11-wmctrl")
                return ToolResult.failure("No usable semantic/X11 window focus backend.")
            if operation == "invoke":
                path = _required_string(request, "accessible_path")
                if self._atspi is None:
                    return ToolResult.failure("AT-SPI is unavailable for semantic invoke.")
                action_name = _optional_string(request, "action_name")
                self._atspi.invoke(path, action_name=action_name)
                return self._post_action(request, operation, "at-spi")
            if operation == "set_text":
                path = _required_string(request, "accessible_path")
                text = _required_string(request, "text", allow_empty=True)
                if self._atspi is None:
                    return ToolResult.failure("AT-SPI is unavailable for semantic text input.")
                self._atspi.set_text(path, text)
                return self._post_action(request, operation, "at-spi")
            if operation == "click":
                x = _coordinate(request, "x")
                y = _coordinate(request, "y")
                button = _optional_string(request, "button") or "left"
                mechanism = self._visual_click(x=x, y=y, button=button)
                return self._post_action(request, operation, mechanism)
            if operation == "type":
                text = _required_string(request, "text", allow_empty=True)
                mechanism = self._visual_type(text)
                return self._post_action(request, operation, mechanism)
            if operation == "shortcut":
                shortcut = _required_string(request, "shortcut")
                if re.fullmatch(r"[A-Za-z0-9_+:-]{1,80}", shortcut) is None:
                    return ToolResult.failure("Invalid shortcut syntax.")
                if not self._env.get("DISPLAY") or not shutil.which("xdotool"):
                    return ToolResult.failure(
                        "Keyboard shortcuts require xdotool in an X11 session; "
                        "no safe Wayland mapping is available."
                    )
                completed = self._runner(("xdotool", "key", "--clearmodifiers", shortcut), 10)
                if completed.returncode != 0:
                    return ToolResult.failure(completed.stderr.strip() or "GUI shortcut failed.")
                return self._post_action(request, operation, "x11-xdotool")
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            return ToolResult.failure(f"GUI operation failed: {exc}")
        return ToolResult.failure(
            "operation must be capabilities|windows|active_window|observe|screenshot|focus|"
            "invoke|set_text|click|type|shortcut"
        )

    def close(self) -> None:
        self._prune_screenshots()

    def _capabilities(self) -> dict[str, JSONValue]:
        session_type = self._env.get("XDG_SESSION_TYPE", "").lower()
        screenshot_backend = self._screenshot_backend()
        input_backend: str | None = None
        if self._env.get("DISPLAY") and shutil.which("xdotool"):
            input_backend = "x11-xdotool"
        elif self._env.get("WAYLAND_DISPLAY") and shutil.which("ydotool"):
            input_backend = "wayland-ydotool"
        return {
            "session_type": session_type,
            "desktop": self._env.get("XDG_CURRENT_DESKTOP", ""),
            "display_available": bool(self._env.get("DISPLAY") or self._env.get("WAYLAND_DISPLAY")),
            "accessibility": self._atspi is not None,
            "screenshot_backend": screenshot_backend,
            "input_backend": input_backend,
            "ocr": bool(shutil.which("tesseract")),
        }

    def _windows(self) -> list[JSONValue]:
        if self._atspi is not None:
            return self._atspi.windows()
        if self._env.get("DISPLAY") and shutil.which("wmctrl"):
            completed = self._runner(("wmctrl", "-lx"), 10)
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or "wmctrl failed")
            items: list[JSONValue] = []
            for line in completed.stdout.splitlines()[:MAX_ACCESSIBLE_NODES]:
                fields = line.split(None, 4)
                if len(fields) < 5:
                    continue
                items.append(
                    {
                        "window_id": fields[0],
                        "desktop": fields[1],
                        "class": fields[3],
                        "title": fields[4],
                        "mechanism": "x11-wmctrl",
                    }
                )
            return items
        raise RuntimeError("Window enumeration unavailable in this desktop session.")

    def _active_window(self) -> dict[str, JSONValue] | None:
        if self._atspi is not None:
            return self._atspi.active_window()
        if self._env.get("DISPLAY") and shutil.which("xdotool"):
            identifier = self._runner(("xdotool", "getactivewindow"), 10)
            if identifier.returncode != 0:
                return None
            window_id = identifier.stdout.strip()
            name = self._runner(("xdotool", "getwindowname", window_id), 10)
            if name.returncode != 0:
                return None
            return {
                "window_id": window_id,
                "title": name.stdout.strip(),
                "mechanism": "x11-xdotool",
            }
        return None

    def _visual_observation(self, *, operation: str) -> ToolResult:
        path, backend = self._capture_screen()
        ocr, truncated = self._ocr(path)
        return ToolResult.success(
            {
                "output": "Visual GUI observation captured with OCR.",
                "operation": operation,
                "mechanism": backend,
                "screenshot_path": str(path),
                "ocr": ocr,
                "truncated": truncated,
            }
        )

    def _post_action(
        self,
        request: ToolRequest,
        operation: str,
        mechanism: str,
    ) -> ToolResult:
        observation = self._visual_observation(operation="observe")
        if not observation.ok:
            return ToolResult.failure(
                f"GUI {operation} ran but post-action observation failed: {observation.error}"
            )
        data: dict[str, JSONValue] = {
            "output": f"GUI {operation} completed with post-action observation.",
            "operation": operation,
            "mechanism": mechanism,
            "requires_followup_observation": True,
            "post_observation": observation.data,
        }
        expected_text = _optional_string(request, "expected_text")
        if expected_text is not None:
            observed_text = " ".join(_observation_text(observation.data))
            if expected_text.casefold() not in observed_text.casefold():
                return ToolResult.failure(
                    f"GUI {operation} ran, but expected post-state text was not observed.",
                    meta={"post_observation": observation.data},
                )
            data["verified"] = True
            data["requires_followup_observation"] = False
            data["expected_text"] = expected_text
        return ToolResult.success(data)

    def _capture_screen(self) -> tuple[Path, str]:
        backend = self._screenshot_backend()
        if backend is None:
            raise RuntimeError("No screenshot backend available for this desktop session.")
        self._artifact_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self._artifact_root, 0o700)
        path = self._artifact_root / f"screen-{time.time_ns()}-{uuid.uuid4().hex[:8]}.png"
        argv: tuple[str, ...]
        if backend == "gnome-shell-dbus":
            argv = (
                "gdbus",
                "call",
                "--session",
                "--dest",
                "org.gnome.Shell.Screenshot",
                "--object-path",
                "/org/gnome/Shell/Screenshot",
                "--method",
                "org.gnome.Shell.Screenshot.Screenshot",
                "false",
                "false",
                str(path),
            )
        elif backend == "wayland-grim":
            argv = ("grim", str(path))
        elif backend == "x11-gnome-screenshot":
            argv = ("gnome-screenshot", "-f", str(path))
        else:
            argv = ("import", "-window", "root", str(path))
        completed = self._runner(argv, 20)
        if completed.returncode != 0 or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(completed.stderr.strip() or f"Screenshot backend {backend} failed")
        self._prune_screenshots()
        return path, backend

    def _screenshot_backend(self) -> str | None:
        desktop = self._env.get("XDG_CURRENT_DESKTOP", "").lower()
        if (
            self._env.get("WAYLAND_DISPLAY")
            and "gnome" in desktop
            and self._env.get("DBUS_SESSION_BUS_ADDRESS")
            and shutil.which("gdbus")
        ):
            return "gnome-shell-dbus"
        if self._env.get("WAYLAND_DISPLAY") and shutil.which("grim"):
            return "wayland-grim"
        if self._env.get("DISPLAY") and shutil.which("gnome-screenshot"):
            return "x11-gnome-screenshot"
        if self._env.get("DISPLAY") and shutil.which("import"):
            return "x11-imagemagick"
        return None

    def _ocr(self, path: Path) -> tuple[list[JSONValue], bool]:
        if not shutil.which("tesseract"):
            return [], False
        completed = self._runner(("tesseract", str(path), "stdout", "tsv"), 30)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "tesseract failed")
        items: list[JSONValue] = []
        lines = completed.stdout.splitlines()[1:]
        for line in lines:
            fields = line.split("\t")
            if len(fields) < 12 or not fields[11].strip():
                continue
            try:
                item: dict[str, JSONValue] = {
                    "text": fields[11][:500],
                    "confidence": float(fields[10]),
                    "x": int(fields[6]),
                    "y": int(fields[7]),
                    "width": int(fields[8]),
                    "height": int(fields[9]),
                }
            except ValueError:
                continue
            items.append(item)
            if len(items) >= MAX_OCR_ITEMS:
                break
        return items, len(items) >= MAX_OCR_ITEMS

    def _visual_click(self, *, x: int, y: int, button: str) -> str:
        if button not in {"left", "middle", "right"}:
            raise ValueError("button must be left|middle|right")
        if self._env.get("DISPLAY") and shutil.which("xdotool"):
            button_number = {"left": "1", "middle": "2", "right": "3"}[button]
            completed = self._runner(
                ("xdotool", "mousemove", "--sync", str(x), str(y), "click", button_number),
                10,
            )
            mechanism = "x11-xdotool"
        elif self._env.get("WAYLAND_DISPLAY") and shutil.which("ydotool"):
            if button != "left":
                raise RuntimeError("Wayland ydotool backend currently supports left click only.")
            move = self._runner(
                ("ydotool", "mousemove", "--absolute", "-x", str(x), "-y", str(y)), 10
            )
            if move.returncode != 0:
                raise RuntimeError(move.stderr.strip() or "ydotool mouse move failed")
            completed = self._runner(("ydotool", "click", "0xC0"), 10)
            mechanism = "wayland-ydotool"
        else:
            raise RuntimeError("No GUI pointer backend available.")
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "GUI click failed")
        return mechanism

    def _visual_type(self, text: str) -> str:
        if self._env.get("DISPLAY") and shutil.which("xdotool"):
            completed = self._runner(("xdotool", "type", "--delay", "0", "--", text), 20)
            mechanism = "x11-xdotool"
        elif self._env.get("WAYLAND_DISPLAY") and shutil.which("ydotool"):
            completed = self._runner(("ydotool", "type", "--key-delay", "0", "--", text), 20)
            mechanism = "wayland-ydotool"
        else:
            raise RuntimeError("No GUI keyboard backend available.")
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "GUI typing failed")
        return mechanism

    def _prune_screenshots(self) -> None:
        if not self._artifact_root.is_dir():
            return
        paths = sorted(
            self._artifact_root.glob("screen-*.png"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for path in paths[MAX_SCREENSHOTS:]:
            try:
                path.unlink()
            except OSError:
                continue


class PyAtspiAdapter:
    def __init__(self) -> None:
        self._module = importlib.import_module("pyatspi")

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("pyatspi") is not None

    def windows(self) -> list[JSONValue]:
        desktop = self._desktop()
        windows: list[JSONValue] = []
        for app_index, app in enumerate(_children(desktop)):
            for window_index, window in enumerate(_children(app)):
                windows.append(
                    {
                        "accessible_path": f"{app_index}/{window_index}",
                        "application": _name(app),
                        "title": _name(window),
                        "role": _role(window),
                        "active": self._has_state(window, "STATE_ACTIVE"),
                        "mechanism": "at-spi",
                    }
                )
                if len(windows) >= MAX_ACCESSIBLE_NODES:
                    return windows
        return windows

    def active_window(self) -> dict[str, JSONValue] | None:
        for item in self.windows():
            if isinstance(item, Mapping) and item.get("active") is True:
                return {str(key): value for key, value in item.items()}
        return None

    def observe(self) -> tuple[list[JSONValue], bool]:
        nodes: list[JSONValue] = []
        truncated = self._walk(self._desktop(), path="", depth=0, nodes=nodes)
        return nodes, truncated

    def focus(self, path: str) -> None:
        node = self._resolve(path)
        component = _call(node, "queryComponent")
        focused = _call(component, "grabFocus")
        if focused is False:
            raise RuntimeError("AT-SPI focus request was rejected.")

    def invoke(self, path: str, *, action_name: str | None) -> None:
        node = self._resolve(path)
        action = _call(node, "queryAction")
        count_raw = _attribute(action, "nActions")
        count = int(count_raw) if isinstance(count_raw, int) else 0
        if count <= 0:
            raise RuntimeError("Accessible object has no actions.")
        selected = 0
        if action_name is not None:
            selected = -1
            for index in range(count):
                name = str(_call(action, "getName", index))
                if name.casefold() == action_name.casefold():
                    selected = index
                    break
            if selected < 0:
                raise RuntimeError(f"Accessible action not found: {action_name}")
        result = _call(action, "doAction", selected)
        if result is False:
            raise RuntimeError("AT-SPI action was rejected.")

    def set_text(self, path: str, text: str) -> None:
        node = self._resolve(path)
        editable = _call(node, "queryEditableText")
        result = _call(editable, "setTextContents", text)
        if result is False:
            raise RuntimeError("AT-SPI text update was rejected.")

    def _desktop(self) -> object:
        registry = _attribute(self._module, "Registry")
        return _call(registry, "getDesktop", 0)

    def _resolve(self, path: str) -> object:
        if re.fullmatch(r"\d+(?:/\d+)*", path) is None:
            raise ValueError("accessible_path must contain child indexes such as 0/1/2")
        node = self._desktop()
        for raw_index in path.split("/"):
            children = _children(node)
            index = int(raw_index)
            if index >= len(children):
                raise ValueError(f"Accessible path is stale: {path}")
            node = children[index]
        return node

    def _walk(
        self,
        node: object,
        *,
        path: str,
        depth: int,
        nodes: list[JSONValue],
    ) -> bool:
        if len(nodes) >= MAX_ACCESSIBLE_NODES:
            return True
        if path:
            nodes.append(
                {
                    "accessible_path": path,
                    "name": _name(node),
                    "role": _role(node),
                    "focused": self._has_state(node, "STATE_FOCUSED"),
                    "enabled": self._has_state(node, "STATE_ENABLED"),
                }
            )
        if depth >= MAX_ACCESSIBLE_DEPTH:
            return bool(_children(node))
        for index, child in enumerate(_children(node)):
            child_path = f"{path}/{index}" if path else str(index)
            if self._walk(child, path=child_path, depth=depth + 1, nodes=nodes):
                return True
        return False

    def _has_state(self, node: object, constant_name: str) -> bool:
        constant = _attribute(self._module, constant_name)
        state_set = _call(node, "getState")
        return bool(_call(state_set, "contains", constant))


def _children(node: object) -> list[object]:
    count_raw = _attribute(node, "childCount")
    count = int(count_raw) if isinstance(count_raw, int) else 0
    children: list[object] = []
    for index in range(max(0, count)):
        child = _call(node, "getChildAtIndex", index)
        if child is not None:
            children.append(child)
    return children


def _name(node: object) -> str:
    value = _attribute(node, "name")
    return value if isinstance(value, str) else ""


def _role(node: object) -> str:
    try:
        value = _call(node, "getRoleName")
    except (AttributeError, RuntimeError):
        return ""
    return value if isinstance(value, str) else str(value)


def _attribute(target: object, name: str) -> object:
    try:
        return getattr(target, name)
    except AttributeError as exc:
        raise RuntimeError(f"AT-SPI attribute unavailable: {name}") from exc


def _call(target: object, name: str, *args: object) -> object:
    method = _attribute(target, name)
    if not callable(method):
        raise RuntimeError(f"AT-SPI member is not callable: {name}")
    return method(*args)


def _run(argv: Sequence[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _operation(request: ToolRequest, *, default: str) -> str:
    raw = request.args.get("operation")
    return raw.strip().lower() if isinstance(raw, str) and raw.strip() else default


def _optional_string(request: ToolRequest, key: str) -> str | None:
    raw = request.args.get(key)
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def _required_string(request: ToolRequest, key: str, *, allow_empty: bool = False) -> str:
    raw = request.args.get(key)
    if not isinstance(raw, str) or (not allow_empty and not raw.strip()):
        raise ValueError(f"{key} is required")
    return raw if allow_empty else raw.strip()


def _coordinate(request: ToolRequest, key: str) -> int:
    raw = request.args.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw <= 20_000:
        raise ValueError(f"{key} must be an integer between 0 and 20000")
    return raw


def _observation_text(value: JSONValue) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [part for nested in value.values() for part in _observation_text(nested)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [part for nested in value for part in _observation_text(nested)]
    return []
