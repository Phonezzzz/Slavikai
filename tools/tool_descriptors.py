from __future__ import annotations

from dataclasses import dataclass

from shared.models import JSONValue


@dataclass(frozen=True)
class ToolMetadata:
    description: str
    parameters_schema: dict[str, JSONValue]


def _schema(
    properties: dict[str, JSONValue],
    *,
    required: list[str] | None = None,
) -> dict[str, JSONValue]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOL_METADATA: dict[str, ToolMetadata] = {
    "fs": ToolMetadata(
        description="List, read, or write files inside the general sandbox directory.",
        parameters_schema=_schema(
            {
                "op": {
                    "type": "string",
                    "enum": ["list", "read", "write"],
                    "description": "Filesystem operation to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Path relative to sandbox/.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content for write operations.",
                },
            },
            required=["op"],
        ),
    ),
    "web": ToolMetadata(
        description="Search the web or fetch an http/https URL when runtime web access is enabled.",
        parameters_schema=_schema(
            {
                "query": {
                    "type": "string",
                    "description": "Search query or http/https URL.",
                }
            },
            required=["query"],
        ),
    ),
    "shell": ToolMetadata(
        description="Run one approved shell command inside the configured sandbox.",
        parameters_schema=_schema(
            {
                "command": {
                    "type": "string",
                    "description": "Single shell command allowed by shell policy.",
                },
                "config_path": {
                    "type": "string",
                    "description": "Optional shell config path.",
                },
                "shell_config": {
                    "type": "object",
                    "description": "Optional runtime shell policy override from UI settings.",
                },
            },
            required=["command"],
        ),
    ),
    "project": ToolMetadata(
        description="Index or search the sandbox/project vector index.",
        parameters_schema=_schema(
            {
                "cmd": {
                    "type": "string",
                    "enum": ["index", "find"],
                    "description": "Project operation.",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Command arguments, such as path for index or query terms for find."
                    ),
                },
            },
            required=["cmd"],
        ),
    ),
    "image_analyze": ToolMetadata(
        description="Read image dimensions and metadata from base64 data or a local path.",
        parameters_schema=_schema(
            {
                "base64": {
                    "type": "string",
                    "description": "Base64-encoded image bytes.",
                },
                "path": {
                    "type": "string",
                    "description": "Local image path to analyze.",
                },
            },
        ),
    ),
    "image_generate": ToolMetadata(
        description="Generate one or more images through the configured xAI image provider.",
        parameters_schema=_schema(
            {
                "prompt": {"type": "string", "description": "Image prompt."},
                "size": {
                    "type": "string",
                    "description": "Optional size in WIDTHxHEIGHT format.",
                },
                "width": {"type": "integer", "description": "Optional image width."},
                "height": {"type": "integer", "description": "Optional image height."},
                "n": {"type": "integer", "description": "Number of images, 1..10."},
                "model": {"type": "string", "description": "Optional provider model."},
                "response_format": {
                    "type": "string",
                    "enum": ["b64_json", "url"],
                    "description": "Provider response format.",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Optional provider aspect ratio hint.",
                },
            },
            required=["prompt"],
        ),
    ),
    "tts": ToolMetadata(
        description="Generate speech audio from text through the configured OpenAI TTS provider.",
        parameters_schema=_schema(
            {
                "text": {"type": "string", "description": "Text to synthesize."},
                "voice": {"type": "string", "description": "Voice name or id."},
                "voice_id": {"type": "string", "description": "Voice id alias."},
                "format": {
                    "type": "string",
                    "enum": ["mp3", "wav"],
                    "description": "Audio output format.",
                },
                "model": {"type": "string", "description": "TTS model."},
            },
            required=["text"],
        ),
    ),
    "stt": ToolMetadata(
        description=(
            "Transcribe an audio file from sandbox/audio through the configured STT provider."
        ),
        parameters_schema=_schema(
            {
                "file_path": {
                    "type": "string",
                    "description": "Audio filename or path under sandbox/audio.",
                },
                "language": {
                    "type": "string",
                    "description": "Optional language code.",
                },
            },
            required=["file_path"],
        ),
    ),
    "workspace_list": ToolMetadata(
        description="List files and directories inside the selected workspace.",
        parameters_schema=_schema(
            {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to workspace root.",
                },
                "recursive": {"type": "boolean", "description": "Whether to walk subdirectories."},
                "max_depth": {"type": "integer", "description": "Maximum recursive depth."},
            },
        ),
    ),
    "workspace_read": ToolMetadata(
        description="Read a text file from the selected workspace.",
        parameters_schema=_schema(
            {
                "path": {
                    "type": "string",
                    "description": "File path relative to workspace root.",
                },
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 2_000_000,
                    "description": "Optional maximum UTF-8 prefix bytes to return.",
                },
            },
            required=["path"],
        ),
    ),
    "workspace_write": ToolMetadata(
        description="Overwrite or create a text file in the selected workspace.",
        parameters_schema=_schema(
            {
                "path": {"type": "string", "description": "File path relative to workspace root."},
                "content": {"type": "string", "description": "Full file content to write."},
            },
            required=["path", "content"],
        ),
    ),
    "workspace_create": ToolMetadata(
        description="Create a new text file in the selected workspace.",
        parameters_schema=_schema(
            {
                "path": {"type": "string", "description": "File path relative to workspace root."},
                "content": {"type": "string", "description": "Initial file content."},
                "overwrite": {
                    "type": "boolean",
                    "description": "Allow replacing an existing file.",
                },
            },
            required=["path"],
        ),
    ),
    "workspace_rename": ToolMetadata(
        description="Rename a workspace file or directory without overwriting an existing target.",
        parameters_schema=_schema(
            {
                "old_path": {"type": "string", "description": "Current workspace-relative path."},
                "new_path": {"type": "string", "description": "New workspace-relative path."},
            },
            required=["old_path", "new_path"],
        ),
    ),
    "workspace_move": ToolMetadata(
        description="Move a workspace file or directory without overwriting an existing target.",
        parameters_schema=_schema(
            {
                "from_path": {"type": "string", "description": "Current workspace-relative path."},
                "to_path": {"type": "string", "description": "New workspace-relative path."},
            },
            required=["from_path", "to_path"],
        ),
    ),
    "workspace_delete": ToolMetadata(
        description="Delete a workspace file or, with recursive=true, a workspace directory.",
        parameters_schema=_schema(
            {
                "path": {"type": "string", "description": "Workspace-relative path to delete."},
                "recursive": {
                    "type": "boolean",
                    "description": "Required for deleting directories.",
                },
            },
            required=["path"],
        ),
    ),
    "workspace_patch": ToolMetadata(
        description="Apply a single-file unified hunk patch to a workspace text file.",
        parameters_schema=_schema(
            {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "patch": {
                    "type": "string",
                    "description": "Single-file unified hunk patch without diff headers.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Validate patch without writing.",
                },
            },
            required=["path", "patch"],
        ),
    ),
    "workspace_run": ToolMetadata(
        description="Run a Python script from the selected workspace.",
        parameters_schema=_schema(
            {
                "path": {
                    "type": "string",
                    "description": "Python file path relative to workspace root.",
                }
            },
            required=["path"],
        ),
    ),
    "workspace_terminal_run": ToolMetadata(
        description="Run one approved terminal command in the selected workspace.",
        parameters_schema=_schema(
            {
                "command": {"type": "string", "description": "Approved command to run."},
                "cwd_mode": {
                    "type": "string",
                    "enum": ["session_root", "sandbox"],
                    "description": "Working directory mode.",
                },
            },
            required=["command"],
        ),
    ),
    "desktop_file_search": ToolMetadata(
        description="Search real host files by approximate name and modification time.",
        parameters_schema=_schema(
            {
                "root": {"type": "string", "description": "Host directory; defaults to home."},
                "query": {"type": "string", "description": "Case-insensitive name fragment."},
                "modified_after": {
                    "type": ["string", "number"],
                    "description": "ISO timestamp or unix timestamp lower bound.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            }
        ),
    ),
    "desktop_file_read": ToolMetadata(
        description="Read a UTF-8 text file from the real host after local policy checks.",
        parameters_schema=_schema(
            {"path": {"type": "string", "description": "Absolute or home-relative host path."}},
            required=["path"],
        ),
    ),
    "desktop_file_write": ToolMetadata(
        description="Atomically create or overwrite a text file on the real host.",
        parameters_schema=_schema(
            {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "overwrite": {"type": "boolean", "description": "Required for existing files."},
            },
            required=["path", "content"],
        ),
    ),
    "desktop_file_transfer": ToolMetadata(
        description="Copy, move, or rename a real host file or directory.",
        parameters_schema=_schema(
            {
                "operation": {"type": "string", "enum": ["copy", "move", "rename"]},
                "source": {"type": "string"},
                "destination": {"type": "string"},
                "overwrite": {"type": "boolean"},
            },
            required=["operation", "source", "destination"],
        ),
    ),
    "desktop_file_delete": ToolMetadata(
        description="Move a real host file or directory to the user's trash (recoverable delete).",
        parameters_schema=_schema(
            {"path": {"type": "string", "description": "Host path to move to trash."}},
            required=["path"],
        ),
    ),
    "desktop_archive_extract": ToolMetadata(
        description=(
            "Safely extract zip or tar archives on the host with traversal and size limits."
        ),
        parameters_schema=_schema(
            {
                "archive": {"type": "string"},
                "destination": {"type": "string"},
            },
            required=["archive", "destination"],
        ),
    ),
    "desktop_shell": ToolMetadata(
        description=(
            "Run one real-host command as an argv array without a shell. Shell indirection, "
            "privilege escalation, disk formatting, and bootloader commands are denied."
        ),
        parameters_schema=_schema(
            {
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "cwd": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
            },
            required=["argv"],
        ),
    ),
    "desktop_clipboard": ToolMetadata(
        description=(
            "Read, write, or clear the real desktop clipboard through the detected Wayland/X11 "
            "backend. Clipboard data may be sensitive and is policy-gated."
        ),
        parameters_schema=_schema(
            {
                "operation": {"type": "string", "enum": ["read", "write", "clear"]},
                "text": {"type": "string"},
            },
            required=["operation"],
        ),
    ),
    "desktop_system_info": ToolMetadata(
        description=(
            "Return bounded structured host OS, CPU, memory, disks, mounts, network, process, "
            "or safe session information. Prefer this over uname/free/df/ps shell commands."
        ),
        parameters_schema=_schema(
            {
                "operation": {
                    "type": "string",
                    "enum": [
                        "summary",
                        "os",
                        "cpu",
                        "memory",
                        "disks",
                        "mounts",
                        "network",
                        "processes",
                        "session",
                    ],
                },
                "query": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            required=["operation"],
        ),
    ),
    "desktop_process": ToolMetadata(
        description=(
            "Typed process management: list/find/inspect, launch without a shell, status/wait, "
            "or stop a PID using its create_time identity. Prefer this over ps/kill and GUI tools."
        ),
        parameters_schema=_schema(
            {
                "operation": {
                    "type": "string",
                    "enum": [
                        "list",
                        "find",
                        "inspect",
                        "launch",
                        "status",
                        "wait",
                        "terminate",
                        "kill",
                    ],
                },
                "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "cwd": {"type": "string"},
                "pid": {"type": "integer", "minimum": 1},
                "expected_create_time": {"type": "number"},
                "query": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
            },
            required=["operation"],
        ),
    ),
    "desktop_systemd": ToolMetadata(
        description=(
            "Typed systemd status/logs/start/stop/restart/enable/disable with structured state "
            "and post-action verification. Never construct systemctl shell commands."
        ),
        parameters_schema=_schema(
            {
                "operation": {
                    "type": "string",
                    "enum": ["status", "logs", "start", "stop", "restart", "enable", "disable"],
                },
                "unit": {"type": "string"},
                "scope": {"type": "string", "enum": ["system", "user"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            required=["operation", "unit"],
        ),
    ),
    "desktop_package": ToolMetadata(
        description=(
            "Typed Ubuntu package search/query/install/remove/metadata update. Install/remove "
            "are policy-gated and verified; system upgrade is intentionally unsupported."
        ),
        parameters_schema=_schema(
            {
                "operation": {
                    "type": "string",
                    "enum": ["search", "query", "install", "remove", "update_metadata"],
                },
                "package": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            required=["operation"],
        ),
    ),
    "desktop_session": ToolMetadata(
        description=(
            "Typed desktop/session integration for capability discovery, freedesktop DBus "
            "notifications, and verified session locking. No arbitrary DBus calls."
        ),
        parameters_schema=_schema(
            {
                "operation": {"type": "string", "enum": ["capabilities", "notify", "lock"]},
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            required=["operation"],
        ),
    ),
    "desktop_open": ToolMetadata(
        description="Open an http/https URL or existing host file through Gio/desktop opener.",
        parameters_schema=_schema(
            {"target": {"type": "string", "description": "URL or host file path."}},
            required=["target"],
        ),
    ),
    "desktop_browser": ToolMetadata(
        description=(
            "Semantic Playwright browser automation using DOM/accessibility selectors. Use this "
            "before visual GUI for web pages. Supports tabs, navigation, accessible snapshots, "
            "find/read/click/input/select/submit/wait, and verified downloads."
        ),
        parameters_schema=_schema(
            {
                "operation": {
                    "type": "string",
                    "enum": [
                        "start",
                        "open",
                        "new_tab",
                        "navigate",
                        "snapshot",
                        "read",
                        "find",
                        "click",
                        "input",
                        "select",
                        "submit",
                        "wait",
                        "tabs",
                        "switch_tab",
                        "close_tab",
                        "download",
                        "close",
                    ],
                },
                "url": {"type": "string"},
                "page_id": {"type": "string"},
                "selector_type": {
                    "type": "string",
                    "enum": ["role", "label", "text", "placeholder", "test_id", "css"],
                },
                "selector": {"type": "string"},
                "name": {"type": "string"},
                "index": {"type": "integer", "minimum": 0},
                "exact": {"type": "boolean"},
                "value": {"type": "string"},
                "state": {"type": "string", "enum": ["attached", "detached", "visible", "hidden"]},
                "destination": {"type": "string"},
                "overwrite": {"type": "boolean"},
                "headless": {"type": "boolean"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            required=["operation"],
        ),
    ),
    "desktop_gui": ToolMetadata(
        description=(
            "Last-resort desktop GUI capability. Prefer AT-SPI semantic observe/focus/invoke/"
            "set_text, then screenshot+OCR and coordinate input only when native, typed, shell, "
            "or browser DOM interfaces cannot solve the task."
        ),
        parameters_schema=_schema(
            {
                "operation": {
                    "type": "string",
                    "enum": [
                        "capabilities",
                        "windows",
                        "active_window",
                        "observe",
                        "screenshot",
                        "focus",
                        "invoke",
                        "set_text",
                        "click",
                        "type",
                        "shortcut",
                    ],
                },
                "accessible_path": {"type": "string"},
                "window_id": {"type": "string"},
                "action_name": {"type": "string"},
                "text": {"type": "string"},
                "x": {"type": "integer", "minimum": 0, "maximum": 20000},
                "y": {"type": "integer", "minimum": 0, "maximum": 20000},
                "button": {"type": "string", "enum": ["left", "middle", "right"]},
                "shortcut": {"type": "string"},
                "expected_text": {
                    "type": "string",
                    "description": "Optional text that must appear in the post-action observation.",
                },
                "visual": {"type": "boolean"},
            },
            required=["operation"],
        ),
    ),
    "desktop_verify": ToolMetadata(
        description="Deterministically verify a host result after state-changing Desktop actions.",
        parameters_schema=_schema(
            {
                "check": {
                    "type": "string",
                    "enum": ["path_exists", "path_missing", "file_contains", "process_running"],
                },
                "path": {"type": "string"},
                "expected": {"type": "string"},
                "pid": {"type": "integer"},
                "expected_create_time": {"type": "number"},
            },
            required=["check"],
        ),
    ),
}


def get_tool_metadata(name: str) -> ToolMetadata:
    return TOOL_METADATA[name]
