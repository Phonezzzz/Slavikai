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
                }
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
            {"path": {"type": "string", "description": "File path relative to workspace root."}},
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
}


def get_tool_metadata(name: str) -> ToolMetadata:
    return TOOL_METADATA[name]
