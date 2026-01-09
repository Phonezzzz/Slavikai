from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from memory.memory_companion_store import MemoryCompanionStore
from shared.memory_companion_models import (
    BlockedReason,
    InteractionKind,
    ToolInteractionLog,
    ToolStatus,
)
from shared.models import LLMMessage


class SimpleBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="ok")


def _get_tool_log(store: MemoryCompanionStore, tool_name: str) -> ToolInteractionLog:
    for item in store.get_recent(100):
        if item.interaction_kind != InteractionKind.TOOL:
            continue
        if isinstance(item, ToolInteractionLog) and item.tool_name == tool_name:
            return item
    raise AssertionError(f"ToolInteractionLog not found for tool_name={tool_name!r}")


def test_commandlane_logs_tool_not_registered(tmp_path: Path) -> None:
    db_path = tmp_path / "memory_companion.db"
    agent = Agent(brain=SimpleBrain(), memory_companion_db_path=str(db_path))

    _ = agent.handle_tool_command("/unknown")

    store = MemoryCompanionStore(db_path)
    log = _get_tool_log(store, "unknown")
    assert log.tool_status == ToolStatus.BLOCKED
    assert log.blocked_reason == BlockedReason.TOOL_NOT_REGISTERED


def test_commandlane_logs_sandbox_violation(tmp_path: Path) -> None:
    db_path = tmp_path / "memory_companion.db"
    agent = Agent(
        brain=SimpleBrain(),
        memory_companion_db_path=str(db_path),
        enable_tools={
            "fs": True,
            "shell": False,
            "web": False,
            "project": False,
            "image_analyze": False,
            "image_generate": False,
            "tts": False,
            "stt": False,
            "workspace_run": False,
            "safe_mode": False,
        },
    )

    _ = agent.handle_tool_command("/fs read ../etc/passwd")

    store = MemoryCompanionStore(db_path)
    log = _get_tool_log(store, "fs")
    assert log.tool_status == ToolStatus.BLOCKED
    assert log.blocked_reason == BlockedReason.SANDBOX_VIOLATION


def test_commandlane_logs_validation_error(tmp_path: Path) -> None:
    db_path = tmp_path / "memory_companion.db"
    agent = Agent(
        brain=SimpleBrain(),
        memory_companion_db_path=str(db_path),
        enable_tools={
            "fs": True,
            "shell": True,
            "web": False,
            "project": True,
            "image_analyze": False,
            "image_generate": False,
            "tts": False,
            "stt": False,
            "workspace_run": True,
            "safe_mode": False,
        },
    )

    _ = agent.handle_tool_command("/sh ls && whoami")

    store = MemoryCompanionStore(db_path)
    log = _get_tool_log(store, "shell")
    assert log.tool_status == ToolStatus.BLOCKED
    assert log.blocked_reason == BlockedReason.VALIDATION_ERROR


def test_commandlane_logs_safe_mode_blocked(tmp_path: Path) -> None:
    db_path = tmp_path / "memory_companion.db"
    agent = Agent(
        brain=SimpleBrain(),
        memory_companion_db_path=str(db_path),
        enable_tools={
            "fs": True,
            "shell": True,
            "web": True,
            "project": True,
            "image_analyze": False,
            "image_generate": False,
            "tts": False,
            "stt": False,
            "workspace_run": True,
            "safe_mode": True,
        },
    )

    _ = agent.handle_tool_command("/sh ls")

    store = MemoryCompanionStore(db_path)
    log = _get_tool_log(store, "shell")
    assert log.tool_status == ToolStatus.BLOCKED
    assert log.blocked_reason == BlockedReason.APPROVAL_REQUIRED


def test_commandlane_logs_tool_disabled(tmp_path: Path) -> None:
    db_path = tmp_path / "memory_companion.db"
    agent = Agent(
        brain=SimpleBrain(),
        memory_companion_db_path=str(db_path),
        enable_tools={
            "fs": True,
            "shell": False,
            "web": False,
            "project": True,
            "image_analyze": False,
            "image_generate": False,
            "tts": False,
            "stt": False,
            "workspace_run": True,
            "safe_mode": False,
        },
    )

    _ = agent.handle_tool_command("/sh ls")

    store = MemoryCompanionStore(db_path)
    log = _get_tool_log(store, "shell")
    assert log.tool_status == ToolStatus.BLOCKED
    assert log.blocked_reason == BlockedReason.TOOL_DISABLED
