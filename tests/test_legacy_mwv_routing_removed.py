"""E01-T08: legacy classifier/MWV production routing удалён (gate).

Проверяет отсутствие production-пути от user request к legacy MWV:
- ни один production-модуль не импортирует classifier/ManagerRuntime/AgentMWVMixin;
- удалённые модули отсутствуют на диске;
- у Agent нет MWV-поверхности, default runtime_mode — "ask";
- respond/respond_stream в legacy режимах (act/plan) идут в обычный chat loop;
- skill deprecated/ambiguous по-прежнему обрабатываются в chat-пути;
- root-gate решает только режим (auto — gated, ask/desktop — нет);
- resume старого MWV-чекпоинта отклоняется run_plan_runner.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

from core.agent import Agent
from core.skills.index import SkillIndex
from core.skills.models import SkillEntry, SkillManifest
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from server.http.common.ui_runtime import _run_plan_runner
from server.http.handlers.ui_chat import _request_requires_root_gate
from shared.models import LLMMessage
from tests.report_utils import extract_report_block
from tests.ui_api.fakes import DummyAgent, _create_client, _enter_act_mode

REPO_ROOT = Path(__file__).resolve().parents[1]
PROD_DIRS = ("core", "server", "shared", "llm", "tools", "memory", "config")
DELETED_MODULES = (
    "core/agent_mwv.py",
    "core/mwv/routing.py",
    "core/mwv/manager.py",
    "core/mwv/coding_skill.py",
    "core/mwv/coding_task.py",
    "core/mwv/single_attempt.py",
    "core/mwv/worker.py",
    "core/mwv/interfaces.py",
)
BANNED_MODULES = {
    "core.mwv.routing",
    "core.mwv.manager",
    "core.mwv.coding_skill",
    "core.mwv.coding_task",
    "core.mwv.single_attempt",
    "core.mwv.worker",
    "core.mwv.interfaces",
    "core.agent_mwv",
}
BANNED_NAMES = {
    "classify_request",
    "ManagerRuntime",
    "MWVRunResult",
    "AgentMWVMixin",
    "MWVWorkerCheckpoint",
}
REMOVED_AGENT_ATTRS = (
    "_run_mwv_flow",
    "run_task_packet",
    "resume_task_packet",
    "validate_mwv_approval",
    "cancel_mwv_approval",
    "_execute_mwv_plan_step",
    "_mwv_worker_runner",
    "_mwv_task_builder",
    "_build_mwv_context",
    "_format_mwv_response",
    "_mwv_outcome_label",
    "_mwv_next_steps",
    "_mwv_checkpoints",
    "last_mwv_result",
)


class ScriptedBrain(Brain):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self.text)


def _make_agent(tmp_path: Path, text: str = "chat") -> tuple[Agent, ScriptedBrain]:
    brain = ScriptedBrain(text)
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]
    return agent, brain


def _skill_entry(
    skill_id: str,
    patterns: list[str],
    *,
    deprecated: bool = False,
    replaced_by: str | None = None,
) -> SkillEntry:
    return SkillEntry(
        id=skill_id,
        version="1.0.0",
        title=f"{skill_id} title",
        entrypoints=["mwv"],
        patterns=patterns,
        requires=[],
        risk="low",
        tests=[],
        path=f"skills/{skill_id}/skill.md",
        content_hash=f"hash-{skill_id}",
        deprecated=deprecated,
        replaced_by=replaced_by,
    )


def _make_skill_index(entries: list[SkillEntry]) -> SkillIndex:
    return SkillIndex(SkillManifest(manifest_version=1, skills=entries))


def test_deleted_mwv_modules_are_gone() -> None:
    for rel in DELETED_MODULES:
        assert not (REPO_ROOT / rel).exists(), f"{rel} должен быть удалён (E01-T08)"


def test_no_production_module_imports_legacy_mwv() -> None:
    violations: list[str] = []
    for dirname in PROD_DIRS:
        for path in sorted((REPO_ROOT / dirname).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module in BANNED_MODULES:
                        violations.append(
                            f"{path.relative_to(REPO_ROOT)}: from {module} import ..."
                        )
                    for alias in node.names:
                        if alias.name in BANNED_NAMES:
                            violations.append(
                                f"{path.relative_to(REPO_ROOT)}: from {module} import {alias.name}"
                            )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in BANNED_MODULES or alias.name.startswith("core.agent_mwv"):
                            violations.append(f"{path.relative_to(REPO_ROOT)}: import {alias.name}")
    assert violations == []


def test_agent_has_no_mwv_surface_and_defaults_to_ask(tmp_path: Path) -> None:
    agent, _ = _make_agent(tmp_path)
    assert agent.runtime_mode == "ask"
    for attr in REMOVED_AGENT_ATTRS:
        assert not hasattr(agent, attr), f"Agent.{attr} должен быть удалён (E01-T08)"
    assert agent.session_context_busy() is False
    agent._pending_chat_approval = object()  # type: ignore[assignment]
    assert agent.session_context_busy() is True
    agent._pending_chat_approval = None
    assert agent.session_context_busy() is False


def test_set_runtime_state_keeps_act_for_canonical_loop(tmp_path: Path) -> None:
    # E01-T08, отклонение от брифа: "act" НЕ нормализуется в "ask":
    # plan_execute.py требует policy.mode == "act" для шагов canonical loop.
    # Гарантия "нет маршрута в MWV" — в fallthrough respond/respond_stream.
    agent, _ = _make_agent(tmp_path)
    agent.set_runtime_state(
        mode="act", active_plan=None, active_task=None, enforce_plan_guard=False
    )
    assert agent.runtime_mode == "act"
    agent.set_runtime_state(
        mode="ask", active_plan=None, active_task=None, enforce_plan_guard=False
    )
    assert agent.runtime_mode == "ask"


def test_legacy_modes_fall_back_to_chat_loop(tmp_path: Path) -> None:
    for mode in ("act", "plan"):
        agent, brain = _make_agent(tmp_path)
        agent.runtime_mode = mode
        response = agent.respond([LLMMessage(role="user", content="поправь баг в коде")])
        assert response.text.startswith("chat")
        assert brain.calls == 1
        assert not hasattr(agent, "_run_mwv_flow")
        report = extract_report_block(response.text)
        assert report["route"] == "chat"


def test_legacy_mode_stream_falls_back_to_chat_loop(tmp_path: Path) -> None:
    from llm.stream_model import TextDelta

    agent, brain = _make_agent(tmp_path)
    agent.runtime_mode = "act"
    events = list(agent.respond_stream([LLMMessage(role="user", content="git commit -m x")]))
    text = "".join(event.text for event in events if isinstance(event, TextDelta))
    assert text.startswith("chat")
    assert brain.calls == 1


def test_fallthrough_deprecated_skill_still_blocks(tmp_path: Path) -> None:
    agent, brain = _make_agent(tmp_path)
    agent.runtime_mode = "act"
    agent.skill_index = _make_skill_index(
        [_skill_entry("legacy", ["legacy"], deprecated=True, replaced_by="alpha")]
    )
    response = agent.respond([LLMMessage(role="user", content="legacy request")]).text
    lowered = response.lower()
    assert "что случилось" in lowered
    assert "deprecated" in lowered
    assert brain.calls == 0
    report = extract_report_block(response)
    assert report["route"] == "blocked"
    assert report["stop_reason_code"] == "BLOCKED_SKILL_DEPRECATED"


def test_fallthrough_ambiguous_skill_still_returns_decision(tmp_path: Path) -> None:
    agent, brain = _make_agent(tmp_path)
    agent.runtime_mode = "act"
    agent.skill_index = _make_skill_index(
        [_skill_entry("alpha", ["alpha"]), _skill_entry("beta", ["alpha"])]
    )
    response = agent.respond([LLMMessage(role="user", content="alpha request")]).text
    payload = json.loads(response)
    assert payload["reason"] == "ambiguous_skill"
    assert any(option["action"] == "select_skill" for option in payload["options"])
    assert brain.calls == 0


def test_root_gate_no_longer_fires() -> None:
    kwargs: dict[str, object] = {
        "llm_messages": [],
        "safe_mode": False,
        "skill_index": None,
    }
    # E01-T08: триггером gate был legacy classifier; без него gate не решает
    # ничего — ни для исполняющих режимов, ни для action-текста, ни для
    # legacy plan/act. Auto preset сам управляет бюджетами и approval'ами.
    assert _request_requires_root_gate(mode="auto", content="поправь баг в коде", **kwargs) is False
    assert _request_requires_root_gate(mode="auto", content="привет", **kwargs) is False
    assert _request_requires_root_gate(mode="ask", content="поправь баг в коде", **kwargs) is False
    assert (
        _request_requires_root_gate(mode="desktop", content="поправь баг в коде", **kwargs) is False
    )
    assert _request_requires_root_gate(mode="auto", content="/trace", **kwargs) is False
    assert _request_requires_root_gate(mode="plan", content="привет", **kwargs) is False


def test_legacy_mwv_checkpoint_resume_is_rejected(tmp_path: Path) -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_payload = await (await client.get("/ui/api/status")).json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _enter_act_mode(client, session_id, goal="legacy resume guard")
            hub = client.server.app["ui_hub"]
            # Дожидаемся завершения быстрого canonical loop.
            for _ in range(100):
                workflow = await hub.get_session_workflow(session_id)
                task = workflow.get("active_task")
                if not isinstance(task, dict) or task.get("status") != "running":
                    break
                await asyncio.sleep(0.05)
            workflow = await hub.get_session_workflow(session_id)
            plan = workflow.get("active_plan")
            task = workflow.get("active_task")
            assert isinstance(plan, dict) and isinstance(task, dict)
            assert isinstance(plan.get("plan_id"), str)
            assert isinstance(task.get("task_id"), str)
            # Подменяем execution на старый MWV-чекпоинт: его id нет
            # в loop_checkpoints, resume обязан вернуть False без эффектов.
            task = dict(task)
            task["status"] = "running"
            task["execution"] = {
                "runner": "mwv_packet_runner",
                "status": "waiting_approval",
                "checkpoint_id": "legacy-mwv-1",
            }
            await hub.set_session_workflow(
                session_id, mode="act", active_plan=plan, active_task=task
            )
            result = await _run_plan_runner(
                app=client.server.app,
                session_id=session_id,
                plan_id=str(plan["plan_id"]),
                task_id=str(task["task_id"]),
                resume_checkpoint_id="legacy-mwv-1",
            )
            assert result is False
            after = await hub.get_session_workflow(session_id)
            after_task = after.get("active_task")
            assert isinstance(after_task, dict)
            assert after_task.get("status") == "running"
            assert after_task.get("execution", {}).get("checkpoint_id") == "legacy-mwv-1"
        finally:
            await client.close()

    asyncio.run(run())
