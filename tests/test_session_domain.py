from __future__ import annotations

import asyncio

from server.ui_hub import UIHub
from server.ui_session_storage import (
    PersistedSession,
    legacy_session_from_domain_records,
    split_persisted_session_domains,
)
from shared.models import JSONValue
from shared.session_domain import (
    ChatThread,
    WorkspaceSession,
    legacy_messages_for_lane,
    normalize_message_lane,
    normalize_policy_profile,
    normalize_session_mode,
)


def test_legacy_messages_for_lane_splits_chat_and_workspace() -> None:
    messages: list[dict[str, JSONValue]] = [
        {"role": "user", "content": "hello", "lane": "chat", "created_at": "t1"},
        {"role": "assistant", "content": "edit", "lane": "workspace", "trace_id": "tr"},
        {"role": "assistant", "content": "implicit chat"},
    ]

    chat_messages = legacy_messages_for_lane(messages, lane="chat")
    workspace_messages = legacy_messages_for_lane(messages, lane="workspace")

    assert [item.content for item in chat_messages] == ["hello", "implicit chat"]
    assert [item.content for item in workspace_messages] == ["edit"]
    assert workspace_messages[0].trace_id == "tr"


def test_session_domain_normalizers_default_to_safe_values() -> None:
    assert normalize_message_lane("workspace") == "workspace"
    assert normalize_message_lane("other") == "chat"
    assert normalize_policy_profile("yolo") == "yolo"
    assert normalize_policy_profile("unknown") == "sandbox"
    assert normalize_session_mode("auto") == "auto"
    assert normalize_session_mode("desktop") == "ask"
    assert normalize_session_mode("unknown") == "ask"


def test_persisted_session_domain_adapter_strips_legacy_lane_from_records() -> None:
    session = PersistedSession(
        session_id="s1",
        principal_id="owner",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:01+00:00",
        status="ok",
        decision={"id": "d1"},
        messages=[
            {"message_id": "c1", "role": "user", "content": "chat", "lane": "chat"},
            {
                "message_id": "w1",
                "role": "assistant",
                "content": "workspace",
                "lane": "workspace",
            },
        ],
        model_provider="local",
        model_id="llama",
        workspace_root="/tmp/project",
        policy_profile="yolo",
        mode="auto",
        active_plan={"id": "plan"},
    )

    records = split_persisted_session_domains(session)

    assert records.chat.thread_id == "s1"
    assert records.chat.messages == [{"message_id": "c1", "role": "user", "content": "chat"}]
    assert records.workspace.session_id == "s1"
    assert records.workspace.messages == [
        {"message_id": "w1", "role": "assistant", "content": "workspace"}
    ]
    assert records.workspace.workspace_root == "/tmp/project"
    assert records.workspace.policy_profile == "yolo"
    assert records.workspace.mode == "auto"
    assert records.workspace.active_plan == {"id": "plan"}

    restored = legacy_session_from_domain_records(records)
    assert restored.messages == session.messages
    assert restored.workspace_root == "/tmp/project"


def test_ui_hub_exposes_chat_thread_and_workspace_session_domains() -> None:
    asyncio.run(_assert_ui_hub_domain_views())


async def _assert_ui_hub_domain_views() -> None:
    hub = UIHub()
    session_id = await hub.create_session("owner")
    await hub.set_session_model(session_id, provider="local", model_id="llama")
    await hub.set_workspace_root(session_id, "/tmp/project")
    await hub.set_session_policy(session_id, profile="yolo")
    await hub.set_session_workflow(
        session_id,
        mode="auto",
        active_plan={"id": "plan"},
        active_task={"id": "task"},
        auto_state={
            "run_id": "auto-1",
            "status": "planning",
            "goal": "goal",
            "started_at": "t1",
            "updated_at": "t2",
        },
    )
    await hub.append_message(
        session_id,
        hub.create_message(role="user", content="chat msg", lane="chat"),
        lane="chat",
    )
    await hub.append_message(
        session_id,
        hub.create_message(role="assistant", content="workspace msg", lane="workspace"),
        lane="workspace",
    )

    chat_thread = await hub.get_chat_thread(session_id)
    workspace_session = await hub.get_workspace_session_domain(session_id)

    assert isinstance(chat_thread, ChatThread)
    assert chat_thread.thread_id == session_id
    assert chat_thread.model_provider == "local"
    assert chat_thread.model_id == "llama"
    assert chat_thread.workspace_session_id == session_id
    assert [message.content for message in chat_thread.messages] == ["chat msg"]

    assert isinstance(workspace_session, WorkspaceSession)
    assert workspace_session.session_id == session_id
    assert workspace_session.root == "/tmp/project"
    assert workspace_session.policy_profile == "yolo"
    assert workspace_session.mode == "auto"
    assert workspace_session.active_plan == {"id": "plan"}
    assert workspace_session.active_task == {"id": "task"}
    assert workspace_session.auto_state is not None
    assert workspace_session.auto_state["run_id"] == "auto-1"
    assert workspace_session.auto_state["status"] == "planning"
    assert [message.content for message in workspace_session.messages] == ["workspace msg"]

    await hub.set_session_workflow(session_id, mode="desktop")
    desktop_workspace = await hub.get_workspace_session_domain(session_id)

    assert desktop_workspace is not None
    assert desktop_workspace.mode == "ask"
