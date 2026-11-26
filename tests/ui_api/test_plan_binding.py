from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from .fakes import DummyAgent, _create_client


def _binding(plan):
    return {key: plan[key] for key in ("plan_id", "plan_hash", "plan_revision")}


async def _draft(client):
    status = await (await client.get("/ui/api/status")).json()
    session_id = status["session_id"]
    headers = {"X-Slavik-Session": session_id}
    assert (await client.post("/ui/api/mode", headers=headers, json={"mode": "plan"})).status == 200
    response = await client.post(
        "/ui/api/plan/draft", headers=headers, json={"goal": "Bound consent"}
    )
    assert response.status == 200
    return session_id, (await response.json())["active_plan"]


@pytest.mark.parametrize("field", ["missing", "plan_id", "plan_hash", "plan_revision"])
def test_approve_requires_the_displayed_plan_binding(field):
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            body = _binding(plan)
            if field != "missing":
                body[field] = 99 if field == "plan_revision" else "another-plan"
            response = await client.post(
                "/ui/api/plan/approve",
                headers={"X-Slavik-Session": session_id},
                **({"json": body} if field != "missing" else {}),
            )
            assert response.status == (428 if field == "missing" else 409)
            workflow = await client.server.app["ui_hub"].get_session_workflow(session_id)
            assert workflow["active_plan"]["status"] == "draft"
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("field", ["plan_id", "plan_hash"])
def test_execute_rejects_another_plan_with_the_same_revision(field):
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            headers = {"X-Slavik-Session": session_id}
            approved = await client.post(
                "/ui/api/plan/approve", headers=headers, json=_binding(plan)
            )
            assert approved.status == 200
            body = _binding((await approved.json())["active_plan"])
            body[field] = "another-plan"
            response = await client.post(
                "/ui/api/plan/execute",
                headers={**headers, "Idempotency-Key": "binding-reject-k1"},
                json=body,
            )
            assert response.status == 409
            assert (await response.json())["error"]["code"] == "plan_binding_mismatch"
            hub = client.server.app["ui_hub"]
            assert (await hub.get_session_workflow(session_id))["active_task"] is None
            assert await hub.get_session_decision(session_id) is None
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("plan_revision", True),
        ("plan_revision", 0),
        ("plan_revision", "1"),
        ("plan_id", ""),
        ("plan_hash", None),
    ],
)
def test_approve_rejects_malformed_binding(field, value):
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            body = _binding(plan)
            body[field] = value
            response = await client.post(
                "/ui/api/plan/approve", headers={"X-Slavik-Session": session_id}, json=body
            )
            assert response.status == 400
            assert (await client.server.app["ui_hub"].get_session_workflow(session_id))[
                "active_plan"
            ]["status"] == "draft"
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("change", ["replace", "delete", "mode"])
def test_approve_rechecks_snapshot_atomically(monkeypatch, change):
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            headers = {"X-Slavik-Session": session_id}
            hub = client.server.app["ui_hub"]
            original = hub.approve_plan_if_current
            entered, release = asyncio.Event(), asyncio.Event()

            async def held(*args, **kwargs):
                entered.set()
                await release.wait()
                return await original(*args, **kwargs)

            monkeypatch.setattr(hub, "approve_plan_if_current", held)
            pending = asyncio.create_task(
                client.post("/ui/api/plan/approve", headers=headers, json=_binding(plan))
            )
            await asyncio.wait_for(entered.wait(), 2)
            if change == "replace":
                response = await client.post(
                    "/ui/api/plan/draft", headers=headers, json={"goal": "Replacement"}
                )
                assert response.status == 200
                replacement = (await response.json())["active_plan"]
            elif change == "delete":
                response = await client.delete(f"/ui/api/sessions/{session_id}", headers=headers)
                assert response.status == 200
            else:
                response = await client.post("/ui/api/mode", headers=headers, json={"mode": "ask"})
                assert response.status == 200
            release.set()
            assert (await pending).status == 409
            if change == "delete":
                response = await client.get(f"/ui/api/sessions/{session_id}", headers=headers)
                assert response.status == 404
            else:
                workflow = await hub.get_session_workflow(session_id)
                assert workflow["active_plan"]["status"] == "draft"
                if change == "replace":
                    assert workflow["active_plan"]["plan_id"] == replacement["plan_id"]
                else:
                    assert workflow["mode"] == "ask"
        finally:
            await client.close()

    asyncio.run(run())


def test_approval_storage_failure_rolls_back_before_publication(monkeypatch):
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            hub = client.server.app["ui_hub"]
            original = hub._persist_session_locked
            events = []
            original_publish = hub._publish_to_subscribers

            def fail_approved(session_id):
                if hub._sessions[session_id].active_plan["status"] == "approved":
                    raise OSError("controlled approval storage failure")
                return original(session_id)

            def observe(subscribers, event):
                events.append(event)
                return original_publish(subscribers, event)

            monkeypatch.setattr(hub, "_persist_session_locked", fail_approved)
            monkeypatch.setattr(hub, "_publish_to_subscribers", observe)
            response = await client.post(
                "/ui/api/plan/approve",
                headers={"X-Slavik-Session": session_id},
                json=_binding(plan),
            )
            assert response.status == 500
            assert (await hub.get_session_workflow(session_id))["active_plan"]["status"] == "draft"
            assert not events
        finally:
            await client.close()

    asyncio.run(run())


def test_execute_requires_complete_binding():
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            headers = {"X-Slavik-Session": session_id}
            approved = await client.post(
                "/ui/api/plan/approve", headers=headers, json=_binding(plan)
            )
            assert approved.status == 200
            response = await client.post(
                "/ui/api/plan/execute",
                headers=headers,
                json={"plan_revision": (await approved.json())["active_plan"]["plan_revision"]},
            )
            assert response.status == 428
            hub = client.server.app["ui_hub"]
            assert (await hub.get_session_workflow(session_id))["active_task"] is None
            assert await hub.get_session_decision(session_id) is None
        finally:
            await client.close()

    asyncio.run(run())


def test_committed_approval_survives_event_delivery_failure(monkeypatch):
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            hub = client.server.app["ui_hub"]
            failures = []

            def fail_delivery(subscribers, event):
                assert hub._sessions[session_id].active_plan["status"] == "approved"
                failures.append(event["type"])
                raise RuntimeError("controlled event delivery failure")

            monkeypatch.setattr(hub, "_publish_to_subscribers", fail_delivery)
            response = await client.post(
                "/ui/api/plan/approve",
                headers={"X-Slavik-Session": session_id},
                json=_binding(plan),
            )
            assert response.status == 200
            assert (await response.json())["active_plan"]["status"] == "approved"
            assert failures == ["session.workflow"]
        finally:
            await client.close()

    asyncio.run(run())


def test_approval_response_does_not_recreate_a_session_deleted_after_commit(monkeypatch):
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            headers = {"X-Slavik-Session": session_id}
            hub = client.server.app["ui_hub"]
            original = hub.approve_plan_if_current
            committed, release = asyncio.Event(), asyncio.Event()

            async def held(*args, **kwargs):
                result = await original(*args, **kwargs)
                committed.set()
                await release.wait()
                return result

            monkeypatch.setattr(hub, "approve_plan_if_current", held)
            pending = asyncio.create_task(
                client.post("/ui/api/plan/approve", headers=headers, json=_binding(plan))
            )
            await asyncio.wait_for(committed.wait(), 2)
            assert (
                await client.delete(f"/ui/api/sessions/{session_id}", headers=headers)
            ).status == 200
            release.set()
            response = await pending
            assert response.status == 200
            assert (
                await client.get(f"/ui/api/sessions/{session_id}", headers=headers)
            ).status == 404
            assert (await response.json())["active_plan"]["status"] == "approved"
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("choice", ["approve_once", "edit_plan"])
def test_old_execute_confirmation_cannot_run_a_replacement_with_same_revision(monkeypatch, choice):
    calls = []

    async def runner(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("server.http.handlers.decision._run_plan_runner", runner)

    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            headers = {"X-Slavik-Session": session_id}
            approved = await client.post(
                "/ui/api/plan/approve", headers=headers, json=_binding(plan)
            )
            binding = _binding((await approved.json())["active_plan"])
            decision_response = await client.post(
                "/ui/api/plan/execute",
                headers={**headers, "Idempotency-Key": "old-confirm-k1"},
                json=binding,
            )
            assert decision_response.status == 409
            decision = (await decision_response.json())["decision"]
            replacement_response = await client.post(
                "/ui/api/plan/draft", headers=headers, json={"goal": "Replacement consent target"}
            )
            assert replacement_response.status == 200
            replacement = (await replacement_response.json())["active_plan"]
            approved_replacement = await client.post(
                "/ui/api/plan/approve", headers=headers, json=_binding(replacement)
            )
            assert approved_replacement.status == 200
            assert (await approved_replacement.json())["active_plan"]["plan_revision"] == binding[
                "plan_revision"
            ]
            response = await client.post(
                "/ui/api/decision/respond",
                headers=headers,
                json={
                    "session_id": session_id,
                    "decision_id": decision["id"],
                    "choice": choice,
                    "edited_plan": {
                        "plan_revision": binding["plan_revision"],
                        "operation": {
                            "op": "reorder_steps",
                            "step_ids": [step["step_id"] for step in replacement["steps"]],
                        },
                    },
                },
            )
            assert response.status == 409
            assert (await response.json())["error"]["code"] == "plan_binding_mismatch"
            workflow = await client.server.app["ui_hub"].get_session_workflow(session_id)
            assert workflow["active_task"] is None
            assert workflow["active_plan"]["plan_id"] == replacement["plan_id"]
            assert workflow["active_plan"]["status"] == "approved"
            await asyncio.sleep(0)
            assert calls == []
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("missing", [None, "plan_id", "plan_hash", "plan_revision"])
def test_execute_confirmation_requires_complete_captured_binding(missing):
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            headers = {"X-Slavik-Session": session_id}
            approved = await client.post(
                "/ui/api/plan/approve", headers=headers, json=_binding(plan)
            )
            approved_plan = (await approved.json())["active_plan"]
            result = await client.post(
                "/ui/api/plan/execute",
                headers={**headers, "Idempotency-Key": "captured-binding-k1"},
                json=_binding(approved_plan),
            )
            decision = (await result.json())["decision"]
            if missing:
                del decision["context"]["resume_payload"][missing]
                await client.server.app["ui_hub"].set_session_decision(session_id, decision)
            response = await client.post(
                "/ui/api/decision/respond",
                headers=headers,
                json={
                    "session_id": session_id,
                    "decision_id": decision["id"],
                    "choice": "edit_plan",
                    "edited_plan": {
                        "plan_revision": approved_plan["plan_revision"],
                        "operation": {
                            "op": "reorder_steps",
                            "step_ids": [step["step_id"] for step in approved_plan["steps"]],
                        },
                    },
                },
            )
            workflow = await client.server.app["ui_hub"].get_session_workflow(session_id)
            if missing:
                assert response.status == 409
                assert (await response.json())["error"]["code"] == "plan_binding_mismatch"
                assert workflow["active_plan"] == approved_plan
            else:
                assert response.status == 200
                assert workflow["active_plan"]["status"] == "draft"
                assert workflow["active_plan"]["plan_revision"] > approved_plan["plan_revision"]
            assert workflow["active_task"] is None
        finally:
            await client.close()

    asyncio.run(run())


def _edit_body(plan, operation):
    return {**_binding(plan), "operation": operation}


def _update_title_op(title: str) -> dict:
    return {
        "op": "update_step",
        "step_id": "step-1",
        "changes": {"title": title},
    }


def test_edit_two_concurrent_edits_from_same_revision_only_one_applies():
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            headers = {"X-Slavik-Session": session_id}
            revision = plan["plan_revision"]

            async def _edit_once(title):
                return await client.post(
                    "/ui/api/plan/edit",
                    headers=headers,
                    json=_edit_body(plan, _update_title_op(title)),
                )

            first_resp, second_resp = await asyncio.gather(
                _edit_once("Title A"), _edit_once("Title B")
            )
            assert sorted([first_resp.status, second_resp.status]) == [200, 409]
            loser = first_resp if first_resp.status == 409 else second_resp
            assert (await loser.json())["error"]["code"] == "plan_binding_mismatch"
            workflow = await client.server.app["ui_hub"].get_session_workflow(session_id)
            final_plan = workflow["active_plan"]
            assert final_plan["plan_revision"] == revision + 1
            assert final_plan["status"] == "draft"
            titles = {
                step.get("title") for step in final_plan["steps"] if step.get("step_id") == "step-1"
            }
            assert titles in ({"Title A"}, {"Title B"})
        finally:
            await client.close()

    asyncio.run(run())


def test_edit_plan_if_current_rejects_when_mode_is_act():
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            hub = client.server.app["ui_hub"]
            await hub.set_session_workflow(
                session_id,
                mode="act",
                active_plan=plan,
                active_task={"task_id": "task-x", "status": "running"},
            )
            updated = deepcopy(plan)
            updated["steps"][0] = dict(updated["steps"][0], title="Edited while running")
            committed, reason = await hub.edit_plan_if_current(
                session_id, expected_plan=deepcopy(plan), updated_plan=updated
            )
            assert committed is None
            assert reason == "mode_mismatch"
            workflow = await hub.get_session_workflow(session_id)
            assert workflow["mode"] == "act"
            assert workflow["active_plan"] == plan
            assert workflow["active_task"]["status"] == "running"
        finally:
            await client.close()

    asyncio.run(run())


def test_edit_plan_if_current_rejects_running_task():
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            hub = client.server.app["ui_hub"]
            await hub.set_session_workflow(
                session_id,
                mode="plan",
                active_plan=plan,
                active_task={"task_id": "task-x", "status": "running"},
            )
            updated = deepcopy(plan)
            committed, reason = await hub.edit_plan_if_current(
                session_id, expected_plan=deepcopy(plan), updated_plan=updated
            )
            assert committed is None
            assert reason == "task_running"
            workflow = await hub.get_session_workflow(session_id)
            assert workflow["active_plan"] == plan
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("status", ["approved", "running", "completed", "failed", "cancelled"])
def test_edit_plan_if_current_rejects_non_draft_status(status):
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            hub = client.server.app["ui_hub"]
            non_draft = dict(plan, status=status)
            await hub.set_session_workflow(
                session_id, mode="plan", active_plan=non_draft, active_task=None
            )
            updated = deepcopy(non_draft)
            committed, reason = await hub.edit_plan_if_current(
                session_id, expected_plan=deepcopy(non_draft), updated_plan=updated
            )
            assert committed is None
            assert reason == "not_draft"
            workflow = await hub.get_session_workflow(session_id)
            assert workflow["active_plan"]["status"] == status
            assert workflow["active_plan"]["plan_revision"] == plan["plan_revision"]
        finally:
            await client.close()

    asyncio.run(run())


def test_edit_requires_full_binding():
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            headers = {"X-Slavik-Session": session_id}
            operation = _update_title_op("New title")

            missing = await client.post(
                "/ui/api/plan/edit",
                headers=headers,
                json={"plan_revision": plan["plan_revision"], "operation": operation},
            )
            assert missing.status == 428
            assert (await missing.json())["error"]["code"] == "plan_precondition_required"

            wrong_id = await client.post(
                "/ui/api/plan/edit",
                headers=headers,
                json={
                    **_binding(plan),
                    "plan_id": "another-plan",
                    "operation": operation,
                },
            )
            assert wrong_id.status == 409
            assert (await wrong_id.json())["error"]["code"] == "plan_binding_mismatch"

            stale = await client.post(
                "/ui/api/plan/edit",
                headers=headers,
                json={
                    **_binding(plan),
                    "plan_revision": plan["plan_revision"] + 5,
                    "operation": operation,
                },
            )
            assert stale.status == 409
            stale_payload = await stale.json()
            assert stale_payload["error"]["code"] == "plan_binding_mismatch"
            details = stale_payload["error"].get("details")
            assert isinstance(details, dict)
            assert details["expected_revision"] == plan["plan_revision"]
            assert details["actual_revision"] == plan["plan_revision"] + 5

            workflow = await client.server.app["ui_hub"].get_session_workflow(session_id)
            assert workflow["active_plan"]["plan_revision"] == plan["plan_revision"]
        finally:
            await client.close()

    asyncio.run(run())


def test_edit_storage_failure_rolls_back_before_publication(monkeypatch):
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            hub = client.server.app["ui_hub"]
            events = []
            original_publish = hub._publish_to_subscribers

            def fail_persist(session_id):
                raise OSError("controlled edit storage failure")

            def observe(subscribers, event):
                events.append(event)
                return original_publish(subscribers, event)

            monkeypatch.setattr(hub, "_persist_session_locked", fail_persist)
            monkeypatch.setattr(hub, "_publish_to_subscribers", observe)
            response = await client.post(
                "/ui/api/plan/edit",
                headers={"X-Slavik-Session": session_id},
                json=_edit_body(plan, _update_title_op("New title")),
            )
            assert response.status == 500
            workflow = await hub.get_session_workflow(session_id)
            assert workflow["active_plan"]["plan_revision"] == plan["plan_revision"]
            assert workflow["active_plan"]["status"] == "draft"
            assert not events
        finally:
            await client.close()

    asyncio.run(run())


def test_edit_draft_plan_happy_path_clears_decision():
    async def run():
        client = await _create_client(DummyAgent())
        try:
            session_id, plan = await _draft(client)
            headers = {"X-Slavik-Session": session_id}
            approved = await client.post(
                "/ui/api/plan/approve", headers=headers, json=_binding(plan)
            )
            assert approved.status == 200
            approved_plan = (await approved.json())["active_plan"]
            decision_response = await client.post(
                "/ui/api/plan/execute",
                headers={**headers, "Idempotency-Key": "happy-path-k1"},
                json=_binding(approved_plan),
            )
            assert decision_response.status == 409
            assert await client.server.app["ui_hub"].get_session_decision(session_id) is not None
            replacement = await client.post(
                "/ui/api/plan/draft", headers=headers, json={"goal": "Replacement draft"}
            )
            assert replacement.status == 200
            fresh_plan = (await replacement.json())["active_plan"]
            assert fresh_plan["status"] == "draft"
            assert await client.server.app["ui_hub"].get_session_decision(session_id) is not None
            old_hash = fresh_plan["plan_hash"]
            response = await client.post(
                "/ui/api/plan/edit",
                headers=headers,
                json=_edit_body(fresh_plan, _update_title_op("Edited title")),
            )
            assert response.status == 200
            payload = await response.json()
            edited = payload["active_plan"]
            assert edited["plan_revision"] == fresh_plan["plan_revision"] + 1
            assert edited["plan_hash"] != old_hash
            assert edited["status"] == "draft"
            assert payload["active_task"] is None
            assert await client.server.app["ui_hub"].get_session_decision(session_id) is None
            workflow = await client.server.app["ui_hub"].get_session_workflow(session_id)
            assert workflow["active_plan"] == edited
        finally:
            await client.close()

    asyncio.run(run())
