from __future__ import annotations

# ruff: noqa: F403,F405
from .fakes import *


def test_memory_triage_preview_apply_undo_flow(tmp_path: Path) -> None:
    async def run() -> None:
        agent = MemoryTriageAgent(tmp_path / "triage.db")
        client = await _create_client(agent)
        try:
            preview_resp = await client.post("/ui/api/memory/triage/preview", json={"limit": 20})
            assert preview_resp.status == 200
            preview_payload = await preview_resp.json()
            plan = preview_payload.get("plan")
            assert isinstance(plan, dict)
            suggestions = plan.get("suggestions")
            assert isinstance(suggestions, list)
            assert suggestions

            apply_resp = await client.post(
                "/ui/api/memory/triage/apply",
                json={"plan": plan, "allow_dangerous": True},
            )
            assert apply_resp.status == 200
            apply_payload = await apply_resp.json()
            result = apply_payload.get("result")
            assert isinstance(result, dict)
            tx_id = result.get("tx_id")
            assert isinstance(tx_id, str) and tx_id
            applied = result.get("applied")
            assert isinstance(applied, list)
            assert applied

            undo_resp = await client.post(
                "/ui/api/memory/triage/undo",
                json={"tx_id": tx_id},
            )
            assert undo_resp.status == 200
            undo_payload = await undo_resp.json()
            undo_result = undo_payload.get("result")
            assert isinstance(undo_result, dict)
            restored = undo_result.get("restored")
            assert isinstance(restored, list)
            assert len(restored) >= 1
        finally:
            await client.close()

    asyncio.run(run())
