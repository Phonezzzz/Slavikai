from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from core.tool_gateway import ToolGateway
from memory.canonical_atom_store import CanonicalAtomStore
from memory.vector_index import VectorIndex
from shared.canonical_atom_models import CanonicalAtom
from shared.models import JSONValue, ToolRequest, ToolResult
from tools.tool_logger import ToolCallLogger
from tools.tool_registry import ToolRegistry


class MemoryManagement:
    """Request-scoped access to already authenticated principal storage paths."""

    def __init__(self, *, canonical_path: Path, vectors_path: Path, principal_id: str):
        self._vectors_path = vectors_path
        self.subject = hashlib.sha256(("memory-management:" + principal_id).encode()).hexdigest()
        self.registry = ToolRegistry(
            call_logger=ToolCallLogger(canonical_path.with_name("memory_tool_calls.jsonl"))
        )
        self.registry.register(
            "memory_delete_confirmed",
            self._delete,
            capability="write",
            risk_classes=["destructive"],
            parameters_schema={
                "type": "object",
                "properties": {"operation_id": {"type": "string"}},
                "required": ["operation_id"],
                "additionalProperties": False,
            },
            model_exposed=False,
            confirmed_decision_only=True,
        )

        self.store = CanonicalAtomStore(str(canonical_path))

    def close(self) -> None:
        self.store.close()

    @staticmethod
    def _item(atom: CanonicalAtom) -> dict[str, JSONValue]:
        return {
            "atom_id": atom.atom_id,
            "stable_key": atom.stable_key,
            "revision": atom.revision,
            "claim_type": atom.claim_type.value,
            "value_json": atom.value_json,
            "summary_text": atom.summary_text,
            "status": atom.status.value,
            "pinned": atom.pinned,
            "last_seen_at": atom.last_seen_at,
        }

    def list_items(self, *, cursor: str | None, limit: int) -> dict[str, JSONValue]:
        after = None
        if cursor is not None:
            try:
                if len(cursor) > 8192:
                    raise ValueError("cursor length")
                raw: object = json.loads(
                    base64.b64decode(cursor.encode("ascii"), altchars=b"-_", validate=True)
                )
                if (
                    not isinstance(raw, list)
                    or len(raw) != 2
                    or raw[0] != self.subject
                    or not isinstance(raw[1], str)
                ):
                    raise ValueError("cursor binding")
                after = raw[1]
            except (ValueError, UnicodeError) as exc:
                raise ValueError("memory_cursor_invalid") from exc
        atoms, next_id = self.store.list_page(after=after, limit=limit)
        next_cursor = (
            base64.urlsafe_b64encode(json.dumps([self.subject, next_id]).encode()).decode()
            if next_id
            else None
        )
        return {
            "items": [self._item(atom) for atom in atoms],
            "next_cursor": next_cursor,
            "subject": self.subject,
        }

    def preview(self, atom_id: str, revision: str) -> dict[str, JSONValue]:
        operation_id, atom, expires_at = self.store.preview_delete(atom_id, revision)
        return {
            "operation_id": operation_id,
            "expires_at": expires_at,
            "item": self._item(atom),
            "subject": self.subject,
            "scope": "canonical_atom_and_atom_vector",
            "retained": (
                "История чатов, inbox, preferences, traces и backups не удаляются. "
                "Это не secure erase диска."
            ),
        }

    def confirm(self, operation_id: str) -> ToolResult:
        return ToolGateway(registry=self.registry, confirmed_decision=True).call(
            ToolRequest(name="memory_delete_confirmed", args={"operation_id": operation_id})
        )

    def _delete(self, request: ToolRequest) -> ToolResult:
        operation_id = request.args.get("operation_id")
        if (
            set(request.args) != {"operation_id"}
            or not isinstance(operation_id, str)
            or not operation_id
            or len(operation_id) > 128
        ):
            return ToolResult.failure(
                "Некорректное подтверждение Memory.", meta={"code": "memory_delete_invalid"}
            )
        # Opening this existing store never initializes/downloads an embedding model.
        vectors = VectorIndex(str(self._vectors_path))
        try:
            self.store.bind_vectors(vectors.db_path)
            status = self.store.confirm_delete_preview(operation_id)
            if status == "deleted":
                return ToolResult.success({"status": status}, meta={"status": status})
            return ToolResult.failure(
                "Подтверждение удаления Memory отклонено.",
                data={"status": status},
                meta={"status": status},
            )
        finally:
            vectors.close()
