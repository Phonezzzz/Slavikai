from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "docs/runtime_contract_claims.json"
STATUS_ORDER = ("implemented", "partial", "target", "legacy")
ALLOWED_STATUSES = set(STATUS_ORDER)
REQUIRED_CLAIM_FIELDS = {
    "id",
    "scope",
    "status",
    "owner",
    "version",
    "source",
    "verified_by",
    "summary",
}
REQUIRED_CANONICAL_DOCS = (
    "docs/SOURCE_OF_TRUTH.md",
    "docs/architecture/ARCH_CANON.md",
    "docs/agent/DevRules.md",
    "docs/workflow/dev_workflow.md",
    "docs/agent/COMMAND_LANE_POLICY.md",
    "docs/agent/ROUTING_POLICY.md",
    "docs/agent/STOP_RESPONSES.md",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_repo_path(raw_path: object, *, field: str) -> None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{field} must be a path")
    path_text = raw_path.strip().split("#", 1)[0]
    path = Path(path_text)
    _require(not path.is_absolute() and ".." not in path.parts, f"unsafe {field}: {raw_path}")
    _require((ROOT / path).exists(), f"missing {field}: {raw_path}")


def main() -> None:
    payload_raw = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    _require(isinstance(payload_raw, dict), "claims registry must be an object")
    _require(payload_raw.get("schema_version") == 1, "unsupported claims schema_version")
    _require(payload_raw.get("statuses") == list(STATUS_ORDER), "status list drift")

    claims = payload_raw.get("claims")
    _require(isinstance(claims, list) and bool(claims), "claims must be a non-empty list")
    seen_ids: set[str] = set()
    for index, claim_raw in enumerate(claims):
        _require(isinstance(claim_raw, dict), f"claim[{index}] must be an object")
        claim = claim_raw
        missing = REQUIRED_CLAIM_FIELDS - claim.keys()
        _require(not missing, f"claim[{index}] missing fields: {sorted(missing)}")
        claim_id = claim["id"]
        _require(isinstance(claim_id, str) and bool(claim_id.strip()), "claim id is required")
        _require(claim_id not in seen_ids, f"duplicate claim id: {claim_id}")
        seen_ids.add(claim_id)
        _require(claim["status"] in ALLOWED_STATUSES, f"invalid status for {claim_id}")
        _require(
            isinstance(claim["version"], int) and claim["version"] > 0,
            f"bad version: {claim_id}",
        )
        for field in ("scope", "owner", "summary"):
            value = claim[field]
            _require(
                isinstance(value, str) and bool(value.strip()),
                f"{field} is required: {claim_id}",
            )
        _require_repo_path(claim["source"], field=f"source for {claim_id}")
        verified_by = claim["verified_by"]
        _require(
            isinstance(verified_by, list) and bool(verified_by),
            f"verification missing: {claim_id}",
        )
        for path in verified_by:
            _require_repo_path(path, field=f"verification for {claim_id}")
        if claim["status"] == "implemented":
            _require(
                any(
                    isinstance(path, str)
                    and (path == "Makefile" or path.startswith(("tests/", "scripts/")))
                    for path in verified_by
                ),
                f"implemented claim lacks executable verification: {claim_id}",
            )

    agents_text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for path in REQUIRED_CANONICAL_DOCS:
        _require(f"`{path}`" in agents_text, f"AGENTS.md does not require {path}")

    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    _require(
        "LOCAL_LLM_URL=http://localhost:11434/v1/chat/completions" in env_example,
        ".env.example must use the complete local chat completions URL",
    )
    _require(
        (ROOT / ".nvmrc").read_text(encoding="utf-8").strip() == "20",
        ".nvmrc must pin Node 20",
    )

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("preflight:", "up-prod:", "down-prod:", "status-prod:", "logs-prod:"):
        _require(target in makefile, f"missing Make target: {target[:-1]}")
    git_check_contracts = (
        "git fetch --prune origin",
        "merge-base --is-ancestor origin/main HEAD",
    )
    for git_check_contract in git_check_contracts:
        _require(git_check_contract in makefile, f"git-check contract drift: {git_check_contract}")

    print(f"OK: {len(claims)} runtime contract claims validated.")


if __name__ == "__main__":
    main()
