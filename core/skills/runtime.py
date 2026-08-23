from __future__ import annotations

from typing import Literal

from core.skills.index import SkillResolution
from shared.models import JSONValue

SkillRunStatus = Literal["completed", "failed", "skipped"]


def skill_run_metadata(resolution: SkillResolution) -> dict[str, JSONValue]:
    return {
        "skill_id": resolution.primary.id,
        "version": resolution.primary.version,
        "supporting_skills": [
            {"skill_id": entry.id, "version": entry.version} for entry in resolution.supporting
        ],
    }


def skill_run_report(
    resolution: SkillResolution | None,
    *,
    status: SkillRunStatus,
    reason: str | None = None,
) -> dict[str, JSONValue]:
    report: dict[str, JSONValue] = {"status": status}
    if resolution is not None:
        report.update(skill_run_metadata(resolution))
    else:
        report["skill_id"] = None
        report["version"] = None
        report["supporting_skills"] = []
    if reason:
        report["reason"] = reason
    return report
