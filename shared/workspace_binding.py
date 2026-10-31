from dataclasses import dataclass
from typing import ClassVar


class WorkspaceUnavailable(ValueError):
    """Workspace не допускается к изолированному исполнению."""

    reason_code: ClassVar[str] = "workspace_unavailable"


class WorkspaceConflict(WorkspaceUnavailable):
    """Исходная привязка или содержимое workspace изменились."""

    reason_code: ClassVar[str] = "workspace_conflict"


class WorkspaceFileBudgetExceeded(WorkspaceUnavailable):
    """File-effect budget denied the staged output before publication."""

    reason_code: ClassVar[str] = "workspace_file_budget_exhausted"


@dataclass(frozen=True)
class WorkspaceBinding:
    """Доверенная привязка операции; не часть model tool arguments."""

    workspace_id: str
    principal_id: str
    session_id: str
    revision: str
    environment_revision: str | None = None
