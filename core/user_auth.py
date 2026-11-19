"""Ручной вход пользователя в браузере (E02-T04, manual login handoff).

Решение владельца (вариант A, 2026-11-19): пауза run на ручной вход — это
тот же механизм continuation, что и у ``ApprovalRequired``
(``ToolLoopContinuation``), но с другой причиной ожидания:

- фаза run — ``RunPhase.WAITING_USER_INPUT``; причина ожидания в UI/API —
  ``NEEDS_USER_AUTH``;
- browser tool перезапускает ТОТ ЖЕ persistent profile в видимом Chromium,
  не отпуская profile lease, и бросает :class:`UserAuthRequired`, когда окно
  готово;
- «Готово» (``done``) означает только «пользователь вернул управление», это
  не доказательство входа: при продолжении tool заново снимает observation
  страницы, и уже модель решает, идти ли дальше;
- таймаут считается от ``window_ready_monotonic``; по истечении окно попытки
  закрывается, run остаётся в ``NEEDS_USER_AUTH``, «Готово» больше не
  принимается, доступен только явный ``retry``.

Пароль и MFA вводятся пользователем прямо в Chromium: в этом модуле нет ни
одного поля для них, и они не проходят ни через модель, ни через tool log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from core.tool_loop import ToolLoopContinuation

NEEDS_USER_AUTH: Final[str] = "NEEDS_USER_AUTH"
USER_AUTH_TIMEOUT_SECONDS: Final[float] = 300.0
USER_AUTH_PROMPT: Final[str] = "Войди в окне Chromium на ПК → Готово"

UserAuthChoice = Literal["done", "retry"]


@dataclass(frozen=True)
class UserAuthRequest:
    """Одна попытка ручного входа: какое окно открыто и когда оно готово.

    ``window_ready_monotonic`` — момент, когда видимое окно Chromium открыто
    на нужной странице; от него runtime считает таймаут. ``control_returned``
    выставляет только runtime, когда владелец run нажал «Готово» до
    истечения попытки; модель и tool arguments его не задают.
    """

    attempt_id: str
    tool: str
    page_id: str
    url: str
    title: str
    window_ready_monotonic: float
    session_id: str | None = None
    # Выставляет DesktopRuntime при паузе: срок попытки (wall clock, epoch)
    # для статуса в UI/API и длительность таймаута.
    expires_at_epoch: float | None = None
    timeout_seconds: float | None = None
    control_returned: bool = False

    @property
    def reason(self) -> str:
        return NEEDS_USER_AUTH


class UserAuthRequired(Exception):
    """Run ждёт ручного входа пользователя (``NEEDS_USER_AUTH``).

    Наследуется от ``Exception``, а не ``RuntimeError``: browser tool
    переводит ``RuntimeError`` в ``ToolResult.failure``, а этот сигнал должен
    дойти до tool loop как пауза, а не как ошибка инструмента.
    """

    def __init__(self, request: UserAuthRequest) -> None:
        super().__init__(NEEDS_USER_AUTH)
        self.request = request
        self.continuation: ToolLoopContinuation | None = None
