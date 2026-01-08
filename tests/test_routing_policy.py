from __future__ import annotations

import pytest

from core.mwv.models import MWVMessage
from core.mwv.routing import RouteDecision, classify_request


@pytest.mark.parametrize(
    ("text", "expected_route", "expected_flags"),
    [
        ("что такое git", "chat", []),
        ("как работает sudo", "chat", []),
        ("объясни как работает docker", "chat", []),
        ("исправь тесты", "mwv", ["code_change"]),
        ("почини тесты", "mwv", ["code_change"]),
        ("написать код для парсера", "mwv", ["code_change"]),
        ("рефактор модуля оплаты", "mwv", ["code_change"]),
        ("добавь фичу логирования", "mwv", ["code_change"]),
        ("сделай PR", "mwv", ["git"]),
        ("сделай коммит", "mwv", ["git"]),
        ('git commit -m "x"', "mwv", ["git"]),
        ("git push origin main", "mwv", ["git"]),
        ("pip install requests", "mwv", ["install"]),
        ("npm install", "mwv", ["install"]),
        ("установи зависимости", "mwv", ["install"]),
        ("sudo systemctl restart nginx", "mwv", ["sudo"]),
        ("systemctl restart nginx", "mwv", ["tools"]),
        ("docker build .", "mwv", ["tools"]),
        ("удали файл README.md", "mwv", ["filesystem"]),
        ("перезапиши файл config.yaml", "mwv", ["filesystem"]),
        ("создай файл notes.txt", "mwv", ["filesystem"]),
        ("применить патч для README", "mwv", ["filesystem"]),
        ("запусти shell команду ls", "mwv", ["tools"]),
    ],
)
def test_classify_request_routes(text: str, expected_route: str, expected_flags: list[str]) -> None:
    decision = classify_request(messages=[], user_input=text, context={})
    assert isinstance(decision, RouteDecision)
    assert decision.route == expected_route
    for flag in expected_flags:
        assert flag in decision.risk_flags
    if expected_route == "chat":
        assert decision.risk_flags == []


def test_classify_request_uses_messages_when_input_empty() -> None:
    messages = [MWVMessage(role="user", content="исправь тесты")]
    decision = classify_request(messages=messages, user_input="", context=None)
    assert decision.route == "mwv"
    assert "code_change" in decision.risk_flags


def test_classify_request_tool_role_forces_mwv() -> None:
    messages = [MWVMessage(role="tool", content="fs")]
    decision = classify_request(messages=messages, user_input="привет", context=None)
    assert decision.route == "mwv"
    assert "tools" in decision.risk_flags
