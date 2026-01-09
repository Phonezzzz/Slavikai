from __future__ import annotations

import pytest

from core.mwv.models import MWVMessage
from core.mwv.routing import RouteDecision, classify_request
from core.skills.index import SkillIndex
from core.skills.models import SkillEntry, SkillManifest


@pytest.mark.parametrize(
    ("text", "expected_route", "expected_flags"),
    [
        ("что такое git", "chat", []),
        ("как работает sudo", "chat", []),
        ("объясни как работает docker", "chat", []),
        ("какая погода завтра", "chat", []),
        ("что значит ошибка 404", "chat", []),
        ("объясни разницу между tcp и udp", "chat", []),
        ("что такое json и зачем нужен", "chat", []),
        ("расскажи про latency и throughput", "chat", []),
        ("помоги выбрать ноутбук для работы", "chat", []),
        ("что такое dependency injection", "chat", []),
        ("поясни термин idempotent", "chat", []),
        ("как устроен http запрос", "chat", []),
        ("почему падает тест", "chat", []),
        ("что означает stacktrace", "chat", []),
        ("исправь тесты", "mwv", ["code_change"]),
        ("почини тесты", "mwv", ["code_change"]),
        ("поправь баг в коде", "mwv", ["code_change"]),
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


def test_classify_request_empty_input_without_triggers() -> None:
    decision = classify_request(messages=[], user_input="", context=None)
    assert decision.route == "chat"
    assert decision.reason == "fallback_messages:no_triggers"
    assert decision.risk_flags == []


def _make_skill_index(skills: list[SkillEntry]) -> SkillIndex:
    return SkillIndex(SkillManifest(manifest_version=1, skills=skills))


def test_classify_request_uses_skill_index_match() -> None:
    index = _make_skill_index(
        [
            SkillEntry(
                id="alpha",
                version="1.0.0",
                title="Alpha",
                entrypoints=["tool_a"],
                patterns=["alpha"],
                requires=[],
                risk="low",
                tests=[],
                path="skills/alpha/skill.md",
                content_hash="hash",
            )
        ]
    )
    decision = classify_request(
        messages=[],
        user_input="use ALPHA skill",
        context=None,
        skill_index=index,
    )
    assert decision.route == "mwv"
    assert decision.skill_decision is not None
    assert decision.skill_decision.status == "matched"
    assert decision.skill_decision.match is not None
    assert decision.skill_decision.match.entry.id == "alpha"
    assert "skill_match:alpha" in decision.reason


def test_classify_request_skill_deprecated() -> None:
    index = _make_skill_index(
        [
            SkillEntry(
                id="legacy",
                version="1.0.0",
                title="Legacy",
                entrypoints=["tool_a"],
                patterns=["legacy"],
                requires=[],
                risk="low",
                tests=[],
                path="skills/legacy/skill.md",
                content_hash="hash",
                deprecated=True,
                replaced_by="modern",
            )
        ]
    )
    decision = classify_request(
        messages=[],
        user_input="legacy task",
        context=None,
        skill_index=index,
    )
    assert decision.route == "mwv"
    assert decision.skill_decision is not None
    assert decision.skill_decision.status == "deprecated"
    assert "skill_deprecated:legacy" in decision.reason


def test_classify_request_skill_ambiguous() -> None:
    index = _make_skill_index(
        [
            SkillEntry(
                id="alpha",
                version="1.0.0",
                title="Alpha",
                entrypoints=["tool_a"],
                patterns=["build"],
                requires=[],
                risk="low",
                tests=[],
                path="skills/alpha/skill.md",
                content_hash="hash",
            ),
            SkillEntry(
                id="beta",
                version="1.0.0",
                title="Beta",
                entrypoints=["tool_b"],
                patterns=["build"],
                requires=[],
                risk="low",
                tests=[],
                path="skills/beta/skill.md",
                content_hash="hash",
            ),
        ]
    )
    decision = classify_request(
        messages=[],
        user_input="build pipeline",
        context=None,
        skill_index=index,
    )
    assert decision.route == "mwv"
    assert decision.skill_decision is not None
    assert decision.skill_decision.status == "ambiguous"
    assert "skill_ambiguous" in decision.reason
