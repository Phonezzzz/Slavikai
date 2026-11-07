from collections import deque

import pytest

from llm.brain_base import Brain
from llm.types import LLMResult, LLMUsage, ToolCall
from scripts.goal_behavior_eval import run_case


class ScriptedBrain(Brain):
    supports_native_tools = True

    def __init__(self, actions):
        self.actions = deque(actions)

    def generate(self, messages, config=None, tools=None):
        if self.actions:
            name, arguments = self.actions.popleft()
            return LLMResult(
                text="",
                tool_calls=[ToolCall(str(len(self.actions)), name, arguments)],
                usage=LLMUsage(1, 1, 2),
            )
        return LLMResult(text="Готово, результат проверен.", usage=LLMUsage(1, 1, 2))


@pytest.mark.parametrize("complete", [True, False])
def test_accepts_both_strategies_with_independent_read_back(tmp_path, complete):
    actions = [("workspace_read", {"path": "export.json"})]
    if not complete:
        actions.append(("workspace_read", {"path": "register.json"}))
    actions += [
        ("workspace_write", {"path": "open.json", "content": '["A17","D42"]'}),
        ("workspace_read", {"path": "open.json"}),
    ]
    root = tmp_path / "workspace"
    root.mkdir()
    assert run_case(ScriptedBrain(actions), root, complete=complete) == ()


@pytest.mark.parametrize(
    ("complete", "actions", "expected_error"),
    [
        (True, [], "incorrect_result"),
        (
            False,
            [
                ("workspace_read", {"path": "export.json"}),
                ("workspace_write", {"path": "open.json", "content": '["A17","D42"]'}),
                ("workspace_read", {"path": "open.json"}),
            ],
            "strategy_evidence_missing",
        ),
        (
            True,
            [
                ("workspace_read", {"path": "export.json"}),
                ("workspace_write", {"path": "open.json", "content": '["A17"]'}),
                ("workspace_read", {"path": "open.json"}),
            ],
            "incorrect_result",
        ),
        (
            True,
            [
                ("workspace_read", {"path": "export.json"}),
                ("workspace_write", {"path": "open.json", "content": '["A17","D42"]'}),
            ],
            "read_back_missing",
        ),
        (
            True,
            [("workspace_write", {"path": "../outside.json", "content": "[]"})],
            "scope_violation",
        ),
    ],
)
def test_rejects_false_success_and_out_of_scope_attempts(
    tmp_path, complete, actions, expected_error
):
    root = tmp_path / "workspace"
    root.mkdir()
    assert expected_error in run_case(ScriptedBrain(actions), root, complete=complete)
    assert not (tmp_path / "outside.json").exists()


def test_rejects_prefetched_strategy_in_complete_case(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    actions = [
        ("workspace_read", {"path": "export.json"}),
        ("workspace_read", {"path": "register.json"}),
        ("workspace_write", {"path": "open.json", "content": '["A17","D42"]'}),
        ("workspace_read", {"path": "open.json"}),
    ]
    assert "unnecessary_alternative" in run_case(ScriptedBrain(actions), root, complete=True)


def test_rejects_batch_that_cannot_have_used_observations(tmp_path):
    class BatchBrain(Brain):
        supports_native_tools = True
        sent = False

        def generate(self, messages, config=None, tools=None):
            if self.sent:
                return LLMResult(text="Готово", usage=LLMUsage(1, 1, 2))
            self.sent = True
            return LLMResult(
                text="",
                usage=LLMUsage(1, 1, 2),
                tool_calls=[
                    ToolCall("read", "workspace_read", {"path": "export.json"}),
                    ToolCall("alternative", "workspace_read", {"path": "register.json"}),
                    ToolCall(
                        "write",
                        "workspace_write",
                        {"path": "open.json", "content": '["A17","D42"]'},
                    ),
                    ToolCall("verify", "workspace_read", {"path": "open.json"}),
                ],
            )

    root = tmp_path / "workspace"
    root.mkdir()
    assert "observation_not_used" in run_case(BatchBrain(), root, complete=False)


def test_rejects_preexisting_result(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "open.json").write_text('["A17","D42"]', encoding="utf-8")
    with pytest.raises(ValueError, match="должна быть пустой"):
        run_case(ScriptedBrain([]), root, complete=True)


def test_accepts_equivalent_workspace_paths(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    actions = [
        ("workspace_read", {"path": "./export.json"}),
        ("workspace_write", {"path": "./open.json", "content": '["A17","D42"]'}),
        ("workspace_read", {"path": "./open.json"}),
    ]
    assert run_case(ScriptedBrain(actions), root, complete=True) == ()
