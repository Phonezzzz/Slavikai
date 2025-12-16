from __future__ import annotations

import re

import pytest

from shared.policy_models import (
    ActionAddInstruction,
    ActionSetResponseStyle,
    ResponseVerbosity,
    TriggerAlways,
    TriggerUserMessageContains,
    TriggerUserMessageRegex,
    policy_action_from_dict,
    policy_action_to_dict,
    policy_trigger_from_dict,
    policy_trigger_to_dict,
)


def test_trigger_contains_validation() -> None:
    with pytest.raises(ValueError, match="substrings"):
        TriggerUserMessageContains(substrings=["  "])

    trig = TriggerUserMessageContains(substrings=[" hi ", "world"])
    as_dict = policy_trigger_to_dict(trig)
    assert as_dict["kind"] == "user_message_contains"
    assert as_dict["substrings"] == ["hi", "world"]


def test_trigger_regex_validation() -> None:
    with pytest.raises(ValueError, match="pattern"):
        TriggerUserMessageRegex(pattern="")
    with pytest.raises(re.error):
        TriggerUserMessageRegex(pattern="(")


def test_trigger_roundtrip() -> None:
    trig = TriggerAlways()
    loaded = policy_trigger_from_dict(policy_trigger_to_dict(trig))  # type: ignore[arg-type]
    assert isinstance(loaded, TriggerAlways)


def test_action_validation_and_roundtrip() -> None:
    with pytest.raises(ValueError, match="короткий"):
        ActionAddInstruction(text="  ")

    act = ActionSetResponseStyle(verbosity=ResponseVerbosity.CONCISE)
    as_dict = policy_action_to_dict(act)
    assert as_dict["verbosity"] == "concise"
    loaded = policy_action_from_dict(as_dict)  # type: ignore[arg-type]
    assert isinstance(loaded, ActionSetResponseStyle)
    assert loaded.verbosity == ResponseVerbosity.CONCISE
