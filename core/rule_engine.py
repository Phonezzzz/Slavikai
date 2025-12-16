from __future__ import annotations

import re
from dataclasses import dataclass

from shared.policy_models import (
    ActionAddInstruction,
    ActionSetResponseStyle,
    PolicyRule,
    ResponseVerbosity,
    TriggerAlways,
    TriggerUserMessageContains,
    TriggerUserMessageRegex,
)


@dataclass(frozen=True)
class PolicyApplication:
    applied_policy_ids: list[str]
    instructions: list[str]


def _verbosity_to_instruction(verbosity: ResponseVerbosity) -> str:
    if verbosity is ResponseVerbosity.CONCISE:
        return "Отвечай кратко и по делу, без лишних деталей."
    if verbosity is ResponseVerbosity.DETAILED:
        return "Отвечай подробно, структурировано и с примерами, если это уместно."
    return "Сохраняй нормальную подробность: достаточно деталей, но без воды."


def _trigger_matches(message: str, trigger: object) -> bool:
    if isinstance(trigger, TriggerAlways):
        return True

    if isinstance(trigger, TriggerUserMessageContains):
        if trigger.case_sensitive:
            return any(substring in message for substring in trigger.substrings)
        lowered = message.lower()
        return any(substring.lower() in lowered for substring in trigger.substrings)

    if isinstance(trigger, TriggerUserMessageRegex):
        flags = re.IGNORECASE if trigger.case_insensitive else 0
        return re.search(trigger.pattern, message, flags=flags) is not None

    raise TypeError(f"Unsupported trigger type: {type(trigger)}")


class RuleEngine:
    """
    Применяет только Approved PolicyRule.

    Runtime не создаёт/не изменяет правила — только читает и применяет в виде pre-gen инструкций.
    """

    def apply(self, *, user_message: str, rules: list[PolicyRule]) -> PolicyApplication:
        message = user_message.strip()
        applied_ids: list[str] = []
        instructions: list[str] = []
        instruction_seen: set[str] = set()

        style_applied = False
        for rule in rules:
            if not _trigger_matches(message, rule.trigger):
                continue

            action = rule.action
            if isinstance(action, ActionAddInstruction):
                text = action.text.strip()
                if text and text not in instruction_seen:
                    instructions.append(text)
                    instruction_seen.add(text)
                applied_ids.append(rule.rule_id)
                continue

            if isinstance(action, ActionSetResponseStyle):
                if style_applied:
                    continue
                instruction = _verbosity_to_instruction(action.verbosity)
                if instruction not in instruction_seen:
                    instructions.append(instruction)
                    instruction_seen.add(instruction)
                applied_ids.append(rule.rule_id)
                style_applied = True
                continue

            raise TypeError(f"Unsupported action type: {type(action)}")

        return PolicyApplication(applied_policy_ids=applied_ids, instructions=instructions)
