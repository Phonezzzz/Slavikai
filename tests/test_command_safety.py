from __future__ import annotations

from shared.command_safety import is_hard_unsafe_command


def test_rm_recursive_force_variants_are_hard_denied() -> None:
    for command in (
        "rm -rf /tmp/x",
        "rm -fr /tmp/x",
        "rm -r -f /tmp/x",
        "rm -R -F /tmp/x",
        "rm --recursive --force /tmp/x",
        "rm --force --recursive /tmp/x",
    ):
        assert is_hard_unsafe_command(command), command


def test_rm_without_both_recursive_and_force_is_not_hard_denied() -> None:
    for command in (
        "rm /tmp/x",
        "rm -r /tmp/x",
        "rm -f /tmp/x",
        "rm --recursive /tmp/x",
        "rm --force /tmp/x",
    ):
        assert not is_hard_unsafe_command(command), command
