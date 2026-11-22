"""Effect classification for generic browser operations (E02-T05).

One explicit table maps every ``desktop_browser`` operation to the effect it
can have, so authorization never treats "the browser is logged in" as a
blanket right to act on the owner's behalf:

- ``read`` — observe only (snapshot/read/find/wait/tabs and session
  bookkeeping that changes no page or external state);
- ``network`` — navigation: fetches pages but submits nothing;
- ``local_mutation`` — draft edits inside the page (typing, selecting);
  the agent performs no explicit submit, but page handlers may react to
  the edit immediately (autosave, fetch/XHR, WebSocket) — verified by a
  real browser test (tests/test_browser_input_external_effect.py:
  Playwright ``fill`` dispatches ``input``, ``select_option`` dispatches
  ``input``/``change``, and handlers sent requests with no submit/click),
  so this class is NOT a guarantee that nothing leaves the browser;
- ``write`` — writes a local file (download payload; the destination path
  itself is classified separately by the download-destination action);
- ``external_side_effect`` — can change state outside this browser
  (submit, click, upload) or discard page state (close_tab);
- ``user_handoff`` — hands control to the owner (manual login).

Unknown operations fail closed as ``external_side_effect``: a new or
mistyped operation must never silently inherit ``read``.
"""

from __future__ import annotations

from typing import Final, Literal

BrowserEffectClass = Literal[
    "read",
    "network",
    "local_mutation",
    "write",
    "external_side_effect",
    "user_handoff",
]

_OPERATION_EFFECTS: Final[dict[str, BrowserEffectClass]] = {
    # Observe / read-only.
    "snapshot": "read",
    "read": "read",
    "find": "read",
    "wait": "read",
    "tabs": "read",
    "switch_tab": "read",
    "start": "read",
    "close": "read",
    # Navigation fetches but submits nothing.
    "open": "network",
    "new_tab": "network",
    "navigate": "network",
    # Draft edits: no explicit submit by the agent, but page handlers
    # may react with network requests (see module docstring).
    "input": "local_mutation",
    "select": "local_mutation",
    # Download writes a local file; the destination path is a separate
    # write/overwrite action in desktop policy derivation.
    "download": "write",
    # Can change state outside this browser. A click is never read: it can
    # trigger any destructive or external action on the page.
    "click": "external_side_effect",
    "submit": "external_side_effect",
    "upload": "external_side_effect",
    # Closing a tab discards unsaved page state; kept at the pre-E02-T05
    # conservative class (the ticket does not cover it).
    "close_tab": "external_side_effect",
    # Manual login handoff waits for the owner's own action in the window
    # and the one-shot Done; no separate approval (E02-T04).
    "login_handoff": "user_handoff",
}


def classify_browser_operation(operation: str) -> BrowserEffectClass:
    """Return the effect class of a generic browser operation.

    Unknown operations fail closed as ``external_side_effect`` so they can
    never inherit ``read`` by accident.
    """
    return _OPERATION_EFFECTS.get(operation.strip().lower(), "external_side_effect")


def known_browser_operations() -> frozenset[str]:
    """Operations with an explicit classification (for coverage tests)."""
    return frozenset(_OPERATION_EFFECTS)


# Effect classes that earlier versions assigned to an operation and that
# persisted owner rules may still carry. Before E02-T05, input/select/
# download were classified ``external_side_effect`` by the inline table
# in desktop policy derivation. Restrictive (deny/ask) rules saved with a
# superseded class are migrated to the current class at load time so a
# reclassification never silently drops an owner's prohibition; allow
# rules are never migrated (fail closed: they simply stop matching).
_SUPERSEDED_EFFECTS: Final[dict[str, frozenset[str]]] = {
    "input": frozenset({"external_side_effect"}),
    "select": frozenset({"external_side_effect"}),
    "download": frozenset({"external_side_effect"}),
}


def superseded_effect_classes(operation: str) -> frozenset[str]:
    """Effect classes previously assigned to ``operation`` (may be empty)."""
    return _SUPERSEDED_EFFECTS.get(operation.strip().lower(), frozenset())
