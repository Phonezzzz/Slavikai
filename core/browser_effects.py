"""Effect classification for generic browser operations (E02-T05).

One explicit table maps every ``desktop_browser`` operation to the effect it
can have, so authorization never treats "the browser is logged in" as a
blanket right to act on the owner's behalf:

- ``read`` — observe only (snapshot/read/find/wait/tabs and session
  bookkeeping that changes no page or external state);
- ``network`` — navigation: fetches pages but submits nothing;
- ``local_mutation`` — draft edits inside the page (typing, selecting);
  nothing leaves the browser until an explicit submit/click;
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
    # Draft edits stay inside the page until an explicit submit/click.
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
