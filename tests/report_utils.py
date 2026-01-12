from __future__ import annotations

import json

from core.mwv.models import MWV_REPORT_PREFIX


def extract_report_block(response: str) -> dict[str, object]:
    if MWV_REPORT_PREFIX not in response:
        raise AssertionError("MWV report block missing from response.")
    payload = response.rsplit(MWV_REPORT_PREFIX, maxsplit=1)[-1].strip()
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise AssertionError("MWV report block is not a JSON object.")
    return data
