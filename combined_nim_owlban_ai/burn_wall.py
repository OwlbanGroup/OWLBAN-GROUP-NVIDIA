"""
Burn Wall guardrail module.

Provides a deterministic, stdlib-only safety gate for inference payloads.
"""

from typing import Any, Dict, List, Tuple

_BLOCKLIST = (
    "social security number",
    "ssn",
    "credit card",
    "card number",
    "cvv",
    "password",
    "private key",
    "ignore system instructions",
    "reveal hidden prompt",
    "api key",
    "bypass policy",
)


def _flatten_payload(data: Any) -> str:
    """Convert any input payload into a lowercase searchable string."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data.lower()
    if isinstance(data, dict):
        return " ".join(f"{k} {_flatten_payload(v)}" for k, v in data.items()).lower()
    if isinstance(data, (list, tuple, set)):
        return " ".join(_flatten_payload(item) for item in data).lower()
    return str(data).lower()


def run_burn_wall(data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Evaluate the payload against deterministic safety rules.

    Returns:
        (allowed, refusal_payload, reasons)
        - allowed: True when payload is safe to process.
        - refusal_payload: structured refusal response when blocked, otherwise {}.
        - reasons: list of matched policy reasons when blocked.
    """
    payload_text = _flatten_payload(data)
    reasons = [f"blocked_term:{term}" for term in _BLOCKLIST if term in payload_text]

    if reasons:
        refusal_payload: Dict[str, Any] = {
            "status": "refused",
            "message": "Request blocked by Burn Wall safety policies.",
        }
        return False, refusal_payload, reasons

    return True, {}, []
