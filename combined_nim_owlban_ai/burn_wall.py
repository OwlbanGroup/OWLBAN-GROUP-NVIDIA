"""
Burn Wall (Breach Containment) - Allow-by-default jailbreak/prompt-injection guardrail.

Goal:
- Attack grok-injection / jailbreak style requests from being processed.
- Do NOT block normal safe inference requests.
- Operate on arbitrary `request.data` shapes (dict/list/str/number/etc).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Dict
import re


@dataclass(frozen=True)
class BurnWallDecision:
    allowed: bool
    reasons: Tuple[str, ...] = ()
    refusal_type: str = "refusal"


class BurnWall:
    """
    Detects likely jailbreak / prompt-injection attempts by scanning
    string fields in the input payload (recursively).
    """

    # Conservative keyword sets (tuneable)
    _INDICATORS: List[Tuple[str, re.Pattern[str]]] = [
        (
            "attempts_to_override_instructions",
            re.compile(r"\b(ignore|override|disregard)\b.*\b(instructions|system|developer|policy)\b", re.I),
        ),
        (
            "asks_for_system_prompt",
            re.compile(r"\b(system prompt|developer message|hidden prompt|jailbreak instructions)\b", re.I),
        ),
        (
            "jailbreak_keywords",
            re.compile(r"\b(jailbreak|pwned|bypass|ungrounded|roleplay|do anything now)\b", re.I),
        ),
        (
            "prompt_injection_tooling",
            re.compile(r"\b(prompt injection|indirect prompt|instructions via context)\b", re.I),
        ),
        (
            "data_exfiltration_style",
            re.compile(r"\b(exfiltrate|leak|reveal|print the full|dump the full)\b", re.I),
        ),
        (
            "request_for_secrets",
            re.compile(r"\b(api key|secret|token|password|credential)\b", re.I),
        ),
    ]

    # Refusal response templates
    _REFUSAL_MESSAGE = {
        "refusal_type": "breach_containment",
        "message": (
            "Request blocked by breach containment guardrail: suspected prompt-injection/jailbreak behavior."
        ),
        "allowed": False,
    }

    def inspect(self, payload: Any) -> BurnWallDecision:
        """
        Scan payload. Returns allowed=False if likely jailbreak/prompt-injection is detected.
        """
        reasons: List[str] = []
        text_blobs = self._collect_strings(payload)

        joined = "\n".join(text_blobs)
        for reason_name, pattern in self._INDICATORS:
            if pattern.search(joined):
                reasons.append(reason_name)

        # Heuristic: if extremely long or suspicious structure, consider it.
        # (Still allow-by-default; only refuse if combined with injection keywords.)
        if not reasons:
            return BurnWallDecision(allowed=True, reasons=())

        # If multiple reasons found, refuse.
        return BurnWallDecision(allowed=False, reasons=tuple(sorted(set(reasons))))

    def _collect_strings(self, obj: Any) -> List[str]:
        out: List[str] = []
        visited_ids = set()

        def walk(x: Any, depth: int) -> None:
            if id(x) in visited_ids:
                return
            visited_ids.add(id(x))

            if depth > 12:
                return

            if x is None:
                return
            if isinstance(x, str):
                if x.strip():
                    out.append(x)
                return
            if isinstance(x, dict):
                for k, v in x.items():
                    # include keys too (often contain prompt injection payloads)
                    if isinstance(k, str) and k.strip():
                        out.append(k)
                    walk(v, depth + 1)
                return
            if isinstance(x, (list, tuple, set)):
                for item in x:
                    walk(item, depth + 1)
                return
            # For numbers/bools: ignore
            return

        walk(obj, 0)
        return out

    def refusal_payload(self) -> Dict[str, Any]:
        return dict(self._REFUSAL_MESSAGE)


def run_burn_wall(payload: Any) -> Tuple[bool, Dict[str, Any], Tuple[str, ...]]:
    """
    Convenience wrapper:
      - returns (allowed, response_payload, reasons)
    response_payload is only meaningful when allowed=False.
    """
    bw = BurnWall()
    decision = bw.inspect(payload)
    if decision.allowed:
        return True, {}, decision.reasons
    return False, bw.refusal_payload(), decision.reasons
