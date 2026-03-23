from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple


def _extract_json_object(text: str) -> str:
    """
    Best-effort extraction of the first JSON object found in `text`.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in Anthropic response.")
    return text[start : end + 1]


def handle_interrupt(message_body: str) -> Dict[str, Any]:
    """
    Classify an incoming SMS message intent and extract relevant data.

    Returns:
      {
        "intent": <category>,
        "data": <extracted fields>,
        "confidence": <float 0..1>,
        "reply": <string>
      }
    """

    # Load environment variables (same pattern as `agent.py`).
    try:
        from dotenv import load_dotenv
    except ImportError as e:
        raise ImportError(
            "python-dotenv is required. Install it with: pip install python-dotenv"
        ) from e

    repo_root = Path(__file__).resolve().parent
    env_path = repo_root / ".env"
    load_dotenv(dotenv_path=str(env_path) if env_path.exists() else None)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ANTHROPIC_API_KEY in environment. Ensure `.env` is present and loaded."
        )

    # Import Anthropic SDK lazily so the module can be imported without deps installed.
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise ImportError(
            "anthropic library is required. Install it with: pip install anthropic"
        ) from e

    client = Anthropic(api_key=api_key)

    categories = [
        "skip_tonight",
        "early_decision",
        "pantry_update",
        "energy_checkin_response",
        "confirmation_response",
        "pause_agent",
        "unknown",
    ]

    system_prompt = (
        "You are a dinner-agent SMS intent classifier.\n"
        "Classify the user's message into exactly one category from this list:\n"
        f"{', '.join(categories)}\n\n"
        "Extract any relevant fields from the message into a JSON object called `data`.\n"
        "Return ONLY valid JSON (no markdown, no preamble) with this exact schema:\n"
        "{\n"
        '  "intent": string,\n'
        '  "data": object,\n'
        '  "confidence": number,  // float 0..1\n'
        '  "reply": string        // short SMS reply\n'
        "}\n\n"
        "Rules for `intent`:\n"
        "- skip_tonight: pizza night, ordering out, eating out.\n"
        "- early_decision: they choose a meal for tonight (e.g., tacos, ramen).\n"
        "- pantry_update: stock changes (out of X, moved X to thaw).\n"
        "- energy_checkin_response: explicit energy indicators (😵/💪) or low/medium/high.\n"
        "- confirmation_response: yes/no to pantry confirmations (Y/N/yes/no).\n"
        "- pause_agent: skip this week, grandma's until monday, etc.\n"
        "- unknown: anything else.\n\n"
        "Guidance for `data` examples (not exhaustive):\n"
        "- skip_tonight: {\"meal\": <string or null>, \"source\": \"takeout\"}\n"
        "- early_decision: {\"meal\": <string>}\n"
        "- pantry_update: {\"item\": <string>, \"change\": <string>}\n"
        "- energy_checkin_response: {\"energy_level\": \"low|medium|high\"}\n"
        "- confirmation_response: {\"answer\": true|false}\n"
        "- pause_agent: {\"until\": <string or null>}\n"
        "- unknown: {}\n\n"
        "Make `reply` friendly and aligned with the intent. Keep it concise (1-2 sentences).\n"
        "If unsure, set `intent` to unknown and confidence low."
    )

    user_prompt = f"SMS message:\n{message_body}"

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Extract text content from the response.
    text_parts = []
    for block in getattr(resp, "content", []):
        t = getattr(block, "text", None)
        if isinstance(t, str):
            text_parts.append(t)
    raw = "\n".join(text_parts).strip()

    try:
        json_text = _extract_json_object(raw)
        parsed = json.loads(json_text)
    except Exception:
        # Fallback: return unknown with minimal info if the model didn't comply.
        return {
            "intent": "unknown",
            "data": {},
            "confidence": 0.1,
            "reply": "Sorry, I couldn't understand that. Can you re-send what you meant?",
        }

    # Normalize expected keys/types a bit.
    intent = parsed.get("intent", "unknown")
    data = parsed.get("data", {}) if isinstance(parsed.get("data", {}), dict) else {}
    confidence = parsed.get("confidence", 0.2)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.2
    reply = parsed.get("reply") or ""

    return {
        "intent": str(intent),
        "data": data,
        "confidence": max(0.0, min(1.0, confidence)),
        "reply": str(reply),
    }


if __name__ == "__main__":
    test_messages: Tuple[str, ...] = (
        # skip_tonight
        "Pizza night — we're ordering out tonight.",
        # early_decision
        "I want tacos tonight please!",
        # pantry_update
        "We're out of eggs.",
        # energy_checkin_response
        "low",
        # confirmation_response
        "Y",
    )

    for msg in test_messages:
        print("=" * 60)
        print(f"SMS in: {msg}")
        try:
            result = handle_interrupt(msg)
            print("Result:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error handling message: {e}")
