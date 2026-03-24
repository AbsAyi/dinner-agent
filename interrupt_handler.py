from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _extract_json_object(text: str) -> str:
    """
    Best-effort extraction of the first JSON object found in `text`.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in Anthropic response.")
    return text[start : end + 1]


def _normalize_pantry_data(data: Any) -> Dict[str, Any]:
    """
    Ensure `pantry_update` payloads match one of the supported shapes.

    Item updates:
      {"action": "update_item", "item": str, "change": str}

    Manual additions:
      {"action": "add_items", "items": [{"name": str}, ...]}
    """
    if not isinstance(data, dict):
        return {}

    action = data.get("action")
    if isinstance(action, str):
        action = action.strip().lower()
    if action == "add_items":
        raw_items = data.get("items")
        items: List[Dict[str, str]] = []
        if isinstance(raw_items, list):
            for it in raw_items:
                if isinstance(it, dict):
                    n = it.get("name")
                    if n is not None and str(n).strip():
                        items.append({"name": str(n).strip()})
                elif isinstance(it, str) and it.strip():
                    items.append({"name": it.strip()})
        return {"action": "add_items", "items": items}

    if action == "update_item":
        item_s = str(data.get("item", "")).strip()
        change_s = str(data.get("change", "")).strip()
        if item_s and change_s:
            return {
                "action": "update_item",
                "item": item_s,
                "change": change_s,
            }
        return {}

    # Legacy: item + change without action
    item_s = str(data.get("item", "")).strip()
    change_s = str(data.get("change", "")).strip()
    if item_s and change_s and "items" not in data:
        return {
            "action": "update_item",
            "item": item_s,
            "change": change_s,
        }

    # Partial add_items without action key
    raw_action = data.get("action")
    if isinstance(data.get("items"), list) and not raw_action:
        nested = _normalize_pantry_data({"action": "add_items", "items": data["items"]})
        if nested.get("items") is not None:
            return nested

    return {}


def handle_interrupt(message_body: str) -> Dict[str, Any]:
    """
    Classify an incoming SMS message intent and extract relevant data.

    Does not modify pantry data — only classifies and structures the message.

    Returns:
      {
        "intent": <category>,
        "data": <extracted fields>,
        "confidence": <float 0..1>,
        "reply": <string>
      }

    For ``pantry_update``, ``data`` is one of:

    - Stock / status change (one item, free-text change):
      ``{"action": "update_item", "item": "...", "change": "..."}``

    - Manual additions (one or more new items the user is declaring):
      ``{"action": "add_items", "items": [{"name": "..."}, ...]}``
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
        "- pantry_update: anything about what's in the pantry — stock/status changes OR manually adding items they bought/have.\n"
        "- energy_checkin_response: explicit energy indicators (😵/💪) or low/medium/high.\n"
        "- confirmation_response: yes/no to pantry confirmations (Y/N/yes/no).\n"
        "- pause_agent: skip this week, grandma's until monday, etc.\n"
        "- unknown: anything else.\n\n"
        "For `pantry_update`, set `data` to EXACTLY one of these shapes (include `\"action\"`):\n\n"
        "A) Stock / location / level change for ONE item (paraphrase the user's words in `change`):\n"
        '   {\"action\": \"update_item\", \"item\": \"eggs\", \"change\": \"out / none left\"}\n'
        '   {\"action\": \"update_item\", \"item\": \"chicken\", \"change\": \"moved to freezer\"}\n'
        '   {\"action\": \"update_item\", \"item\": \"milk\", \"change\": \"running low\"}\n\n'
        "B) Manual additions — user is adding new pantry items (shopping, restock, or \"we have X, Y, Z\"):\n"
        '   {\"action\": \"add_items\", \"items\": [{\"name\": \"soy sauce\"}]}\n'
        '   {\"action\": \"add_items\", \"items\": [{\"name\": \"olive oil\"}, {\"name\": \"taco seasoning\"}, {\"name\": \"broth\"}]}\n'
        '   {\"action\": \"add_items\", \"items\": [{\"name\": \"frozen chicken nuggets\"}]}\n\n'
        "How to choose A vs B:\n"
        "- Use `update_item` when they report status/location/quantity for something already tracked (out of, low, empty, finished, moved to freezer/fridge, thawing, etc.).\n"
        "- Use `add_items` when they say add/put/bought/got/picked up/we have/we now have — listing things to add to the pantry list.\n"
        "- If they mix both in one message, prefer the clearest primary request; if equal, use `update_item` for the first stock issue mentioned.\n\n"
        "Guidance for other intents `data` (not exhaustive):\n"
        "- skip_tonight: {\"meal\": <string or null>, \"source\": \"takeout\"}\n"
        "- early_decision: {\"meal\": <string>}\n"
        "- energy_checkin_response: {\"energy_level\": \"low|medium|high\"}\n"
        "- confirmation_response: {\"answer\": true|false}\n"
        "- pause_agent: {\"until\": <string or null>}\n"
        "- unknown: {}\n\n"
        "Make `reply` warm and short (1-2 sentences). Acknowledge what you'll do without claiming the pantry file was already updated.\n"
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
    if str(intent) == "pantry_update":
        normalized = _normalize_pantry_data(data)
        data = normalized if normalized else data
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
        # pantry_update: stock / status
        "We're out of eggs.",
        "Moved the chicken to the freezer.",
        "Milk is low.",
        # pantry_update: manual adds
        "add soy sauce to pantry",
        "We have olive oil, taco seasoning, and broth",
        "add frozen nuggets",
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
