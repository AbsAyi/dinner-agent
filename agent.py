from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file: {path}") from e


def load_data() -> Dict[str, Any]:
    """
    Loads environment variables and reads required JSON data files.

    Returns a dict with keys: 'pantry', 'meals', 'preferences', 'state'.
    """

    # Load environment variables from the repo's `.env` file (if present).
    try:
        from dotenv import load_dotenv
    except ImportError as e:
        raise ImportError(
            "python-dotenv is required. Install it with: pip install python-dotenv"
        ) from e

    repo_root = Path(__file__).resolve().parent
    env_path = repo_root / ".env"
    # If `.env` doesn't exist, python-dotenv will just not load anything.
    load_dotenv(dotenv_path=str(env_path) if env_path.exists() else None)

    data_dir = repo_root / "data"
    pantry_path = data_dir / "pantry_inventory.json"
    meals_path = data_dir / "meal_log.json"
    preferences_path = data_dir / "family_preferences.json"
    state_path = data_dir / "agent_state.json"

    # Validate required data files exist before attempting to parse them.
    missing = [
        str(p)
        for p in (pantry_path, meals_path, preferences_path, state_path)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required data file(s):\n" + "\n".join(missing)
        )

    # Pantry inventory: current stock, confirmation queue, and grocery list signals.
    pantry = _read_json(pantry_path)

    # Meal log: recent meals used to avoid repeats and infer usage patterns.
    meals = _read_json(meals_path)

    # Family preferences: dietary, household likes/dislikes, and constraints for recs.
    preferences = _read_json(preferences_path)

    # Agent state: what the agent is currently doing/planning (e.g., tonight's meal).
    state = _read_json(state_path)

    return {
        "pantry": pantry,
        "meals": meals,
        "preferences": preferences,
        "state": state,
    }


def get_dinner_recommendation(data: Dict[str, Any]) -> str:
    # Load the system prompt that defines output format and decision rules.
    repo_root = Path(__file__).resolve().parent
    system_prompt_path = repo_root / "prompts" / "dinner_brief.txt"
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"Missing system prompt file: {system_prompt_path}")
    system_prompt = system_prompt_path.read_text(encoding="utf-8")

    # Extract pantry items with sufficient confidence (> 0.5).
    pantry_items: Dict[str, Any] = (
        data.get("pantry", {}).get("items", {}) if isinstance(data.get("pantry"), dict) else {}
    )
    confident_pantry_items = []
    for item_key, item in pantry_items.items():
        if not isinstance(item, dict):
            continue
        confidence = item.get("confidence")
        try:
            confidence_f = float(confidence)
        except (TypeError, ValueError):
            continue
        if confidence_f > 0.5:
            name = item.get("name") or item_key
            confident_pantry_items.append((str(name), confidence_f))
    confident_pantry_items.sort(key=lambda x: x[1], reverse=True)

    # Extract the last 3 meals (date + meal name) from the meal log.
    meal_entries = []
    meals_obj = data.get("meals")
    if isinstance(meals_obj, dict):
        meal_entries = meals_obj.get("meals", []) or []
    if not isinstance(meal_entries, list):
        meal_entries = []

    def _parse_date(d: Any) -> datetime:
        # ISO dates (YYYY-MM-DD) are expected in the JSON files.
        if isinstance(d, str):
            return datetime.strptime(d, "%Y-%m-%d")
        return datetime.min

    normalized_meals = []
    for entry in meal_entries:
        if not isinstance(entry, dict):
            continue
        normalized_meals.append(
            {
                "date": entry.get("date"),
                "meal": entry.get("meal"),
            }
        )
    normalized_meals.sort(key=lambda m: _parse_date(m.get("date")), reverse=True)
    last_3_meals = normalized_meals[:3]

    # Determine tonight's energy level (default to "medium" when missing).
    state_obj = data.get("state", {})
    energy_level = "medium"
    if isinstance(state_obj, dict):
        tonight_obj = state_obj.get("tonight", {})
        if isinstance(tonight_obj, dict) and tonight_obj.get("energy_level"):
            energy_level = str(tonight_obj.get("energy_level"))
        elif state_obj.get("energy_level"):
            energy_level = str(state_obj.get("energy_level"))

    # Pull kid preferences from family preferences (aggregate across kids).
    preferences_obj = data.get("preferences", {})
    kid_prefs = []
    if isinstance(preferences_obj, dict):
        household = preferences_obj.get("household", {})
        if isinstance(household, dict):
            kids = household.get("kids", [])
            if isinstance(kids, list):
                for kid in kids:
                    if not isinstance(kid, dict):
                        continue
                    prefs = kid.get("preferences") or []
                    dislikes = kid.get("dislikes") or []
                    notes = kid.get("notes")
                    kid_prefs.append(
                        {
                            "preferences": prefs if isinstance(prefs, list) else [str(prefs)],
                            "dislikes": dislikes if isinstance(dislikes, list) else [str(dislikes)],
                            "notes": notes,
                        }
                    )

    # Build the user message with the required context for the model.
    pantry_lines = (
        "\n".join([f"- {name}: {conf:.2f}" for name, conf in confident_pantry_items])
        if confident_pantry_items
        else "- (none with confidence > 0.5)"
    )
    meals_lines = (
        "\n".join([f"- {m.get('date')}: {m.get('meal')}" for m in last_3_meals])
        if last_3_meals
        else "- (no meals found)"
    )
    if kid_prefs:
        kids_lines = []
        for idx, kp in enumerate(kid_prefs, start=1):
            pref_list = ", ".join([str(x) for x in kp.get("preferences", [])]) or "(none)"
            dislike_list = ", ".join([str(x) for x in kp.get("dislikes", [])]) or "(none)"
            kids_lines.append(f"- Kid {idx} preferences: {pref_list}")
            kids_lines.append(f"  Kid {idx} dislikes: {dislike_list}")
            if kp.get("notes"):
                kids_lines.append(f"  Notes: {kp['notes']}")
        # Flatten to keep formatting stable (avoid indentation complexity in prompts).
        kids_block = "\n".join(kids_lines)
    else:
        kids_block = "- (no kid preferences found)"

    user_message = "\n".join(
        [
            "PANTRY (confidence > 0.5):",
            pantry_lines,
            "",
            "LAST 3 MEALS:",
            meals_lines,
            "",
            f"TONIGHT ENERGY LEVEL: {energy_level}",
            "",
            "KID PREFERENCES:",
            kids_block,
        ]
    )

    # Call the Anthropic API and return Claude's text response.
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise ImportError(
            "anthropic library is required. Install it with: pip install anthropic"
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ANTHROPIC_API_KEY in environment. Ensure `.env` is present and loaded."
        )

    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    # The SDK returns a list of content blocks; extract all text parts.
    texts = []
    for block in getattr(resp, "content", []):
        text = getattr(block, "text", None)
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts).strip()


def _count_items(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (dict, list, tuple, set)):
        return len(value)
    return 1


if __name__ == "__main__":
    data = load_data()
    recommendation = get_dinner_recommendation(data)
    print("TONIGHT'S RECOMMENDATION:")
    print(recommendation)
    from sms import send_sms

    sms_ok = send_sms(recommendation)
    if sms_ok:
        print("Recommendation sent via SMS")
    else:
        print("SMS failed - check sms.py")

