from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from interrupt_handler import handle_interrupt
from inventory import (
    add_manual_items,
    manually_update_item,
    interpret_inventory_update_message,
    _find_pantry_match_key,
    _infer_category_from_name,
    _now_ts,
    _read_json,
    _repo_root,
)


def _synthetic_inventory_message(item_label: str, change: str) -> str:
    """Build a short phrase the rule-based interpreter understands."""
    il = item_label.strip()
    cl = change.lower()
    if any(x in cl for x in ("out", "none left", "empty", "gone", "finished")):
        return f"we're out of {il}"
    if "low" in cl:
        return f"{il} is low"
    if any(x in cl for x in ("moved", "move", "put")) and re.search(
        r"\b(freezer|fridge|pantry|counter)\b", cl
    ):
        m = re.search(r"\b(freezer|fridge|pantry|counter)\b", cl)
        if m:
            return f"moved {il} to {m.group(1)}"
    return f"{il} {change}"


def _updates_from_change_heuristic(change: str, existing_notes: Any) -> Dict[str, Any]:
    """Map free-text change notes to pantry fields when rules/LLM did not."""
    now_ts = _now_ts()
    cl = change.lower()
    if any(x in cl for x in ("out", "none left", "empty", "gone", "finished")):
        return {"estimated_quantity": 0, "last_confirmed": now_ts, "confidence": 0.95}
    if "low" in cl:
        return {"estimated_quantity": "low", "last_confirmed": now_ts, "confidence": 0.5}
    if ("freezer" in cl or "fridge" in cl or "pantry" in cl or "counter" in cl) and any(
        x in cl for x in ("moved", "move", "put", "to", "in")
    ):
        m = re.search(r"\b(freezer|fridge|pantry|counter)\b", cl)
        if m:
            return {
                "storage_location": m.group(1),
                "last_confirmed": now_ts,
                "confidence": 0.95,
            }
    if any(x in cl for x in ("restock", "bought", "picked up")):
        return {
            "estimated_quantity": "restocked",
            "last_purchased": now_ts,
            "last_confirmed": now_ts,
            "confidence": 0.95,
        }
    merged = change
    if isinstance(existing_notes, str) and existing_notes.strip():
        merged = f"{existing_notes.strip()} | {change}"
    return {"notes": merged, "last_confirmed": now_ts, "confidence": 0.95}


def _apply_pantry_update_item(
    item_label: str, change: str, message_body: str
) -> Dict[str, Any]:
    """
    Resolve item + change onto a pantry row using the rule interpreter first,
    then fuzzy key match + heuristics.
    """
    pantry_path = _repo_root() / "data" / "pantry_inventory.json"
    try:
        pantry = _read_json(pantry_path)
    except Exception as e:
        return {"applied": False, "error": "read_failed", "detail": str(e)}

    pantry_items = pantry.get("items", {}) if isinstance(pantry, dict) else {}
    if not isinstance(pantry_items, dict):
        pantry_items = {}

    # 1) Original SMS (works when the user text matches interpreter patterns)
    interp = interpret_inventory_update_message(message_body, pantry)
    if interp.get("intent") in ("update_item", "confirm_item") and interp.get("item_key"):
        ok = manually_update_item(interp["item_key"], interp["updates"])
        return {
            "applied": ok,
            "item_key": interp["item_key"],
            "method": "interpret_inventory_update_message_full",
            "interpretation": interp,
        }

    # 2) Synthetic line from classifier's item + change
    synth = _synthetic_inventory_message(item_label, change)
    interp2 = interpret_inventory_update_message(synth, pantry)
    if interp2.get("intent") in ("update_item", "confirm_item") and interp2.get("item_key"):
        ok = manually_update_item(interp2["item_key"], interp2["updates"])
        return {
            "applied": ok,
            "item_key": interp2["item_key"],
            "method": "interpret_inventory_update_message_synthetic",
            "synthetic_message": synth,
            "interpretation": interp2,
        }

    # 3) Fuzzy pantry key + heuristic updates
    cat = _infer_category_from_name(item_label)
    key = _find_pantry_match_key(item_label, cat, pantry_items)
    if not key:
        return {
            "applied": False,
            "error": "no_item_match",
            "item_label": item_label,
            "change": change,
        }

    entry = pantry_items.get(key, {})
    existing_notes = entry.get("notes") if isinstance(entry, dict) else None
    updates = _updates_from_change_heuristic(change, existing_notes)
    ok = manually_update_item(key, updates)
    return {
        "applied": ok,
        "item_key": key,
        "method": "fuzzy_match_heuristic",
        "updates": updates,
    }


def process_sms_message(message_body: str) -> Dict[str, Any]:
    """
    Classify an SMS with ``handle_interrupt``, then apply pantry side-effects
    when ``intent == pantry_update``.

    Adds ``inventory_result`` for pantry actions (or error info). Does not
    swallow classifier errors from ``handle_interrupt`` (API/env issues).
    """
    result = handle_interrupt(message_body)

    if result.get("intent") != "pantry_update":
        return result

    data = result.get("data", {})
    if not isinstance(data, dict):
        result["inventory_result"] = {"error": "invalid_data"}
        return result

    action = data.get("action")

    if action == "add_items":
        raw_items = data.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = []
        items_list: List[Dict[str, Any]] = []
        for it in raw_items:
            if isinstance(it, dict) and it.get("name"):
                items_list.append(dict(it))
            elif isinstance(it, str) and it.strip():
                items_list.append({"name": it.strip()})
        summary = add_manual_items(items_list)
        result["inventory_result"] = summary

    elif action == "update_item":
        item_s = str(data.get("item", "")).strip()
        change_s = str(data.get("change", "")).strip()
        if item_s and change_s:
            result["inventory_result"] = _apply_pantry_update_item(
                item_s, change_s, message_body
            )
        else:
            result["inventory_result"] = {"error": "missing_item_or_change"}

    else:
        result["inventory_result"] = {
            "error": "unknown_pantry_action",
            "action": action,
        }

    return result


if __name__ == "__main__":
    demos = (
        "We're out of eggs.",
        "add soy sauce to pantry",
        "We have olive oil, taco seasoning, and broth",
    )
    for msg in demos:
        print("=" * 60)
        print("SMS:", msg)
        try:
            out = process_sms_message(msg)
            print(json.dumps(out, indent=2))
        except Exception as exc:
            print("Error:", exc)
