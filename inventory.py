from __future__ import annotations

import base64
import json
import os
import re
import hashlib
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_ts() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _parse_json_array_from_claude_response(raw: str) -> Optional[Any]:
    """
    Parse JSON from Claude output that may include markdown fences or prose.

    Attempts in order:
    1. stripped raw
    2. strip leading ``` / ```json and trailing ```
    3. substring from first '[' to last ']' (on fence-stripped, then on original)

    Returns parsed value or None only if every attempt fails.
    """
    text = (raw or "").strip()
    if not text:
        return None

    candidates: List[str] = []
    seen: set[str] = set()

    def add(c: str) -> None:
        c = c.strip()
        if not c or c in seen:
            return
        seen.add(c)
        candidates.append(c)

    add(text)

    # Remove opening ```json / ``` and closing ```
    fence_stripped = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
    fence_stripped = re.sub(r"\s*```\s*$", "", fence_stripped, count=1)
    fence_stripped = fence_stripped.strip()
    add(fence_stripped)

    def bracket_slice(s: str) -> Optional[str]:
        lo = s.find("[")
        hi = s.rfind("]")
        if lo == -1 or hi == -1 or hi <= lo:
            return None
        return s[lo : hi + 1]

    for base in (fence_stripped, text):
        br = bracket_slice(base)
        if br:
            add(br)

    last_err: Optional[BaseException] = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError as e:
            last_err = e
            continue

    if last_err is not None:
        print(
            f"[parse_grocery_items] JSONDecodeError after all cleanup attempts: {last_err!r}"
        )
    return None


def _preprocess_grocery_email_for_parse(text: str, *, max_chars: int = 8000) -> str:
    """
    Clean forwarded / quoted grocery email noise and prefer the itemized section.

    - Removes common forward/thread scaffolding and noisy headers (not Subject:).
    - Strips markdown-style quote prefixes.
    - Replaces very long URLs (typical tracking links).
    - Tries to start the prompt at retailer item/summary anchors.
    - Truncates to ``max_chars``.
    """
    if not isinstance(text, str):
        text = str(text)

    s = text.replace("\r\n", "\n").replace("\r", "\n")

    forward_line_res = [
        re.compile(r"^\s*[-_=]{3,}\s*forwarded message\s*[-_=]{3,}\s*$", re.I),
        re.compile(r"^\s*begin forwarded message\s*$", re.I),
        re.compile(r"^\s*end forwarded message\s*$", re.I),
        re.compile(r"^\s*forwarded message\s*$", re.I),
        re.compile(r"^\s*-+\s*original message\s*-+\s*$", re.I),
        re.compile(r"^\s*original message\s*$", re.I),
    ]
    header_noise_res = [
        re.compile(
            r"^(reply-to|return-path|list-unsubscribe|list-id|list-help|precedence|"
            r"x-mailer|x-google|x-ms|message-id|mime-version|content-transfer-encoding|"
            r"dkim-signature|authentication-results)\s*:",
            re.I,
        ),
    ]

    kept: List[str] = []
    for line in s.split("\n"):
        st = line.rstrip()
        if any(p.match(st) for p in forward_line_res):
            continue
        if any(p.match(st) for p in header_noise_res):
            continue
        kept.append(line)
    s = "\n".join(kept)

    # Strip leading '>' quote markers (email / markdown quoting).
    unquoted: List[str] = []
    for line in s.split("\n"):
        q = line
        while q.startswith(">"):
            q = q[1:]
            if q.startswith(" "):
                q = q[1:]
        unquoted.append(q)
    s = "\n".join(unquoted)

    # Long tracking / redirect URLs → drop (keep short same-line links readable).
    s = re.sub(r"https?://[^\s<>\[\]()\"']{90,}", " ", s)

    # Decorative whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()

    # Prefer itemized / order sections (most specific first).
    section_patterns = [
        r"\bitems found\b",
        r"\badjustments\b",
        r"\byour items\b",
        r"\bitems in your order\b",
        r"\bitems in your delivery\b",
        r"\bitems in this (?:order|delivery)\b",
        r"\border items\b",
        r"\bitem details\b",
        r"\bline items\b",
        r"(?m)^\s*items\s*:\s*$",
        r"(?m)^\s*items\s*-\s*$",
        r"(?m)^\s*items\s*$",
        r"\bwhat(?:'s| is) in your (?:order|bag|cart)\b",
        r"\border summary\b",
        r"\bshopping list\b",
        r"(?m)^\s*(?:qty|quantity)\b[^\n]{0,40}\b(?:item|description|product)\b",
        r"\bsubtotal\b",
        r"\border total\b",
    ]
    for pat in section_patterns:
        m = re.search(pat, s, re.IGNORECASE | re.MULTILINE)
        if m:
            ctx_start = max(0, m.start() - 450)
            snap = s.rfind("\n\n", ctx_start, m.start() + 1)
            if snap != -1 and snap >= ctx_start and (m.start() - snap) < 1200:
                ctx_start = snap + 2
            s = s[ctx_start:].lstrip()
            break

    if len(s) > max_chars:
        s = s[:max_chars]
    return s.rstrip()


def _load_env_like_agent_py() -> None:
    # Loads environment variables from the repo's `.env` file (if present).
    try:
        from dotenv import load_dotenv
    except ImportError as e:
        raise ImportError(
            "python-dotenv is required. Install it with: pip install python-dotenv"
        ) from e

    root = _repo_root()
    env_path = root / ".env"
    load_dotenv(dotenv_path=str(env_path) if env_path.exists() else None)


def _anthropic_client():
    _load_env_like_agent_py()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ANTHROPIC_API_KEY in environment. Ensure `.env` is present and loaded."
        )
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise ImportError(
            "anthropic library is required. Install it with: pip install anthropic"
        ) from e
    return Anthropic(api_key=api_key)


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found.")
    return text[start : end + 1]


def _extract_json_array(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found.")
    return text[start : end + 1]


def classify_grocery_email(email_text: str) -> Dict[str, Any]:
    """
    Classify a grocery-related retailer email conservatively.

    Returns a strict dict with this shape:

      {
        "email_type": "receipt" | "order_confirmed" | "shopping_invite" | "delivery_update" | "unknown",
        "retailer": str | null,
        "order_id": str | null,
        "should_parse_items": bool,
        "should_update_inventory": bool,
        "confidence": float,
        "reason": str
      }

    Notes:
    - Only finalized purchase records like receipts should set should_update_inventory=True.
    - If the LLM call fails or returns malformed JSON, returns a safe fallback and does not raise.
    """

    fallback: Dict[str, Any] = {
        "email_type": "unknown",
        "retailer": None,
        "order_id": None,
        "should_parse_items": False,
        "should_update_inventory": False,
        "confidence": 0.0,
        "reason": "Failed to classify email",
    }

    allowed_types = {"receipt", "order_confirmed", "shopping_invite", "delivery_update", "unknown"}

    system = (
        "You are a conservative classifier for grocery-related retailer emails.\n"
        "Return ONLY valid JSON. Do not include markdown or extra text.\n\n"
        "Classify the email into exactly one email_type from:\n"
        "- receipt\n"
        "- order_confirmed\n"
        "- shopping_invite\n"
        "- delivery_update\n"
        "- unknown\n\n"
        "Rules:\n"
        "- Only finalized purchase records like receipts should set should_update_inventory=true.\n"
        "- order_confirmed, shopping_invite, and delivery_update must set should_update_inventory=false.\n"
        "- should_parse_items should be true only if line items are actually present in this email.\n\n"
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "email_type": string,\n'
        '  "retailer": string | null,\n'
        '  "order_id": string | null,\n'
        '  "should_parse_items": boolean,\n'
        '  "should_update_inventory": boolean,\n'
        '  "confidence": number,\n'
        '  "reason": string\n'
        "}\n\n"
        "Be conservative: if unsure, use email_type=unknown, should_parse_items=false, should_update_inventory=false,\n"
        "and confidence <= 0.4."
    )

    try:
        client = _anthropic_client()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=350,
            system=system,
            messages=[{"role": "user", "content": email_text}],
        )

        text_parts: List[str] = []
        for b in getattr(resp, "content", []):
            t = getattr(b, "text", None)
            if isinstance(t, str):
                text_parts.append(t)
        raw = "\n".join(text_parts).strip()

        parsed = json.loads(_extract_json_object(raw))
        if not isinstance(parsed, dict):
            return fallback

        email_type = parsed.get("email_type", "unknown")
        if not isinstance(email_type, str) or email_type not in allowed_types:
            email_type = "unknown"

        retailer = parsed.get("retailer", None)
        if retailer is not None and not isinstance(retailer, str):
            retailer = None

        order_id = parsed.get("order_id", None)
        if order_id is not None and not isinstance(order_id, str):
            order_id = None

        should_parse_items = bool(parsed.get("should_parse_items", False))
        should_update_inventory = bool(parsed.get("should_update_inventory", False))

        confidence = parsed.get("confidence", 0.0)
        try:
            confidence_f = float(confidence)
        except (TypeError, ValueError):
            confidence_f = 0.0
        confidence_f = max(0.0, min(1.0, confidence_f))

        reason = parsed.get("reason", "")
        if not isinstance(reason, str) or not reason.strip():
            reason = "No reason provided"

        # Enforce conservative inventory update rules.
        if email_type != "receipt":
            should_update_inventory = False
        if should_update_inventory is True and email_type != "receipt":
            should_update_inventory = False

        # If not a receipt, we should almost never parse items.
        if email_type != "receipt":
            should_parse_items = False

        return {
            "email_type": email_type,
            "retailer": retailer,
            "order_id": order_id,
            "should_parse_items": should_parse_items,
            "should_update_inventory": should_update_inventory,
            "confidence": confidence_f,
            "reason": reason.strip(),
        }
    except Exception:
        return fallback


def parse_grocery_items(email_text: str) -> List[Dict[str, Any]]:
    """
    Extract line items conservatively from an email.

    Returns a conservative list of parsed line items with this exact shape:

      [
        {
          "raw_name": str,
          "normalized_name": str,
          "quantity": float | null,
          "unit": str | null,
          "category": "produce" | "protein" | "dairy_eggs" | "bread_grains" | "pantry_staple" | "frozen" |
                      "beverage" | "household_nonfood" | "personal_care" | "unknown",
          "pantry_eligible": bool,
          "inventory_action": "add" | "ignore" | "review"
        }
      ]

    If parsing fails, returns [].
    """
    allowed_categories = {
        "produce",
        "protein",
        "dairy_eggs",
        "bread_grains",
        "pantry_staple",
        "frozen",
        "beverage",
        "household_nonfood",
        "personal_care",
        "unknown",
    }
    allowed_actions = {"add", "ignore", "review"}

    def _none_if_empty(value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        return value if value else None

    def _simple_normalize_name(raw_name: str) -> str:
        s = raw_name.lower()
        s = re.sub(r"[\(\)\[\]\{\},]", " ", s)
        s = re.sub(r"[^a-z0-9\s&'-]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()

        # Remove common packaging/grade/unit tokens and a few common brand tokens.
        stop = {
            "grade",
            "a",
            "large",
            "extra",
            "ct",
            "count",
            "pack",
            "pk",
            "oz",
            "lb",
            "lbs",
            "g",
            "kg",
            "ml",
            "l",
            "fl",
            "pint",
            "qt",
            "quart",
            "gal",
            "gallon",
            "organic",
            "fresh",
            "frozen",
            "boneless",
            "skinless",
            "value",
            "gather",
            "good",
            "great",
            "signature",
            "kirkland",
        }

        parts = [p for p in s.replace("&", " ").split(" ") if p and p not in stop]
        if not parts:
            return s

        # Keep two-word proteins (e.g., "chicken thighs") when possible.
        if len(parts) >= 2 and parts[-1] in {"thighs", "breasts", "breast", "wings", "tenders", "ground"}:
            return f"{parts[-2]} {parts[-1]}".strip()
        return parts[-1]

    system = (
        "Extract grocery line-items conservatively from a retailer email.\n"
        "Return ONLY a JSON array. No markdown, no preamble, no extra text.\n\n"
        "Each array element MUST be an object with exactly these keys:\n"
        "- raw_name: string\n"
        "- normalized_name: string (simple pantry name, e.g. \"Grade A Eggs 12 ct\" -> \"eggs\")\n"
        "- quantity: number or null\n"
        "- unit: string or null (e.g. \"lb\", \"oz\", \"count\", \"pint\")\n"
        "- category: one of [produce, protein, dairy_eggs, bread_grains, pantry_staple, frozen, beverage, household_nonfood, personal_care, unknown]\n"
        "- pantry_eligible: boolean\n"
        "- inventory_action: one of [add, ignore, review]\n\n"
        "Conservative rules:\n"
        "- Food and beverages can be pantry_eligible=true.\n"
        "- Household items, toiletries, beauty products, and cleaning supplies should usually be pantry_eligible=false and inventory_action=\"ignore\".\n"
        "- If uncertain whether an item belongs in pantry, set inventory_action=\"review\" and pantry_eligible=false.\n"
        "- Extract quantity as a number when possible and unit separately when possible.\n"
    )

    MAX_EMAIL_CHARS = 8000
    if not isinstance(email_text, str):
        email_text = str(email_text)
    original_len = len(email_text)
    prompt_text = _preprocess_grocery_email_for_parse(
        email_text, max_chars=MAX_EMAIL_CHARS
    )
    if len(prompt_text) != original_len or len(prompt_text) == MAX_EMAIL_CHARS:
        print(
            f"[parse_grocery_items] preprocessed email for Claude: "
            f"{original_len} -> {len(prompt_text)} chars (cap={MAX_EMAIL_CHARS})"
        )

    preview = prompt_text[:500]
    print(
        f"[parse_grocery_items] email to Claude: first 500 chars of {len(prompt_text)} total:\n"
        f"{preview!r}"
    )

    try:
        client = _anthropic_client()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1600,
            system=system,
            messages=[{"role": "user", "content": prompt_text}],
        )
        text_parts: List[str] = []
        for b in getattr(resp, "content", []):
            t = getattr(b, "text", None)
            if isinstance(t, str):
                text_parts.append(t)
        raw = "\n".join(text_parts).strip()

        print(f"[parse_grocery_items] Claude raw response ({len(raw)} chars):\n{raw}")

        items = _parse_json_array_from_claude_response(raw)
        if items is None:
            print(
                f"[parse_grocery_items] raw text that failed to parse after all attempts "
                f"({len(raw)} chars):\n{raw}"
            )
            return []

        if not isinstance(items, list):
            print(
                f"[parse_grocery_items] parsed JSON is not a list (got {type(items).__name__}), returning []"
            )
            return []

        cleaned: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue

            raw_name = _none_if_empty(it.get("raw_name"))
            if raw_name is None:
                continue

            normalized_name = _none_if_empty(it.get("normalized_name")) or _simple_normalize_name(raw_name)
            if not normalized_name:
                continue

            unit = _none_if_empty(it.get("unit"))

            quantity_val = it.get("quantity", None)
            quantity: Optional[float]
            if quantity_val is None or quantity_val == "":
                quantity = None
            else:
                try:
                    quantity = float(quantity_val)
                except (TypeError, ValueError):
                    quantity = None

            category = _none_if_empty(it.get("category")) or "unknown"
            category = category.lower()
            if category not in allowed_categories:
                category = "unknown"

            inventory_action = _none_if_empty(it.get("inventory_action")) or "review"
            inventory_action = inventory_action.lower()
            if inventory_action not in allowed_actions:
                inventory_action = "review"

            pantry_eligible = bool(it.get("pantry_eligible", False))

            # Conservative enforcement: if it's clearly non-food categories, force ignore.
            if category in {"household_nonfood", "personal_care"}:
                pantry_eligible = False
                inventory_action = "ignore"

            cleaned.append(
                {
                    "raw_name": raw_name,
                    "normalized_name": normalized_name.lower(),
                    "quantity": quantity,
                    "unit": unit,
                    "category": category,
                    "pantry_eligible": pantry_eligible,
                    "inventory_action": inventory_action,
                }
            )

        return cleaned
    except Exception as e:
        print(f"[parse_grocery_items] unexpected error: {type(e).__name__}: {e!r}")
        return []


def _normalize_key(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _best_fuzzy_match(
    candidate_name: str, pantry_items: Dict[str, Any], threshold: float = 0.82
) -> Tuple[Optional[str], float]:
    cand = _normalize_key(candidate_name)
    best_key = None
    best_score = 0.0
    for key, item in pantry_items.items():
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or key)
        score = SequenceMatcher(None, cand, _normalize_key(name)).ratio()
        if score > best_score:
            best_score = score
            best_key = key
    if best_score >= threshold:
        return best_key, best_score
    return None, best_score


def _category_defaults(category: str) -> Dict[str, Any]:
    defaults: Dict[str, Dict[str, Any]] = {
        "produce": {"perishability": "very_high", "physical_decay_rate": 0.22, "usage_decay_rate": 0.15, "storage_location": "fridge"},
        "protein": {"perishability": "high", "physical_decay_rate": 0.18, "usage_decay_rate": 0.2, "storage_location": "fridge"},
        "dairy_eggs": {"perishability": "high", "physical_decay_rate": 0.08, "usage_decay_rate": 0.14, "storage_location": "fridge"},
        "bread_grains": {"perishability": "medium", "physical_decay_rate": 0.06, "usage_decay_rate": 0.11, "storage_location": "pantry"},
        "pantry_staple": {"perishability": "very_low", "physical_decay_rate": 0.005, "usage_decay_rate": 0.07, "storage_location": "pantry"},
        "frozen": {"perishability": "very_low", "physical_decay_rate": 0.01, "usage_decay_rate": 0.12, "storage_location": "freezer"},
        "beverage": {"perishability": "low", "physical_decay_rate": 0.02, "usage_decay_rate": 0.06, "storage_location": "pantry"},
    }
    return defaults.get(category, {"perishability": "medium", "physical_decay_rate": 0.05, "usage_decay_rate": 0.1, "storage_location": "pantry"})


def _find_pantry_match_key(
    norm_name: str, category: str, pantry_items: Dict[str, Any]
) -> Optional[str]:
    """
    Find an existing pantry item key for a normalized product name (exact + fuzzy).
    Same rules as receipt-driven updates.
    """
    norm = _normalize_key(norm_name)
    # 1) Prefer exact normalized_name/key match.
    for key, entry in pantry_items.items():
        if not isinstance(entry, dict):
            continue
        entry_norm = _normalize_key(str(entry.get("normalized_name") or key))
        if entry_norm == norm:
            return key

    # 2) Conservative fuzzy matching with category guardrails.
    best_key = None
    best_score = 0.0
    for key, entry in pantry_items.items():
        if not isinstance(entry, dict):
            continue
        entry_name = str(entry.get("normalized_name") or entry.get("name") or key)
        score = SequenceMatcher(None, norm, _normalize_key(entry_name)).ratio()

        entry_category = str(entry.get("category") or "unknown")
        if category in {"protein", "produce"} and entry_category == category and score < 0.9:
            continue
        if score > best_score:
            best_score = score
            best_key = key

    return best_key if best_score >= 0.86 else None


_MANUAL_ALLOWED_CATEGORIES = frozenset(
    {
        "produce",
        "protein",
        "dairy_eggs",
        "bread_grains",
        "pantry_staple",
        "frozen",
        "beverage",
        "household_nonfood",
        "personal_care",
        "unknown",
    }
)


def _coerce_manual_category(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    c = str(raw).strip().lower()
    if c not in _MANUAL_ALLOWED_CATEGORIES:
        return None
    return c


def _infer_category_from_name(name: str) -> str:
    """Pick a sensible category when the user did not supply one."""
    n = _normalize_key(name)

    if any(
        x in n
        for x in (
            "frozen",
            "nugget",
            "nuggets",
            "ice cream",
            "popsicle",
            "waffle",
            "fries",
        )
    ):
        return "frozen"
    if any(
        x in n
        for x in (
            "milk",
            "cheese",
            "cream",
            "butter",
            "yogurt",
            "egg",
            "eggs",
        )
    ):
        return "dairy_eggs"
    if any(
        x in n
        for x in (
            "chicken",
            "beef",
            "pork",
            "fish",
            "salmon",
            "shrimp",
            "turkey",
            "steak",
            "ground",
        )
    ):
        return "protein"
    if any(
        x in n
        for x in (
            "lettuce",
            "spinach",
            "tomato",
            "onion",
            "apple",
            "banana",
            "potato",
            "carrot",
            "broccoli",
        )
    ):
        return "produce"
    if any(x in n for x in ("bread", "tortilla", "bagel", "bun", "roll")):
        return "bread_grains"
    if any(
        x in n
        for x in (
            "soda",
            "juice",
            "water",
            "coffee",
            "tea",
            "beer",
            "wine",
        )
    ):
        return "beverage"

    if any(
        x in n
        for x in (
            "sauce",
            "oil",
            "olive",
            "soy",
            "vinegar",
            "seasoning",
            "spice",
            "salt",
            "pepper",
            "flour",
            "sugar",
            "rice",
            "pasta",
            "broth",
            "stock",
            "honey",
            "mustard",
            "ketchup",
            "mayo",
            "mayonnaise",
            "seasoning",
            "alfredo",
            "marinara",
            "cereal",
            "oat",
        )
    ):
        return "pantry_staple"

    return "pantry_staple"


def _household_staple_for_manual(name: str, category: str) -> bool:
    """
    Pantry/cooking basics -> True; snacks, frozen convenience, fresh meat/produce -> False.
    """
    n = _normalize_key(name)

    if category in {"frozen"}:
        return False
    if category in {"produce", "protein"}:
        return False

    non_staple = (
        "nugget",
        "nuggets",
        "pizza roll",
        "ice cream",
        "cookie",
        "candy",
        "chip",
        "chips",
        "popcorn",
        "hot pocket",
        "burrito frozen",
    )
    if any(x in n for x in non_staple):
        return False

    staple_markers = (
        "soy sauce",
        "olive oil",
        "vegetable oil",
        "canola",
        "coconut oil",
        "sesame oil",
        "salt",
        "pepper",
        "sugar",
        "flour",
        "rice",
        "pasta",
        "broth",
        "stock",
        "honey",
        "vinegar",
        "milk",
        "egg",
        "butter",
        "taco seasoning",
        "seasoning",
        "worcestershire",
        "mustard",
        "ketchup",
        "mayonnaise",
        "alfredo",
        "marinara",
        "tomato sauce",
        "bread",
        "tortilla",
    )
    if any(x in n for x in staple_markers):
        return True

    if category in {"pantry_staple", "dairy_eggs", "bread_grains"}:
        return True

    return False


def _unique_pantry_item_key(norm_name: str, pantry_items: Dict[str, Any]) -> str:
    base = re.sub(r"\s+", "_", _normalize_key(norm_name))[:64] or "item"
    base = re.sub(r"_+", "_", base).strip("_") or "item"
    key = base
    i = 2
    while key in pantry_items:
        key = f"{base}_{i}"
        i += 1
    return key


def _format_manual_quantity(q: Any) -> str:
    if q is None:
        return "1"
    if isinstance(q, bool):
        return "1"
    if isinstance(q, (int, float)):
        if float(q) == int(q):
            return str(int(q))
        return str(q)
    s = str(q).strip()
    return s if s else "1"


def add_manual_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Seed or update pantry items from manual entry (SMS, UI), not from a receipt.

    Each input dict must include ``name``. Optional: ``category``, ``quantity``,
    ``unit``, ``storage_location``, ``notes``.

    Returns summary: ``added``, ``updated``, ``count_added``, ``count_updated`` (item keys).
    """
    if not items:
        return {"added": [], "updated": [], "count_added": 0, "count_updated": 0}

    root = _repo_root()
    pantry_path = root / "data" / "pantry_inventory.json"
    pantry = _read_json(pantry_path)

    meta = pantry.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
        pantry["_meta"] = meta

    pantry_items = pantry.get("items")
    if not isinstance(pantry_items, dict):
        pantry_items = {}
        pantry["items"] = pantry_items

    ts = _now_ts()
    added: List[str] = []
    updated: List[str] = []

    for raw in items:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        display_name = name.strip()

        cat_raw = _coerce_manual_category(raw.get("category"))
        category = cat_raw if cat_raw is not None else _infer_category_from_name(display_name)

        match_key = _find_pantry_match_key(display_name, category, pantry_items)

        if match_key:
            entry = pantry_items.get(match_key)
            if not isinstance(entry, dict):
                entry = {}
                pantry_items[match_key] = entry

            if "quantity" in raw and raw.get("quantity") is not None:
                entry["estimated_quantity"] = _format_manual_quantity(raw.get("quantity"))
            if "unit" in raw and raw.get("unit") is not None:
                u = raw.get("unit")
                if str(u).strip():
                    entry["quantity_unit"] = str(u).strip()

            entry["confidence"] = 0.95
            entry["last_confirmed"] = ts

            if "notes" in raw:
                n = raw.get("notes")
                if isinstance(n, str) and n.strip():
                    entry["notes"] = n.strip()

            if "storage_location" in raw and raw.get("storage_location") is not None:
                loc = str(raw.get("storage_location")).strip().lower()
                if loc in {"fridge", "freezer", "pantry", "counter", "unknown"}:
                    entry["storage_location"] = loc

            if "category" in raw and cat_raw is not None:
                entry["category"] = cat_raw
                defaults = _category_defaults(cat_raw)
                entry["perishability"] = defaults["perishability"]
                entry["physical_decay_rate"] = defaults["physical_decay_rate"]
                entry["usage_decay_rate"] = defaults["usage_decay_rate"]

            updated.append(match_key)
            continue

        # New item
        key = _unique_pantry_item_key(display_name, pantry_items)
        defaults = _category_defaults(category)
        storage = defaults["storage_location"]
        if raw.get("storage_location") is not None:
            loc = str(raw.get("storage_location")).strip().lower()
            if loc in {"fridge", "freezer", "pantry", "counter", "unknown"}:
                storage = loc

        notes_val: Any = None
        if "notes" in raw:
            n = raw.get("notes")
            if isinstance(n, str) and n.strip():
                notes_val = n.strip()

        unit_s: Optional[str] = None
        if raw.get("unit") is not None and str(raw.get("unit")).strip():
            unit_s = str(raw.get("unit")).strip()

        staple = _household_staple_for_manual(display_name, category)

        if "quantity" in raw and raw.get("quantity") is not None:
            qty_new = _format_manual_quantity(raw.get("quantity"))
        else:
            qty_new = "1"

        pantry_items[key] = {
            "name": display_name,
            "normalized_name": _normalize_key(display_name),
            "category": category,
            "perishability": defaults["perishability"],
            "confidence": 0.95,
            "physical_decay_rate": defaults["physical_decay_rate"],
            "usage_decay_rate": defaults["usage_decay_rate"],
            "last_confirmed": ts,
            "last_purchased": None,
            "estimated_quantity": qty_new,
            "quantity_unit": unit_s,
            "storage_location": storage,
            "times_used_in_recs": 0,
            "household_staple": staple,
            "notes": notes_val,
        }
        added.append(key)

    if not added and not updated:
        return {"added": [], "updated": [], "count_added": 0, "count_updated": 0}

    meta["last_updated"] = ts
    _write_json(pantry_path, pantry)

    return {
        "added": added,
        "updated": updated,
        "count_added": len(added),
        "count_updated": len(updated),
    }


def _processed_at_sort_key(meta: Any) -> float:
    """Epoch seconds for ordering; missing/invalid timestamps sort oldest."""
    if not isinstance(meta, dict):
        return float("-inf")
    ts = meta.get("processed_at")
    if not isinstance(ts, str) or not ts.strip():
        return float("-inf")
    s = ts.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).timestamp()
    except ValueError:
        return float("-inf")


def _prune_processed_orders(
    processed_orders: Dict[str, Any], *, max_entries: int = 200
) -> None:
    """
    Keep only the ``max_entries`` most recent entries by ``processed_at``.
    Mutates ``processed_orders`` in place; does not touch other pantry keys.
    """
    if not isinstance(processed_orders, dict) or len(processed_orders) <= max_entries:
        return

    ranked = sorted(
        processed_orders.items(),
        key=lambda kv: _processed_at_sort_key(kv[1]),
        reverse=True,
    )
    keep = {k for k, _ in ranked[:max_entries]}
    for k in list(processed_orders.keys()):
        if k not in keep:
            del processed_orders[k]


def update_pantry_from_receipt(parsed_email: Dict[str, Any], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Update pantry inventory from parsed receipt email items.
    """
    root = _repo_root()
    pantry_path = root / "data" / "pantry_inventory.json"
    pantry = _read_json(pantry_path)

    meta = pantry.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
        pantry["_meta"] = meta

    processed_orders = pantry.get("_processed_orders")
    if not isinstance(processed_orders, dict):
        processed_orders = {}
        pantry["_processed_orders"] = processed_orders

    retailer = parsed_email.get("retailer") if isinstance(parsed_email.get("retailer"), str) else "unknown"
    retailer_key = _normalize_key(retailer).replace(" ", "_") or "unknown"
    order_id = parsed_email.get("order_id")
    order_id = order_id.strip() if isinstance(order_id, str) and order_id.strip() else None
    email_type = parsed_email.get("email_type") if isinstance(parsed_email.get("email_type"), str) else "unknown"

    if order_id:
        processed_order_key = f"{retailer_key}_{order_id}"
    else:
        name_fingerprint = "|".join(
            sorted(
                _normalize_key(str(it.get("normalized_name") or it.get("raw_name") or ""))
                for it in items
                if isinstance(it, dict)
            )
        )
        seed = f"{retailer_key}|{email_type}|{name_fingerprint}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        processed_order_key = f"{retailer_key}_{digest}"

    if processed_order_key in processed_orders:
        return {
            "added": [],
            "updated": [],
            "ignored": [],
            "review": [],
            "skipped_duplicate": True,
            "processed_order_key": processed_order_key,
        }

    pantry_items = pantry.get("items")
    if not isinstance(pantry_items, dict):
        pantry_items = {}
        pantry["items"] = pantry_items

    added: List[str] = []
    updated: List[str] = []
    ignored: List[str] = []
    review: List[str] = []

    ts = _now_ts()

    def _get_qty_and_unit(it: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        quantity = it.get("quantity", None)
        unit = it.get("unit", None)
        unit_s = str(unit).strip() if unit is not None and str(unit).strip() else None
        if quantity is None or quantity == "":
            qty_s = "unknown"
        else:
            qty_s = str(quantity)
        return qty_s, unit_s

    for it in items:
        if not isinstance(it, dict):
            continue
        raw_name = str(it.get("raw_name") or "").strip()
        norm_name = str(it.get("normalized_name") or raw_name).strip()
        action = str(it.get("inventory_action") or "").lower()
        pantry_eligible = it.get("pantry_eligible") is True

        if action == "review":
            review.append(raw_name or norm_name)
            continue
        if (not pantry_eligible) or action == "ignore":
            ignored.append(raw_name or norm_name)
            continue
        if action != "add" or not pantry_eligible:
            ignored.append(raw_name or norm_name)
            continue

        if not norm_name:
            ignored.append(raw_name or "")
            continue

        category = str(it.get("category") or "unknown")
        qty_str, unit_s = _get_qty_and_unit(it)
        match_key = _find_pantry_match_key(norm_name, category, pantry_items)

        if match_key:
            entry = pantry_items.get(match_key)
            if not isinstance(entry, dict):
                entry = {}
                pantry_items[match_key] = entry
            entry["normalized_name"] = str(entry.get("normalized_name") or _normalize_key(norm_name))
            entry["last_purchased"] = ts
            entry["last_confirmed"] = ts
            entry["estimated_quantity"] = qty_str
            if unit_s:
                entry["quantity_unit"] = unit_s
            entry["confidence"] = 0.95
            updated.append(match_key)
        else:
            key = _unique_pantry_item_key(norm_name, pantry_items)

            defaults = _category_defaults(category)
            pantry_items[key] = {
                "name": norm_name,
                "normalized_name": _normalize_key(norm_name),
                "category": category,
                "perishability": defaults["perishability"],
                "confidence": 0.95,
                "physical_decay_rate": defaults["physical_decay_rate"],
                "usage_decay_rate": defaults["usage_decay_rate"],
                "last_confirmed": ts,
                "last_purchased": ts,
                "estimated_quantity": qty_str,
                "quantity_unit": unit_s,
                "storage_location": defaults["storage_location"],
                "times_used_in_recs": 0,
                "household_staple": False,
                "notes": None,
            }
            added.append(key)

    meta["last_updated"] = ts
    meta["last_grocery_order"] = ts
    processed_orders[processed_order_key] = {
        "retailer": retailer,
        "order_id": order_id,
        "processed_at": ts,
        "email_type": email_type,
    }
    _prune_processed_orders(processed_orders, max_entries=200)

    _write_json(pantry_path, pantry)

    return {
        "added": added,
        "updated": updated,
        "ignored": ignored,
        "review": review,
        "skipped_duplicate": False,
        "processed_order_key": processed_order_key,
    }


def manually_update_item(item_key: str, updates: Dict[str, Any]) -> bool:
    """
    Apply a manual update to a pantry item (e.g., "we're out of eggs").
    """
    allowed_fields = {
        "confidence",
        "estimated_quantity",
        "quantity_unit",
        "storage_location",
        "notes",
        "last_confirmed",
        "last_purchased",
        "perishability",
        "category",
    }
    allowed_locations = {"fridge", "freezer", "pantry", "counter", "unknown"}

    root = _repo_root()
    pantry_path = root / "data" / "pantry_inventory.json"

    try:
        pantry = _read_json(pantry_path)
    except Exception:
        return False

    items = pantry.get("items")
    if not isinstance(items, dict):
        return False
    if item_key not in items or not isinstance(items.get(item_key), dict):
        return False
    if not isinstance(updates, dict):
        return False

    entry = items[item_key]
    for field, value in updates.items():
        if field not in allowed_fields:
            continue

        if field == "confidence":
            try:
                conf = float(value)
            except (TypeError, ValueError):
                continue
            entry["confidence"] = max(0.0, min(1.0, conf))
            continue

        if field == "storage_location":
            loc = str(value).strip().lower() if value is not None else ""
            if loc not in allowed_locations:
                continue
            entry["storage_location"] = loc
            continue

        entry[field] = value

    ts = _now_ts()
    pantry_meta = pantry.get("_meta")
    if not isinstance(pantry_meta, dict):
        pantry_meta = {}
        pantry["_meta"] = pantry_meta
    pantry_meta["last_updated"] = ts

    try:
        _write_json(pantry_path, pantry)
    except Exception:
        return False
    return True


def interpret_inventory_update_message(message_text: str, pantry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interpret lightweight inventory update/confirmation SMS messages without using an LLM.

    Returns:
      {
        "intent": "update_item" | "confirm_item" | "unknown",
        "item_key": str | null,
        "updates": dict,
        "confidence": float,
        "reason": str
      }
    """
    base_unknown = {
        "intent": "unknown",
        "item_key": None,
        "updates": {},
        "confidence": 0.0,
        "reason": "No clear inventory intent or pantry item match",
    }
    if not isinstance(message_text, str) or not message_text.strip():
        return base_unknown

    pantry_items = pantry.get("items", {}) if isinstance(pantry, dict) else {}
    if not isinstance(pantry_items, dict) or not pantry_items:
        return base_unknown

    text = message_text.strip().lower()
    text_norm = _normalize_key(text)
    now_ts = _now_ts()

    def _extract_candidate_phrase(s: str) -> str:
        # Trim common leading verbs/pronouns to isolate item phrase.
        s = re.sub(
            r"^(we|were|weve|i|im|ive|still|just|the|a|an|our)\s+",
            "",
            s,
        ).strip()
        s = re.sub(
            r"^(out of|moved|move|put|bought more|bought|have|has|is|are)\s+",
            "",
            s,
        ).strip()
        # Stop at common trailing location/qualifier phrases.
        s = re.split(r"\b(to|in|into|on)\b", s)[0].strip()
        return s

    def _match_item_key(candidate_text: str) -> Optional[str]:
        cand = _normalize_key(candidate_text)
        if not cand:
            return None

        # Exact normalized_name match first.
        for key, entry in pantry_items.items():
            if not isinstance(entry, dict):
                continue
            n = _normalize_key(str(entry.get("normalized_name") or ""))
            if n and n == cand:
                return key

        # Token subset match (conservative).
        cand_tokens = set(cand.split())
        if cand_tokens:
            token_matches: List[str] = []
            for key, entry in pantry_items.items():
                if not isinstance(entry, dict):
                    continue
                n = _normalize_key(str(entry.get("normalized_name") or entry.get("name") or key))
                item_tokens = set(n.split())
                if cand_tokens.issubset(item_tokens) or item_tokens.issubset(cand_tokens):
                    token_matches.append(key)
            if len(token_matches) == 1:
                return token_matches[0]

        # Conservative fuzzy fallback.
        best_key = None
        best_score = 0.0
        for key, entry in pantry_items.items():
            if not isinstance(entry, dict):
                continue
            name = _normalize_key(str(entry.get("normalized_name") or entry.get("name") or key))
            score = SequenceMatcher(None, cand, name).ratio()
            if score > best_score:
                best_score = score
                best_key = key
        return best_key if best_score >= 0.86 else None

    patterns: List[Tuple[str, str, Dict[str, Any], float, str]] = []

    # we're out of eggs
    m_out = re.search(r"\b(out of|no more|ran out of)\s+([a-z0-9\s'&-]+)", text_norm)
    if m_out:
        item_phrase = _extract_candidate_phrase(m_out.group(2))
        patterns.append(
            (
                "update_item",
                item_phrase,
                {"estimated_quantity": 0, "last_confirmed": now_ts, "confidence": 0.99},
                0.95,
                "Detected out-of-stock wording",
            )
        )

    # milk is low
    m_low = re.search(r"\b([a-z0-9\s'&-]+?)\s+(is|are)\s+low\b", text_norm)
    if m_low:
        item_phrase = _extract_candidate_phrase(m_low.group(1))
        patterns.append(
            (
                "update_item",
                item_phrase,
                {"estimated_quantity": "low", "last_confirmed": now_ts, "confidence": 0.5},
                0.7,
                "Detected low-quantity wording",
            )
        )

    # moved chicken to freezer / put X in fridge/pantry/counter
    m_move = re.search(
        r"\b(moved|move|put)\s+([a-z0-9\s'&-]+?)\s+(to|in|into)\s+(freezer|fridge|pantry|counter)\b",
        text_norm,
    )
    if m_move:
        item_phrase = _extract_candidate_phrase(m_move.group(2))
        loc = m_move.group(4)
        patterns.append(
            (
                "update_item",
                item_phrase,
                {"storage_location": loc, "last_confirmed": now_ts},
                0.85,
                "Detected storage-location move wording",
            )
        )

    # we bought more tortillas
    m_bought = re.search(r"\b(bought|got)\s+(more\s+)?([a-z0-9\s'&-]+)", text_norm)
    if m_bought:
        item_phrase = _extract_candidate_phrase(m_bought.group(3))
        patterns.append(
            (
                "update_item",
                item_phrase,
                {"estimated_quantity": "restocked", "last_purchased": now_ts, "last_confirmed": now_ts, "confidence": 0.95},
                0.8,
                "Detected restock wording",
            )
        )

    # still have spinach
    m_have = re.search(r"\b(still have|have)\s+([a-z0-9\s'&-]+)", text_norm)
    if m_have:
        item_phrase = _extract_candidate_phrase(m_have.group(2))
        patterns.append(
            (
                "confirm_item",
                item_phrase,
                {"last_confirmed": now_ts, "confidence": 0.9},
                0.85,
                "Detected positive confirmation wording",
            )
        )

    if not patterns:
        return base_unknown

    # Evaluate candidates and keep the best confident match.
    best_result: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for intent, item_phrase, updates, conf, reason in patterns:
        item_key = _match_item_key(item_phrase)
        if not item_key:
            continue
        # Small boost for cleaner phrase length (less ambiguous).
        phrase_tokens = len(_normalize_key(item_phrase).split())
        score = conf + (0.05 if 1 <= phrase_tokens <= 3 else 0.0)
        if score > best_score:
            best_score = score
            best_result = {
                "intent": intent,
                "item_key": item_key,
                "updates": updates,
                "confidence": max(0.0, min(1.0, score)),
                "reason": reason,
            }

    return best_result if best_result else base_unknown


def get_confirmation_queue(max_items: int = 1) -> List[Dict[str, Any]]:
    """
    Return top items that should be confirmed next.

    Scoring factors:
    - lower confidence -> higher priority
    - higher perishability -> higher priority
    - more days since last_confirmed -> higher priority
    """
    root = _repo_root()
    pantry_path = root / "data" / "pantry_inventory.json"
    try:
        pantry = _read_json(pantry_path)
    except Exception:
        return []
    items = pantry.get("items", {})
    if not isinstance(items, dict):
        return []
    if max_items <= 0:
        return []

    perishability_weight = {
        "very_high": 1.0,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.3,
        "very_low": 0.15,
    }
    pantryish_categories = {"pantry_staple", "bread_grains"}

    def _parse_iso(ts: Any) -> Optional[datetime]:
        if not isinstance(ts, str) or not ts.strip():
            return None
        s = ts.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None

    def _days_since_last_confirmed(ts: Any) -> float:
        dt = _parse_iso(ts)
        if dt is None:
            return 30.0
        now = datetime.now(timezone.utc)
        delta = now - dt
        return max(0.0, delta.total_seconds() / 86400.0)

    def _build_question(name: str, category: str, storage: str) -> str:
        lower_name = name.lower()
        if category == "bread_grains":
            return f"Are {lower_name} running low?"
        if category in {"produce", "dairy_eggs", "protein"} and storage in {"fridge", "freezer"}:
            loc = "fridge" if storage == "fridge" else "freezer"
            return f"Do you still have {lower_name} in the {loc}?"
        return f"Still have {lower_name}?"

    candidates: List[Tuple[float, str, str, float, str, str]] = []
    for key, entry in items.items():
        if not isinstance(entry, dict):
            continue

        name = str(entry.get("name") or key)
        try:
            conf = float(entry.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        conf = max(0.0, min(1.0, conf))

        perishability = str(entry.get("perishability") or "medium").lower()
        p_weight = perishability_weight.get(perishability, 0.5)

        category = str(entry.get("category") or "unknown").lower()
        storage = str(entry.get("storage_location") or "unknown").lower()
        days_since = _days_since_last_confirmed(entry.get("last_confirmed"))
        days_factor = min(1.0, days_since / 14.0)

        # Higher means more urgent to confirm.
        score = ((1.0 - conf) * 0.55) + (p_weight * 0.30) + (days_factor * 0.15)

        # Prioritize perishables over pantry staples when otherwise similar.
        if category in pantryish_categories:
            score -= 0.05
        if category in {"produce", "protein", "dairy_eggs"}:
            score += 0.05

        if score <= 0.20:
            continue

        question = _build_question(name, category, storage)
        candidates.append((score, key, name, conf, question, category))

    candidates.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for _score, key, name, conf, question, _category in candidates[:max_items]:
        out.append(
            {
                "item_key": key,
                "name": name,
                "confidence": conf,
                "question": question,
            }
        )
    return out


def _decode_gmail_message_body(payload: Dict[str, Any]) -> str:
    """
    Extract readable text from a Gmail API message payload.

    Order:
    1. Concatenate all text/plain parts (recursive: multipart/alternative, mixed, etc.)
    2. If none, decode text/html parts and convert with html2text
    3. Handles nested multipart at any depth
    """
    import base64

    def decode_part_data(data_b64: str) -> str:
        if not data_b64:
            return ""
        pad = 4 - len(data_b64) % 4
        if pad != 4:
            data_b64 += "=" * pad
        try:
            return base64.urlsafe_b64decode(data_b64).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _html_to_plain(html: str) -> str:
        if not html.strip():
            return ""
        try:
            import html2text
        except ImportError:
            return ""
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        return h.handle(html).strip()

    plain_chunks: List[str] = []
    html_chunks: List[str] = []

    def walk(part: Dict[str, Any]) -> None:
        if not isinstance(part, dict):
            return
        mime = str(part.get("mimeType") or "").lower().strip()
        body = part.get("body") or {}
        data = body.get("data")

        if data:
            raw = decode_part_data(data)
            if mime.startswith("text/plain"):
                plain_chunks.append(raw)
            elif mime.startswith("text/html"):
                html_chunks.append(raw)

        for sub in part.get("parts") or []:
            if isinstance(sub, dict):
                walk(sub)

    if not isinstance(payload, dict):
        return ""

    walk(payload)

    plain = "\n".join(plain_chunks).strip()
    if plain:
        return plain

    html_combined = "\n".join(html_chunks).strip()
    if html_combined:
        converted = _html_to_plain(html_combined)
        if converted:
            return converted

    # Top-level single part (no nested parts list) — e.g. simple text/html only
    body = payload.get("body") or {}
    data = body.get("data")
    top_mime = str(payload.get("mimeType") or "").lower().strip()
    if data and not (payload.get("parts") or []):
        raw = decode_part_data(data)
        if top_mime.startswith("text/plain"):
            return raw.strip()
        if top_mime.startswith("text/html"):
            return _html_to_plain(raw)

    return ""


ALLOWED_COMMAND_SENDERS = {
    "bayiloge@gmail.com", "bayiloge@yahoo.com",  # Approved senders for command emails (must match Gmail fetch filter).
}

# Only messages matching this recency window are fetched (stops reprocessing old mail).
GMAIL_RECENCY_QUERY = "newer_than:2d"

# Command fetch uses one explicit From: address (must be in ALLOWED_COMMAND_SENDERS).
GMAIL_COMMAND_FETCH_FROM = "bayiloge@gmail.com"


def _gmail_receipt_fetch_query() -> str:
    return f"label:grocery-receipt -label:receipt-processed {GMAIL_RECENCY_QUERY}"


def _gmail_command_fetch_query() -> str:
    addr = GMAIL_COMMAND_FETCH_FROM.strip().lower()
    return f"from:{addr} -label:command-processed {GMAIL_RECENCY_QUERY}"


def _extract_message_headers(payload: Dict[str, Any]) -> Dict[str, str]:
    out = {"from": "", "subject": ""}
    headers = payload.get("headers") if isinstance(payload, dict) else []
    if not isinstance(headers, list):
        return out
    for h in headers:
        if not isinstance(h, dict):
            continue
        name = str(h.get("name") or "").strip().lower()
        val = str(h.get("value") or "").strip()
        if name == "from":
            out["from"] = val
        elif name == "subject":
            out["subject"] = val
    return out


def _sender_email(raw_from: str) -> str:
    if not isinstance(raw_from, str):
        return ""
    # "Name <user@example.com>" -> "user@example.com"
    m = re.search(r"<([^>]+)>", raw_from)
    if m:
        return m.group(1).strip().lower()
    return raw_from.strip().lower()


def _route_inbox_message(label_names: List[str], sender_email: str) -> str:
    labels = {str(x).strip().lower() for x in label_names if str(x).strip()}
    if "grocery-receipt" in labels:
        return "grocery_receipt"
    if sender_email in ALLOWED_COMMAND_SENDERS:
        return "command"
    return "ignore"


def _list_gmail_message_ids(service: Any, q: str) -> List[str]:
    """Paginate Gmail messages.list and return message ids only."""
    ids: List[str] = []
    page_token: Optional[str] = None
    while True:
        req = service.users().messages().list(userId="me", q=q, pageToken=page_token)
        res = req.execute()
        for ref in res.get("messages") or []:
            if isinstance(ref, dict) and ref.get("id"):
                ids.append(str(ref["id"]))
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return ids


def _merge_gmail_message_ids(receipt_ids: List[str], command_ids: List[str]) -> List[str]:
    """Preserve order: receipt query first, then command-only ids."""
    seen: set[str] = set()
    out: List[str] = []
    for mid in receipt_ids + command_ids:
        if mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


def _reply_subject_line(original_subject: str) -> str:
    o = (original_subject or "").strip()
    if not o:
        return "Re: (no subject)"
    if re.match(r"^re:\s*", o, re.IGNORECASE):
        return o
    return f"Re: {o}"


def _format_item_list_for_confirmation(names: List[str]) -> str:
    n = [str(x).strip() for x in names if str(x).strip()]
    if not n:
        return ""
    if len(n) == 1:
        return n[0]
    if len(n) == 2:
        return f"{n[0]} and {n[1]}"
    return f"{', '.join(n[:-1])}, and {n[-1]}"


def build_command_confirmation(
    result: Dict[str, Any],
    inventory_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a short natural confirmation for a successful command (email reply body).
    Returns empty string when no confirmation should be sent for this result.
    """
    cmd = result.get("command_result")
    if not isinstance(cmd, dict) or not cmd.get("success"):
        return ""
    intent = str(cmd.get("intent") or "")
    action = str(cmd.get("action_taken") or "")
    inv: Any = inventory_result if inventory_result is not None else cmd.get("inventory_result")
    if inv is not None and not isinstance(inv, dict):
        inv = {}

    if intent == "pantry_update" and action == "add_manual_items":
        names = cmd.get("added_items")
        if not isinstance(names, list):
            names = []
        joined = _format_item_list_for_confirmation([str(x) for x in names])
        if not joined:
            return "Got it — I updated your pantry."
        return f"Got it — I added {joined} to your pantry."

    if intent == "pantry_update" and action == "update_item":
        item = str(cmd.get("item_label") or "that item").strip()
        change_s = str(cmd.get("change_text") or "").lower()
        if any(x in change_s for x in ("out", "none left", "empty", "gone", "finished")):
            return f"Got it — I marked {item} as out."
        return f"Got it — I updated {item}."

    if intent == "skip_tonight":
        reason = str(cmd.get("skip_reason") or "").strip()
        if "pizza" in reason.lower():
            return "Got it — I marked tonight as pizza night."
        if reason:
            return f"Got it — I noted tonight: {reason}."
        return "Got it — I noted you'll skip tonight."

    if intent == "early_decision":
        meal = str(cmd.get("meal_chosen") or "").strip()
        if not meal:
            return ""
        ml = meal.lower()
        if "taco" in ml:
            return "Got it — I logged tacos for tonight."
        return f"Got it — I logged {meal} for tonight."

    return ""


def build_command_failure_message(command_result: Dict[str, Any]) -> str:
    """Short outbound text when command processing did not succeed."""
    if not isinstance(command_result, dict):
        return "Sorry — something went wrong with that command."
    act = str(command_result.get("action_taken") or "")
    if act == "handler_exception":
        return "Sorry — I couldn't parse that command right now. Please try again."
    if act == "invalid_update_item_payload":
        return "Sorry — I couldn't tell which item to update."
    if act == "invalid_add_items_payload":
        return "Sorry — I couldn't tell which items to add."
    if act == "missing_meal_in_early_decision":
        return "Sorry — I couldn't tell which meal you wanted."
    if act == "no_op_for_intent":
        return "Sorry — I didn't understand that command."
    if act == "update_item" and command_result.get("processing_status") == "failed":
        return "Sorry — I couldn't update that pantry item."
    if act == "add_manual_items" and command_result.get("processing_status") == "failed":
        return "Sorry — I couldn't add those items to the pantry."
    return "Sorry — something went wrong with that command."


def send_gmail_reply(
    service: Any,
    original_message_id: str,
    to_email: str,
    subject: str,
    body: str,
) -> None:
    """
    Send a plain-text reply from the authenticated Gmail account, threading when possible.
    """
    full = (
        service.users()
        .messages()
        .get(userId="me", id=original_message_id, format="full")
        .execute()
    )
    thread_id = full.get("threadId")
    payload = full.get("payload") or {}
    headers = payload.get("headers") or []
    msg_id_header = ""
    if isinstance(headers, list):
        for h in headers:
            if isinstance(h, dict) and str(h.get("name", "")).lower() == "message-id":
                msg_id_header = str(h.get("value") or "").strip()
                break

    mime = MIMEText(body, "plain", "utf-8")
    mime["to"] = to_email
    mime["subject"] = _reply_subject_line(subject)
    if msg_id_header:
        mime["In-Reply-To"] = msg_id_header
        mime["References"] = msg_id_header

    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("utf-8")
    send_body: Dict[str, Any] = {"raw": raw}
    if thread_id:
        send_body["threadId"] = thread_id
    service.users().messages().send(userId="me", body=send_body).execute()


def _gmail_send_reply_safe(
    service: Any,
    original_message_id: str,
    to_email: str,
    subject: str,
    body: str,
) -> bool:
    if not (to_email or "").strip():
        print("[gmail] skip reply: missing recipient address")
        return False
    try:
        send_gmail_reply(service, original_message_id, to_email.strip(), subject, body)
        return True
    except Exception as e:
        print(f"[gmail] reply send failed: {e}")
        return False


def _should_skip_claude_calls(route: str, sender_email: str, body_text: str) -> bool:
    """
    Safety gate before any Anthropic call (classify/parse/interrupt).
    - Never blocks grocery receipts on the word 'instruction' (retailer copy often contains it).
    """
    if route == "ignore":
        return True
    if route == "grocery_receipt":
        return False
    if route == "command":
        if "instruction" in body_text.lower() and sender_email not in ALLOWED_COMMAND_SENDERS:
            return True
    return False


def _updates_from_change_heuristic(change: str, existing_notes: Any) -> Dict[str, Any]:
    now_ts = _now_ts()
    cl = str(change or "").lower()
    if any(x in cl for x in ("out", "none left", "empty", "gone", "finished")):
        return {"estimated_quantity": 0, "last_confirmed": now_ts, "confidence": 0.95}
    if "low" in cl:
        return {"estimated_quantity": "low", "last_confirmed": now_ts, "confidence": 0.5}
    m = re.search(r"\b(freezer|fridge|pantry|counter)\b", cl)
    if m and any(x in cl for x in ("moved", "move", "put", "to", "in")):
        return {
            "storage_location": m.group(1),
            "last_confirmed": now_ts,
            "confidence": 0.95,
        }
    merged = str(change).strip()
    if isinstance(existing_notes, str) and existing_notes.strip():
        merged = f"{existing_notes.strip()} | {merged}"
    return {"notes": merged, "last_confirmed": now_ts, "confidence": 0.95}


def _apply_update_item_command(item_label: str, change: str, message_body: str) -> Dict[str, Any]:
    pantry_path = _repo_root() / "data" / "pantry_inventory.json"
    try:
        pantry = _read_json(pantry_path)
    except Exception as e:
        return {"applied": False, "error": "read_failed", "detail": str(e)}

    interp = interpret_inventory_update_message(message_body, pantry)
    if interp.get("intent") in {"update_item", "confirm_item"} and interp.get("item_key"):
        ok = manually_update_item(str(interp["item_key"]), dict(interp.get("updates") or {}))
        return {
            "applied": ok,
            "item_key": interp.get("item_key"),
            "method": "interpret_inventory_update_message",
        }

    pantry_items = pantry.get("items", {}) if isinstance(pantry, dict) else {}
    if not isinstance(pantry_items, dict):
        pantry_items = {}
    cat = _infer_category_from_name(item_label)
    key = _find_pantry_match_key(item_label, cat, pantry_items)
    if not key:
        return {"applied": False, "error": "no_item_match", "item": item_label}

    entry = pantry_items.get(key)
    existing_notes = entry.get("notes") if isinstance(entry, dict) else None
    updates = _updates_from_change_heuristic(change, existing_notes)
    ok = manually_update_item(key, updates)
    return {"applied": ok, "item_key": key, "method": "fuzzy_match_heuristic"}


def _upsert_tonight_state(*, status: str, meal: Optional[str], source: Optional[str], reason: Optional[str]) -> None:
    state_path = _repo_root() / "data" / "agent_state.json"
    try:
        state = _read_json(state_path)
    except Exception:
        state = {}
    if not isinstance(state, dict):
        state = {}
    tonight = state.get("tonight")
    if not isinstance(tonight, dict):
        tonight = {}
    tonight["status"] = status
    tonight["reason"] = reason
    tonight["decided_at"] = _now_ts()
    tonight["meal"] = meal
    tonight["source"] = source
    tonight["skip_checkin"] = bool(status == "skip_requested")
    state["tonight"] = tonight
    _write_json(state_path, state)


def _append_meal_log_for_today(meal: str, source: str) -> Dict[str, Any]:
    meal_path = _repo_root() / "data" / "meal_log.json"
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        log_doc = _read_json(meal_path)
    except Exception:
        log_doc = {"_meta": {"version": "1.0", "window_days": 7}, "meals": []}
    if not isinstance(log_doc, dict):
        log_doc = {"_meta": {"version": "1.0", "window_days": 7}, "meals": []}
    meals = log_doc.get("meals")
    if not isinstance(meals, list):
        meals = []
        log_doc["meals"] = meals
    for m in meals:
        if not isinstance(m, dict):
            continue
        if str(m.get("date") or "") == today and str(m.get("meal") or "").strip().lower() == meal.strip().lower():
            return {"logged": False, "reason": "duplicate_today"}
    meals.insert(
        0,
        {
            "date": today,
            "meal": meal,
            "category": "Unknown",
            "source": source,
            "energy_level": None,
            "ingredients_used": [],
            "kid_approved": None,
            "rating": None,
        },
    )
    _write_json(meal_path, log_doc)
    return {"logged": True}


def _parse_energy_from_reply_text(text: str) -> Optional[str]:
    """Map free-text reply to low | medium | high, or None if unclear."""
    raw = text or ""
    t = raw.strip().lower()
    if t in ("low", "medium", "high"):
        return t
    if re.search(r"\bhigh\b", t) or "💪" in raw:
        return "high"
    if re.search(r"\blow\b", t) or "😵" in raw:
        return "low"
    if re.search(r"\bmedium\b", t) or re.search(r"\bmid\b", t):
        return "medium"
    return None


def _coerce_energy_level_from_reply(data: Dict[str, Any], message_body: str) -> Optional[str]:
    """Prefer classifier `energy_level`; fall back to parsing the email body."""
    raw = data.get("energy_level")
    if raw is not None:
        s = str(raw).strip().lower()
        if s in ("low", "medium", "high"):
            return s
    return _parse_energy_from_reply_text(message_body)


def _ranked_option_to_stored_record(opt: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Shape stored in ``tonight.recommendation_options`` when recommendations are sent."""
    ct = str(opt.get("candidate_type") or "")
    src = "template" if ct == "template" else "inferred"
    cat = opt.get("category") or opt.get("cuisine") or "Unknown"
    return {
        "rank": rank,
        "meal_key": str(opt.get("meal_key") or ""),
        "display_name": str(opt.get("display_name") or ""),
        "category": str(cat),
        "source": src,
        "matched_items": list(opt.get("matched_items") or []),
        "missing_items": list(opt.get("missing_items") or []),
        "time_minutes": int(opt.get("time_minutes") or 25),
        "takeout": bool(opt.get("takeout")),
        "quick_steps": [str(s).strip() for s in (opt.get("quick_steps") or []) if str(s).strip()],
        "optional_matched_items": list(opt.get("optional_matched_items") or []),
    }


def _match_recommendation_option_by_text(text: str, options: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Match free text to a stored option via display_name or meal_key."""
    t = (text or "").strip().lower()
    if not t:
        return None
    for o in options:
        if not isinstance(o, dict):
            continue
        dn = str(o.get("display_name") or "").strip().lower()
        mk = str(o.get("meal_key") or "").strip().lower().replace("_", " ")
        mk_slug = str(o.get("meal_key") or "").strip().lower()
        if dn and (dn in t or t in dn):
            return o
        if mk and (mk in t or t in mk or mk_slug in t.replace(" ", "_")):
            return o
        for tok in re.findall(r"[a-z]{4,}", t):
            if len(tok) >= 4 and tok in dn:
                return o
    return None


def _append_meal_log_for_selection(
    display_name: str,
    category: str,
    source: str,
    ingredients_used: List[str],
    energy_level: Optional[str],
) -> Dict[str, Any]:
    """Append meal log entry; skip duplicate (same local date + same meal name)."""
    meal_path = _repo_root() / "data" / "meal_log.json"
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        log_doc = _read_json(meal_path)
    except Exception:
        log_doc = {"_meta": {"version": "1.0", "window_days": 7}, "meals": []}
    if not isinstance(log_doc, dict):
        log_doc = {"_meta": {"version": "1.0", "window_days": 7}, "meals": []}
    meals = log_doc.get("meals")
    if not isinstance(meals, list):
        meals = []
        log_doc["meals"] = meals
    dnl = display_name.strip().lower()
    for m in meals:
        if not isinstance(m, dict):
            continue
        if str(m.get("date") or "") == today and str(m.get("meal") or "").strip().lower() == dnl:
            print("[meal_log] duplicate meal for today; skipped")
            return {"logged": False, "reason": "duplicate_today"}
    meals.insert(
        0,
        {
            "date": today,
            "meal": display_name.strip(),
            "category": category or "Unknown",
            "source": source,
            "energy_level": energy_level,
            "ingredients_used": ingredients_used,
            "kid_approved": None,
            "rating": None,
        },
    )
    _write_json(meal_path, log_doc)
    print(f"[meal_log] logged meal: {display_name!r} ({today})")
    return {"logged": True}


def _meal_selection_followup_body(selected: Dict[str, Any]) -> str:
    """After user picks an option: details for cooked meals, short line for takeout."""
    takeout = bool(selected.get("takeout"))
    display_name = str(selected.get("display_name") or "Tonight")
    if takeout:
        return f"Got it — I logged {display_name} for tonight. Enjoy!"
    from agent import build_selected_meal_message

    opt = {
        "display_name": selected.get("display_name"),
        "meal_key": selected.get("meal_key"),
        "matched_items": selected.get("matched_items") or [],
        "optional_matched_items": selected.get("optional_matched_items") or [],
        "missing_items": selected.get("missing_items") or [],
        "quick_steps": selected.get("quick_steps") or [],
        "time_minutes": int(selected.get("time_minutes") or 25),
        "takeout": False,
    }
    return build_selected_meal_message(opt)


def _finalize_meal_selection_command(
    service: Any,
    msg_id: str,
    sender: str,
    subject: str,
    selected: Dict[str, Any],
    state_path: Path,
) -> Dict[str, Any]:
    """Update agent_state, meal_log, send follow-up email; returns command_output wrapper."""
    try:
        state = _read_json(state_path)
    except Exception:
        state = {}
    if not isinstance(state, dict):
        state = {}
    tonight = state.get("tonight")
    if not isinstance(tonight, dict):
        tonight = {}
    now_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    meal_key = str(selected.get("meal_key") or "")
    display_name = str(selected.get("display_name") or meal_key)
    tonight["status"] = "meal_confirmed"
    tonight["meal_selected"] = meal_key or display_name
    tonight["meal_selected_at"] = now_ts
    state["tonight"] = tonight
    _write_json(state_path, state)

    energy = tonight.get("energy_level")
    if energy is not None:
        energy_s = str(energy)
    else:
        energy_s = None

    cat = str(selected.get("category") or "Unknown")
    src = "takeout" if selected.get("takeout") else "cooked"
    ingredients = list(selected.get("matched_items") or [])
    log_result = _append_meal_log_for_selection(
        display_name=display_name,
        category=cat,
        source=src,
        ingredients_used=ingredients,
        energy_level=energy_s,
    )

    if log_result.get("logged") is False and log_result.get("reason") == "duplicate_today":
        body = (
            f"I already have {display_name} logged for today — no change to the meal log."
        )
    else:
        body = _meal_selection_followup_body(selected)
    ok = _gmail_send_reply_safe(service, msg_id, sender, subject, body)
    print("[meal] selection resolved; follow-up email sent" if ok else "[meal] follow-up email failed")

    return {
        "intent": "meal_selection",
        "data": {},
        "confidence": 1.0,
        "reply": "",
        "command_result": {
            "intent": "meal_selection",
            "action_taken": "meal_confirmed",
            "success": True,
            "processing_status": "ok",
            "meal_log_result": log_result,
            "selected_display_name": display_name,
        },
    }


def _try_meal_selection_command(
    service: Any,
    msg_id: str,
    sender: str,
    subject: str,
    body_text: str,
    interrupt_result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Resolve 1/2/3 or meal name against ``tonight.recommendation_options``.
    Returns a full ``_process_command_message``-shaped dict if handled, else None.
    """
    intent = str(interrupt_result.get("intent") or "unknown")
    data = interrupt_result.get("data") if isinstance(interrupt_result.get("data"), dict) else {}

    if intent in ("pantry_update", "pause_agent", "confirmation_response"):
        return None
    if intent == "energy_checkin_response":
        return None
    if intent == "skip_tonight":
        return None

    state_path = _repo_root() / "data" / "agent_state.json"
    try:
        state = _read_json(state_path)
    except Exception:
        state = {}
    tonight = state.get("tonight") if isinstance(state.get("tonight"), dict) else {}
    opts = tonight.get("recommendation_options")
    if not isinstance(opts, list):
        opts = []

    body_stripped = body_text.strip()

    def _clarification_email() -> str:
        return (
            "I couldn't match that to tonight's numbered options.\n\n"
            "Reply with 1, 2, or 3, or reply with the meal name from the list."
        )

    def _no_stored_list_error() -> str:
        return (
            "I don't have tonight's ranked dinner list yet (nothing stored to pick from). "
            "Send your energy level (low, medium, or high) first if you haven't, "
            "then choose from the recommendations I send."
        )

    # --- Numeric 1–3 only ---
    mnum = re.match(r"^\s*([123])\s*$", body_stripped)
    if mnum:
        if not opts:
            print("[meal] numeric selection but no recommendation_options stored")
            ok = _gmail_send_reply_safe(service, msg_id, sender, subject, _no_stored_list_error())
            return {
                "intent": "unknown",
                "data": {},
                "confidence": 0.0,
                "reply": "",
                "command_result": {
                    "intent": "unknown",
                    "action_taken": "meal_select_no_options_error",
                    "success": False,
                    "processing_status": "failed",
                },
            }
        rank = int(mnum.group(1))
        selected = next((o for o in opts if isinstance(o, dict) and int(o.get("rank") or -1) == rank), None)
        if not selected:
            ok = _gmail_send_reply_safe(service, msg_id, sender, subject, _clarification_email())
            print("[meal] numeric selection out of range; clarification sent")
            return {
                "intent": "meal_selection",
                "data": {},
                "confidence": 0.5,
                "reply": "",
                "command_result": {
                    "intent": "meal_selection",
                    "action_taken": "meal_selection_clarification",
                    "success": False,
                    "processing_status": "failed",
                },
            }
        print(f"[meal] numeric selection resolved: rank={rank} -> {selected.get('meal_key')}")
        return _finalize_meal_selection_command(service, msg_id, sender, subject, selected, state_path)

    if not opts:
        return None

    # --- Free-text match (including early_decision meal string) ---
    selected: Optional[Dict[str, Any]] = None
    if intent == "early_decision":
        meal = str(data.get("meal") or "").strip()
        if meal:
            selected = _match_recommendation_option_by_text(meal, opts)
        if not selected:
            selected = _match_recommendation_option_by_text(body_stripped, opts)
    else:
        selected = _match_recommendation_option_by_text(body_stripped, opts)

    if selected:
        print(f"[meal] text selection resolved -> {selected.get('meal_key')}")
        return _finalize_meal_selection_command(service, msg_id, sender, subject, selected, state_path)

    if intent == "early_decision" and opts:
        ok = _gmail_send_reply_safe(service, msg_id, sender, subject, _clarification_email())
        print("[meal] early_decision did not match options; clarification sent")
        return {
            "intent": "early_decision",
            "data": data,
            "confidence": float(interrupt_result.get("confidence") or 0.5),
            "reply": "",
            "command_result": {
                "intent": "early_decision",
                "action_taken": "meal_selection_clarification",
                "success": False,
                "processing_status": "failed",
            },
        }

    return None


def _run_energy_checkin_recommendation_flow(
    service: Any,
    msg_id: str,
    sender: str,
    subject: str,
    energy_level: str,
) -> bool:
    """
    Store energy, rank top dinner options (no meal_log), persist options to agent_state,
    send recommendation email in-thread, then set tonight.status = recommendation_sent.
    """
    state_path = _repo_root() / "data" / "agent_state.json"
    try:
        state = _read_json(state_path)
    except Exception:
        state = {}
    if not isinstance(state, dict):
        state = {}
    tonight = state.get("tonight")
    if not isinstance(tonight, dict):
        tonight = {}
    now_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    tonight["energy_level"] = energy_level
    tonight["status"] = "energy_received"
    tonight["energy_received_at"] = now_ts
    tonight["checkin_channel"] = "gmail"
    state["tonight"] = tonight
    _write_json(state_path, state)
    print(f"[energy] reply received; level={energy_level}")

    try:
        from agent import build_recommendation_message, rank_dinner_options

        top = rank_dinner_options(energy_level, max_results=3)
        stored = [
            _ranked_option_to_stored_record(opt, i + 1) for i, opt in enumerate(top)
        ]
        recommendation = build_recommendation_message(top, energy_level)
    except Exception as e:
        print(f"[energy] recommendation generation failed: {e}")
        err_body = (
            "I couldn't generate dinner ideas just now (something went wrong on my side). "
            "Try again in a minute, or resend your energy as low, medium, or high."
        )
        return bool(_gmail_send_reply_safe(service, msg_id, sender, subject, err_body))

    try:
        state2 = _read_json(state_path)
    except Exception:
        state2 = {}
    if not isinstance(state2, dict):
        state2 = {}
    tn = state2.get("tonight")
    if not isinstance(tn, dict):
        tn = {}
    sent_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    tn["recommendation_options"] = stored
    tn["status"] = "recommendation_sent"
    tn["recommendation_sent_at"] = sent_ts
    state2["tonight"] = tn
    _write_json(state_path, state2)
    print(f"[recommendations] stored {len(stored)} option(s) in agent_state")

    if not _gmail_send_reply_safe(service, msg_id, sender, subject, recommendation):
        return False
    print("[energy] recommendation email sent (in-thread reply)")
    return True


def _process_command_message(
    message_body: str,
    interrupt_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from interrupt_handler import handle_interrupt

    command_result: Dict[str, Any] = {
        "intent": "unknown",
        "action_taken": "none",
        "success": False,
        "processing_status": "failed",
        "added_items": [],
        "updated_items": [],
        "error": None,
    }

    try:
        result = interrupt_result if interrupt_result is not None else handle_interrupt(message_body)
    except Exception as e:
        command_result.update(
            {
                "intent": "error",
                "action_taken": "handler_exception",
                "success": False,
                "processing_status": "failed",
                "error": str(e),
            }
        )
        return {
            "intent": "error",
            "data": {},
            "confidence": 0.0,
            "reply": "",
            "command_result": command_result,
        }

    intent = str(result.get("intent") or "unknown")
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    command_result["intent"] = intent

    if intent == "pantry_update":
        action = data.get("action")
        if action == "add_items":
            raw_items = data.get("items", [])
            items_list: List[Dict[str, Any]] = []
            if isinstance(raw_items, list):
                for it in raw_items:
                    if isinstance(it, dict) and it.get("name"):
                        items_list.append({"name": str(it["name"]).strip()})
                    elif isinstance(it, str) and it.strip():
                        items_list.append({"name": it.strip()})
            if not items_list:
                command_result.update(
                    {
                        "action_taken": "invalid_add_items_payload",
                        "success": False,
                        "processing_status": "failed",
                    }
                )
            else:
                summary = add_manual_items(items_list)
                added_names: List[str] = []
                for it in items_list:
                    if isinstance(it, dict) and it.get("name"):
                        added_names.append(str(it["name"]).strip())
                    elif isinstance(it, str) and it.strip():
                        added_names.append(it.strip())
                ok = int(summary.get("count_added") or 0) + int(summary.get("count_updated") or 0) > 0
                upd_keys = summary.get("updated") if isinstance(summary.get("updated"), list) else []
                command_result.update(
                    {
                        "action_taken": "add_manual_items",
                        "inventory_result": summary,
                        "added_items": added_names,
                        "updated_items": [str(x) for x in upd_keys],
                        "success": ok,
                        "processing_status": "ok" if ok else "failed",
                    }
                )
        elif action == "update_item":
            item_s = str(data.get("item") or "").strip()
            change_s = str(data.get("change") or "").strip()
            if item_s and change_s:
                update_result = _apply_update_item_command(item_s, change_s, message_body)
                applied = update_result.get("applied") is True
                command_result.update(
                    {
                        "action_taken": "update_item",
                        "inventory_result": update_result,
                        "item_label": item_s,
                        "change_text": change_s,
                        "updated_items": [item_s] if applied else [],
                        "success": applied,
                        "processing_status": "ok" if applied else "failed",
                    }
                )
            else:
                command_result.update(
                    {
                        "action_taken": "invalid_update_item_payload",
                        "success": False,
                        "processing_status": "failed",
                    }
                )
    elif intent == "skip_tonight":
        reason = str(data.get("meal") or "skip requested").strip()
        _upsert_tonight_state(status="skip_requested", meal=None, source="email_command", reason=reason)
        command_result.update(
            {
                "action_taken": "update_agent_state_skip",
                "skip_reason": reason,
                "success": True,
                "processing_status": "ok",
            }
        )
    elif intent == "early_decision":
        meal = str(data.get("meal") or "").strip()
        state_path_early = _repo_root() / "data" / "agent_state.json"
        try:
            st_early = _read_json(state_path_early)
            ropts = (
                st_early.get("tonight", {}).get("recommendation_options")
                if isinstance(st_early.get("tonight"), dict)
                else None
            )
        except Exception:
            ropts = None
        if isinstance(ropts, list) and len(ropts) > 0:
            # Ranked recommendations active — meal choice + meal_log go through selection flow only.
            command_result.update(
                {
                    "action_taken": "no_op_for_intent",
                    "success": False,
                    "processing_status": "failed",
                }
            )
        elif meal:
            _upsert_tonight_state(status="meal_decided", meal=meal, source="email_command", reason=None)
            log_summary = _append_meal_log_for_today(
                meal,
                "takeout" if "pizza" in meal.lower() or "takeout" in meal.lower() else "cooked",
            )
            command_result.update(
                {
                    "action_taken": "decide_meal_and_log",
                    "meal_chosen": meal,
                    "meal_log_result": log_summary,
                    "success": True,
                    "processing_status": "ok",
                }
            )
        else:
            command_result.update(
                {
                    "action_taken": "missing_meal_in_early_decision",
                    "success": False,
                    "processing_status": "failed",
                }
            )
    elif intent == "energy_checkin_response":
        el = _coerce_energy_level_from_reply(data, message_body)
        if el:
            command_result.update(
                {
                    "action_taken": "energy_checkin_valid",
                    "energy_level": el,
                    "success": True,
                    "processing_status": "ok",
                }
            )
        else:
            command_result.update(
                {
                    "action_taken": "energy_checkin_invalid",
                    "success": False,
                    "processing_status": "failed",
                }
            )
    else:
        command_result.update(
            {
                "action_taken": "no_op_for_intent",
                "success": False,
                "processing_status": "failed",
            }
        )

    result["command_result"] = command_result
    return result


def fetch_and_process_emails() -> Dict[str, Any]:
    """
    Authenticate with Gmail (OAuth2), route inbox messages, and process each route.

    Requires in project root:
      - gmail_credentials.json (OAuth client secret from Google Cloud Console)
      - token.json created on first successful auth

    Gmail user labels:
      - grocery-receipt (applied by Gmail rules to receipts)
      - receipt-processed (only for grocery_receipt route)
      - command-processed (only for command route; created automatically if missing)

    Fetch scope: two Gmail searches (receipts and commands), each with ``newer_than:2d``,
      merged and de-duplicated by message id. Other mail never enters this pipeline.

    Install deps:
      pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client html2text
    """
    summary: Dict[str, Any] = {
        "emails_found": 0,
        "receipt_query_count": 0,
        "command_query_count": 0,
        "merged_unique_count": 0,
        "emails_processed": 0,
        "emails_skipped": 0,
        "confirmations_sent": 0,
        "energy_checkin_replies": 0,
        "meal_selections_confirmed": 0,
        "meal_log_duplicate_skipped": 0,
        "total_items_added": 0,
        "total_items_updated": 0,
        "route_counts": {"grocery_receipt": 0, "command": 0, "ignore": 0},
    }

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        print(
            "Gmail integration missing packages. Install:\n"
            "  pip install google-auth google-auth-oauthlib "
            "google-auth-httplib2 google-api-python-client html2text"
        )
        return summary

    SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/gmail.send",
    ]

    root = _repo_root()
    cred_path = root / "gmail_credentials.json"
    token_path = root / "token.json"

    if not cred_path.exists():
        print(f"Missing OAuth client file: {cred_path}")
        return summary

    creds: Optional[Any] = None
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception as e:
            print(f"Could not load token.json: {e}")
            creds = None

    if not creds or not creds.valid:
        try:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(cred_path), SCOPES)
                creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json(), encoding="utf-8")
        except Exception as e:
            print(f"Gmail OAuth failed: {e}")
            return summary

    try:
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    except Exception as e:
        print(f"Could not build Gmail service: {e}")
        return summary

    # Resolve user label IDs by name
    try:
        labels_resp = service.users().labels().list(userId="me").execute()
    except Exception as e:
        print(f"Could not list Gmail labels: {e}")
        return summary

    name_to_id: Dict[str, str] = {}
    for lab in labels_resp.get("labels", []):
        if isinstance(lab, dict) and lab.get("name") and lab.get("id"):
            name_to_id[str(lab["name"])] = str(lab["id"])

    if not name_to_id.get("grocery-receipt"):
        print(
            "Note: Gmail label 'grocery-receipt' not found. "
            "Create it and use it for receipt filtering; allowlist-only mail will still be fetched."
        )

    receipt_processed_label_id = name_to_id.get("receipt-processed")
    command_processed_label_id = name_to_id.get("command-processed")

    if not receipt_processed_label_id:
        print("Gmail label 'receipt-processed' not found. Create it in Gmail.")
        return summary
    if not command_processed_label_id:
        # Create once, then use for command-route messages.
        try:
            created = (
                service.users()
                .labels()
                .create(
                    userId="me",
                    body={
                        "name": "command-processed",
                        "labelListVisibility": "labelShow",
                        "messageListVisibility": "show",
                    },
                )
                .execute()
            )
            command_processed_label_id = str(created.get("id") or "")
            if command_processed_label_id:
                print("[init] Created Gmail label 'command-processed'.")
        except Exception as e:
            print(f"Could not create Gmail label 'command-processed': {e}")
            return summary

    q_receipt = _gmail_receipt_fetch_query()
    q_command = _gmail_command_fetch_query()
    try:
        receipt_ids = _list_gmail_message_ids(service, q_receipt)
    except Exception as e:
        print(f"Gmail receipt query failed ({q_receipt!r}): {e}")
        return summary
    try:
        command_ids = _list_gmail_message_ids(service, q_command)
    except Exception as e:
        print(f"Gmail command query failed ({q_command!r}): {e}")
        return summary

    merged_ids = _merge_gmail_message_ids(receipt_ids, command_ids)
    print(f"[gmail] receipt query: {q_receipt}")
    print(f"[gmail] receipt query: {len(receipt_ids)} message id(s)")
    print(f"[gmail] command query: {q_command}")
    print(f"[gmail] command query: {len(command_ids)} message id(s)")
    print(f"[gmail] merged unique message id(s): {len(merged_ids)}")

    summary["receipt_query_count"] = len(receipt_ids)
    summary["command_query_count"] = len(command_ids)
    summary["merged_unique_count"] = len(merged_ids)
    summary["emails_found"] = len(merged_ids)

    for msg_id in merged_ids:
        try:
            full = (
                service.users()
                .messages()
                .get(userId="me", id=msg_id, format="full")
                .execute()
            )
            payload = full.get("payload") or {}
            headers = _extract_message_headers(payload)
            sender_raw = headers.get("from") or ""
            sender = _sender_email(sender_raw)
            subject = headers.get("subject") or ""

            label_ids = full.get("labelIds") or []
            label_names: List[str] = []
            if isinstance(label_ids, list):
                for lid in label_ids:
                    lid_s = str(lid)
                    # Reverse lookup label name from ID.
                    for n, i in name_to_id.items():
                        if i == lid_s:
                            label_names.append(n)
                            break

            route = _route_inbox_message(label_names, sender)
            summary["route_counts"][route] += 1

            body_text = _decode_gmail_message_body(payload)
            if not body_text.strip():
                print(
                    f"[msg] sender={sender or sender_raw!r} subject={subject!r} "
                    f"route={route} action=skip_no_body confirmation_sent=False"
                )
                summary["emails_skipped"] += 1
                continue

            action_taken = "none"
            confirmation_sent = False
            command_output: Optional[Dict[str, Any]] = None

            if route == "ignore":
                print(
                    f"[msg] sender={sender or sender_raw!r} subject={subject!r} "
                    f"route={route} action=route_ignore_no_pipeline confirmation_sent=False"
                )
                summary["emails_skipped"] += 1
                continue

            skip_claude = _should_skip_claude_calls(route, sender, body_text)
            if skip_claude:
                action_taken = "skipped_claude_safety_guard"
                if route == "command" and command_processed_label_id:
                    service.users().messages().modify(
                        userId="me",
                        id=msg_id,
                        body={"addLabelIds": [command_processed_label_id]},
                    ).execute()
                summary["emails_processed"] += 1
                print(
                    f"[msg] sender={sender or sender_raw!r} subject={subject!r} "
                    f"route={route} action={action_taken} confirmation_sent=False"
                )
                continue

            if route == "grocery_receipt":
                classification = classify_grocery_email(body_text)
                parsed_items: List[Dict[str, Any]] = []
                if classification.get("should_parse_items") is True:
                    parsed_items = parse_grocery_items(body_text)

                pantry_summary: Dict[str, Any] = {}
                if classification.get("should_update_inventory") is True:
                    pantry_summary = update_pantry_from_receipt(classification, parsed_items)
                    summary["total_items_added"] += len(pantry_summary.get("added") or [])
                    summary["total_items_updated"] += len(pantry_summary.get("updated") or [])
                    action_taken = (
                        f"receipt_pipeline add={len(pantry_summary.get('added') or [])} "
                        f"upd={len(pantry_summary.get('updated') or [])}"
                    )
                else:
                    action_taken = "receipt_pipeline_no_inventory_update"
            elif route == "command":
                from interrupt_handler import handle_interrupt

                interrupt_result = handle_interrupt(body_text)
                meal_pick = _try_meal_selection_command(
                    service, msg_id, sender, subject, body_text, interrupt_result
                )
                command_output = (
                    meal_pick
                    if meal_pick is not None
                    else _process_command_message(body_text, interrupt_result=interrupt_result)
                )
                cmd = command_output.get("command_result")
                action_taken = (
                    str(cmd.get("action_taken"))
                    if isinstance(cmd, dict) and cmd.get("action_taken")
                    else "command_no_op"
                )
                inv = cmd.get("inventory_result") if isinstance(cmd, dict) else None
                if isinstance(inv, dict):
                    summary["total_items_added"] += int(inv.get("count_added") or 0)
                    summary["total_items_updated"] += int(inv.get("count_updated") or 0)
            else:
                summary["emails_skipped"] += 1
                action_taken = "ignored_unexpected_route"
                print(
                    f"[msg] sender={sender or sender_raw!r} subject={subject!r} "
                    f"route={route} action={action_taken} confirmation_sent=False"
                )
                continue

            add_label_ids: List[str] = []
            if route == "grocery_receipt":
                add_label_ids.append(receipt_processed_label_id)
            elif route == "command" and command_processed_label_id:
                add_label_ids.append(command_processed_label_id)

            if add_label_ids:
                service.users().messages().modify(
                    userId="me",
                    id=msg_id,
                    body={"addLabelIds": add_label_ids},
                ).execute()

            if route == "command" and command_output is not None:
                cmd = command_output.get("command_result")
                if isinstance(cmd, dict):
                    act = str(cmd.get("action_taken") or "")
                    if act == "meal_confirmed":
                        confirmation_sent = True
                        summary["confirmations_sent"] += 1
                        summary["meal_selections_confirmed"] += 1
                        mlr = cmd.get("meal_log_result")
                        if isinstance(mlr, dict) and mlr.get("reason") == "duplicate_today":
                            summary["meal_log_duplicate_skipped"] += 1
                    elif act in (
                        "meal_selection_clarification",
                        "meal_select_no_options_error",
                    ):
                        confirmation_sent = True
                        summary["confirmations_sent"] += 1
                    elif act == "energy_checkin_valid":
                        el = str(cmd.get("energy_level") or "").strip().lower()
                        if el in ("low", "medium", "high"):
                            if _run_energy_checkin_recommendation_flow(
                                service, msg_id, sender, subject, el
                            ):
                                confirmation_sent = True
                                summary["confirmations_sent"] += 1
                                summary["energy_checkin_replies"] += 1
                    elif act == "energy_checkin_invalid":
                        from gmail_sender import energy_clarification_email_body

                        if _gmail_send_reply_safe(
                            service,
                            msg_id,
                            sender,
                            subject,
                            energy_clarification_email_body(),
                        ):
                            confirmation_sent = True
                            summary["confirmations_sent"] += 1
                        print("[energy] invalid reply; clarification email sent")
                    elif cmd.get("success"):
                        txt = build_command_confirmation(command_output)
                        if not (txt or "").strip():
                            txt = "Got it — I processed your command."
                        if _gmail_send_reply_safe(service, msg_id, sender, subject, txt):
                            confirmation_sent = True
                            summary["confirmations_sent"] += 1
                    else:
                        fail_txt = build_command_failure_message(cmd)
                        if _gmail_send_reply_safe(service, msg_id, sender, subject, fail_txt):
                            confirmation_sent = True
                            summary["confirmations_sent"] += 1

            summary["emails_processed"] += 1
            print(
                f"[msg] sender={sender or sender_raw!r} subject={subject!r} "
                f"route={route} action={action_taken} confirmation_sent={confirmation_sent}"
            )
        except Exception as e:
            print(f"[error] {msg_id}: {e}")
            summary["emails_skipped"] += 1

    return summary


if __name__ == "__main__":
    simulated_email = """
Subject: Instacart Receipt - Order Complete
Retailer: Instacart
Order ID: IC-98421-XY

Items:
1x Grade A Eggs 12 ct - $4.99
2 lb Chicken Thighs - $7.50
1 pint Good & Gather Blueberries - $3.99
1 pack Flour Tortillas 10 ct - $2.49
1 bag Shredded Cheese 8 oz - $3.29
1 bottle Body Lotion - $6.49
1 bottle Dish Soap - $3.99

Subtotal: $32.74
Total: $35.02
"""

    print("=== 1) CLASSIFY EMAIL ===")
    classification = classify_grocery_email(simulated_email)
    print(json.dumps(classification, indent=2))

    print("\n=== 2) PARSE GROCERY ITEMS ===")
    if classification.get("should_parse_items") is True:
        parsed_items = parse_grocery_items(simulated_email)
    else:
        parsed_items = []
    print(json.dumps(parsed_items, indent=2))

    print("\n=== 3) UPDATE PANTRY (FIRST PASS) ===")
    if classification.get("should_update_inventory") is True:
        update_summary = update_pantry_from_receipt(classification, parsed_items)
    else:
        update_summary = {
            "added": [],
            "updated": [],
            "ignored": [],
            "review": [],
            "skipped_duplicate": False,
            "processed_order_key": None,
            "reason": "Classification did not allow inventory updates",
        }
    print(json.dumps(update_summary, indent=2))

    print("\n=== 4) UPDATE PANTRY AGAIN (DUPLICATE CHECK) ===")
    if classification.get("should_update_inventory") is True:
        duplicate_summary = update_pantry_from_receipt(classification, parsed_items)
    else:
        duplicate_summary = {
            "added": [],
            "updated": [],
            "ignored": [],
            "review": [],
            "skipped_duplicate": False,
            "processed_order_key": None,
            "reason": "Classification did not allow inventory updates",
        }
    print(json.dumps(duplicate_summary, indent=2))

    print("\n=== 5) CONFIRMATION QUEUE (TOP 3) ===")
    confirmation_queue = get_confirmation_queue(max_items=3)
    print(json.dumps(confirmation_queue, indent=2))

    print("\n=== 6) MANUAL SMS UPDATE ===")
    sms_message = "we're out of eggs"
    pantry_data = _read_json(_repo_root() / "data" / "pantry_inventory.json")
    interpreted = interpret_inventory_update_message(sms_message, pantry_data)
    print("Interpretation:")
    print(json.dumps(interpreted, indent=2))

    manual_apply_result = False
    if (
        interpreted.get("intent") in {"update_item", "confirm_item"}
        and isinstance(interpreted.get("item_key"), str)
        and isinstance(interpreted.get("updates"), dict)
    ):
        manual_apply_result = manually_update_item(
            interpreted["item_key"],
            interpreted["updates"],
        )
    print("Manual apply result:")
    print(json.dumps({"applied": manual_apply_result}, indent=2))

    print("\n=== 7) MANUAL ADD ITEMS (demo) ===")
    manual_add_summary = add_manual_items(
        [
            {"name": "Soy Sauce"},
            {"name": "Olive Oil"},
            {"name": "Frozen Chicken Nuggets"},
            {"name": "Taco Seasoning"},
            {"name": "Alfredo Sauce"},
        ]
    )
    print(json.dumps(manual_add_summary, indent=2))

    print("\n=== 8) FETCH AND PROCESS GMAIL ===")
    gmail_summary = fetch_and_process_emails()
    print(json.dumps(gmail_summary, indent=2))
