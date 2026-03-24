"""
Dinner recommendation helpers.

- ``meal_templates.json`` lists structured *known* meals (rich fields, optional quick_steps / substitution_hints).
- ``infer_meals_from_pantry`` adds pattern-based options so the engine is not limited to that file.
- Ranking mixes both pools; templates receive a small score bonus (see ``TEMPLATE_STRUCTURE_BONUS``).
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Template ingredient keys -> possible pantry `items` keys (first match by confidence wins).
ITEM_ALIASES: Dict[str, List[str]] = {
    "chicken": ["chicken_thighs", "frozen_chicken_nuggets"],
    "chicken_nuggets": ["frozen_chicken_nuggets"],
    "chicken_thighs": ["chicken_thighs"],
    "vegetables": ["spinach", "roma_tomatoes", "cilantro", "blueberries"],
    "rotisserie_chicken": ["chicken_thighs"],
    "cooked_chicken": ["chicken_thighs"],
    "cheese": ["shredded_cheese"],
    "shrimp": ["shrimp"],
    "pasta": ["pasta"],
    "marinara_sauce": ["marinara_sauce", "tomato_sauce"],
    "frozen_vegetables": ["spinach"],
    "frozen_pizza": ["frozen_pizza"],
    "garlic_bread": ["bread"],
    "bacon": ["bacon"],
    "sausage": ["sausage"],
    "fruit": ["blueberries"],
    "apple_slices": ["apples"],
    "grapes": ["grapes"],
    "lettuce": ["spinach"],
    "ranch": ["ranch"],
    "microwave_rice": ["rice"],
    "deli_turkey": ["deli_turkey"],
    "parmesan": ["shredded_cheese"],
    "tomato_sauce": ["marinara_sauce"],
}

# meal_templates.json holds structured *known* meals (rich metadata, optional quick_steps / hints).
# Pantry inference adds flexible candidates so recommendations are not limited to that list alone.
# Small bonus so structured templates stay slightly preferred when scores are close.
TEMPLATE_STRUCTURE_BONUS = 3.0

# Confidence reflects belief in pantry state; availability reflects whether you can cook with it.
CONFIDENCE_AVAILABILITY_THRESHOLD = 0.5


def _estimated_quantity_is_unavailable(eq: Any) -> bool:
    """True when stock is empty / unknown in a way that should not count as 'have it'."""
    if eq is None:
        return True
    if isinstance(eq, str):
        s = eq.strip().lower()
        if s in ("", "0", "0.0", "out", "none"):
            return True
        return False
    if isinstance(eq, (int, float)):
        try:
            return float(eq) == 0.0
        except (TypeError, ValueError):
            return True
    return True


def is_item_available(item: dict) -> bool:
    """
    Whether an item counts as in-stock for meal matching and scoring.

    High confidence with zero quantity is still unavailable. Use ``get_item_confidence``
    when you need the raw confidence value regardless of quantity (e.g. confirmations).
    """
    if not isinstance(item, dict):
        return False
    if _estimated_quantity_is_unavailable(item.get("estimated_quantity")):
        return False
    try:
        conf_f = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    return conf_f > CONFIDENCE_AVAILABILITY_THRESHOLD


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file: {path}") from e


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON object from disk. Relative paths are resolved from this file's directory."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def _pantry_items_dict(pantry: Any) -> Dict[str, Any]:
    if not isinstance(pantry, dict):
        return {}
    items = pantry.get("items")
    return items if isinstance(items, dict) else {}


def get_item_confidence(pantry: Dict[str, Any], item_key: str) -> Optional[float]:
    """
    Highest confidence among exact key and alias pantry rows (does not consider quantity).

    For cooking availability, use ``is_item_available`` / ``get_item_match_key`` instead.
    """
    items = _pantry_items_dict(pantry)
    keys_to_try = [item_key]
    keys_to_try.extend(ITEM_ALIASES.get(item_key, []))
    best: Optional[float] = None
    for k in keys_to_try:
        if k not in items or not isinstance(items[k], dict):
            continue
        try:
            c = float(items[k].get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if best is None or c > best:
            best = c
    return best


def _ingredient_unavailable_in_pantry(pantry: Dict[str, Any], template_key: str) -> bool:
    """
    True if some pantry row exists for this template ingredient (or alias) but none are
    available (e.g. estimated_quantity is zero). Used to surface restock-style missing items.
    """
    items = _pantry_items_dict(pantry)
    keys_to_try = [template_key]
    keys_to_try.extend(ITEM_ALIASES.get(template_key, []))
    any_row = False
    for k in keys_to_try:
        if k not in items or not isinstance(items[k], dict):
            continue
        any_row = True
        if is_item_available(items[k]):
            return False
    return any_row


def _pantry_item_display_name(pantry: Dict[str, Any], pantry_key: str) -> str:
    items = _pantry_items_dict(pantry)
    entry = items.get(pantry_key)
    if isinstance(entry, dict) and entry.get("name"):
        return str(entry["name"]).strip()
    return str(pantry_key).replace("_", " ").strip().title()


def _template_ingredient_display(name: str) -> str:
    return str(name).replace("_", " ").strip().title()


def get_item_match_key(pantry: Dict[str, Any], item_key: str) -> Optional[str]:
    """Return pantry item key with highest confidence among exact + aliases (in-stock items only)."""
    items = _pantry_items_dict(pantry)
    keys_to_try = [item_key]
    keys_to_try.extend(ITEM_ALIASES.get(item_key, []))
    best_key: Optional[str] = None
    best_conf: float = -1.0
    for k in keys_to_try:
        if k not in items or not isinstance(items[k], dict):
            continue
        entry = items[k]
        if not is_item_available(entry):
            continue
        try:
            c = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            c = 0.0
        if c > best_conf:
            best_conf = c
            best_key = k
    return best_key


def _normalize_energy(energy: str) -> str:
    e = (energy or "medium").strip().lower()
    if e in ("low", "medium", "high"):
        return e
    return "medium"


def _avg_pantry_confidence(pantry: Dict[str, Any]) -> float:
    """Average confidence across *available* items only (for weak-pantry heuristics)."""
    items = _pantry_items_dict(pantry)
    confs: List[float] = []
    for v in items.values():
        if isinstance(v, dict) and is_item_available(v) and v.get("confidence") is not None:
            try:
                confs.append(float(v["confidence"]))
            except (TypeError, ValueError):
                continue
    if not confs:
        return 0.5
    return sum(confs) / len(confs)


def _parse_meal_date(s: Any) -> Optional[datetime]:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return datetime.strptime(s.strip()[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _meal_similar_to_template(logged_meal: str, display_name: str, meal_key: str) -> bool:
    lm = logged_meal.lower()
    dn = display_name.lower().strip()
    mk_slug = meal_key.replace("_", " ").lower()
    if dn and (dn in lm or lm in dn):
        return True
    if mk_slug and mk_slug in lm:
        return True
    for tok in re.findall(r"[a-z]{4,}", dn):
        if len(tok) >= 4 and tok in lm:
            return True
    return False


def _days_since_last_similar_meal(
    meal_log: Dict[str, Any], display_name: str, meal_key: str
) -> Optional[int]:
    entries = meal_log.get("meals", [])
    if not isinstance(entries, list):
        return None
    dated: List[tuple[datetime, str]] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        d = _parse_meal_date(e.get("date"))
        meal_name = str(e.get("meal") or "")
        if d is not None:
            dated.append((d, meal_name))
    if not dated:
        return None
    dated.sort(key=lambda x: x[0], reverse=True)
    today = date.today()
    for d, meal_name in dated:
        if _meal_similar_to_template(meal_name, display_name, meal_key):
            delta = today - d.date()
            return max(0, delta.days)
    return None


def _generic_fallback_steps() -> List[str]:
    return [
        "Pull out what you have from the lists above.",
        "Cook the main protein or starch first.",
        "Combine, season, and heat through.",
        "Taste and serve warm.",
    ]


def _first_protein_match_key(pantry: Dict[str, Any]) -> Optional[str]:
    for key in (
        "chicken",
        "chicken_thighs",
        "ground_beef",
        "beef",
        "shrimp",
        "deli_turkey",
        "rotisserie_chicken",
        "cooked_chicken",
        "sausage",
        "pork",
    ):
        mk = get_item_match_key(pantry, key)
        if mk:
            return mk
    return None


def _try_infer_mexican_plate(pantry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t = get_item_match_key(pantry, "tortillas")
    c = get_item_match_key(pantry, "cheese") or get_item_match_key(pantry, "shredded_cheese")
    p = _first_protein_match_key(pantry)
    if not (t and c and p):
        return None
    matched = list(dict.fromkeys([t, c, p]))
    return {
        "meal_key": "inferred_taco_quesadilla_wrap",
        "display_name": "Tacos, quesadillas, or wraps",
        "max_minutes": 25,
        "energy_fit": ["low", "medium"],
        "matched_keys": matched,
        "quick_steps": [
            "Warm protein with seasoning if you like.",
            "Fill tortillas with cheese and protein; fold or roll.",
            "Cook in a pan until hot and melty.",
            "Slice and add salsa or sour cream if you have them.",
        ],
        "substitution_hints": [
            "Swap protein for beans or any cooked leftovers you already have.",
        ],
        "reasons": ["Pantry has tortillas, cheese, and a protein for a Mexican-style plate."],
    }


def _try_infer_pasta_red_sauce(pantry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pasta = get_item_match_key(pantry, "pasta")
    sauce = get_item_match_key(pantry, "marinara_sauce") or get_item_match_key(
        pantry, "tomato_sauce"
    )
    p = _first_protein_match_key(pantry)
    if not (pasta and sauce and p):
        return None
    matched = list(dict.fromkeys([pasta, sauce, p]))
    return {
        "meal_key": "inferred_pasta_with_red_sauce",
        "display_name": "Pasta with red sauce and protein",
        "max_minutes": 25,
        "energy_fit": ["low", "medium", "high"],
        "matched_keys": matched,
        "quick_steps": [
            "Boil pasta to al dente; reserve a splash of water.",
            "Warm sauce; cook or heat protein separately if needed.",
            "Toss pasta with sauce and protein; add cheese if you have it.",
        ],
        "substitution_hints": [
            "Tomato sauce or jarred marinara both work; a pinch of sugar tames harsh sauce.",
        ],
        "reasons": ["Pantry has pasta, tomato sauce, and protein for a simple pasta dinner."],
    }


def _try_infer_fried_rice_style(pantry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rice = get_item_match_key(pantry, "rice") or get_item_match_key(pantry, "microwave_rice")
    eggs = get_item_match_key(pantry, "eggs")
    soy = get_item_match_key(pantry, "soy_sauce")
    if not (rice and eggs and soy):
        return None
    matched = list(dict.fromkeys([rice, eggs, soy]))
    return {
        "meal_key": "inferred_fried_rice_style",
        "display_name": "Fried rice",
        "max_minutes": 20,
        "energy_fit": ["low", "medium"],
        "matched_keys": matched,
        "quick_steps": [
            "Scramble eggs in a hot pan; set aside.",
            "Stir-fry rice with a little oil until hot.",
            "Add soy sauce and fold eggs back in; add protein or veg if you have them.",
        ],
        "substitution_hints": [
            "Day-old rice browns best; cool fresh rice on a plate first if it is sticky.",
        ],
        "reasons": ["Pantry has rice, eggs, and soy sauce for a quick fried-rice style meal."],
    }


def _try_infer_grilled_cheese(pantry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    bread = get_item_match_key(pantry, "bread") or get_item_match_key(pantry, "garlic_bread")
    cheese = get_item_match_key(pantry, "cheese") or get_item_match_key(
        pantry, "shredded_cheese"
    )
    if not (bread and cheese):
        return None
    matched = list(dict.fromkeys([bread, cheese]))
    return {
        "meal_key": "inferred_grilled_cheese",
        "display_name": "Grilled cheese",
        "max_minutes": 15,
        "energy_fit": ["low", "medium"],
        "matched_keys": matched,
        "quick_steps": [
            "Butter bread if you like; fill with cheese.",
            "Cook in a pan until golden and melty.",
            "Serve with soup, fruit, or salad if you have it.",
        ],
        "substitution_hints": [
            "Any melty cheese works; add ham or tomato if on hand.",
        ],
        "reasons": ["Pantry has bread and cheese for a fast grilled cheese."],
    }


def _try_infer_spaghetti_meatballs_style(pantry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pasta = get_item_match_key(pantry, "pasta")
    sauce = get_item_match_key(pantry, "marinara_sauce") or get_item_match_key(
        pantry, "tomato_sauce"
    )
    meat = (
        get_item_match_key(pantry, "frozen_meatballs")
        or get_item_match_key(pantry, "ground_beef")
        or get_item_match_key(pantry, "italian_sausage")
        or get_item_match_key(pantry, "sausage")
    )
    if not (pasta and sauce and meat):
        return None
    matched = list(dict.fromkeys([pasta, sauce, meat]))
    return {
        "meal_key": "inferred_spaghetti_meatballs_style",
        "display_name": "Spaghetti and meatballs",
        "max_minutes": 25,
        "energy_fit": ["low", "medium"],
        "matched_keys": matched,
        "quick_steps": [
            "Boil pasta to al dente.",
            "Heat meatballs or brown ground meat in sauce until cooked through.",
            "Combine, top with cheese if you have it, and serve.",
        ],
        "substitution_hints": [
            "Ground beef or sausage can stand in for frozen meatballs; simmer in sauce until done.",
        ],
        "reasons": ["Pantry has pasta, tomato sauce, and meat for a spaghetti-style bowl."],
    }


_INFERENCE_FUNCS = (
    _try_infer_mexican_plate,
    _try_infer_fried_rice_style,
    _try_infer_grilled_cheese,
    _try_infer_pasta_red_sauce,
    _try_infer_spaghetti_meatballs_style,
)


def score_inferred_candidate(
    raw: Dict[str, Any],
    pantry: Dict[str, Any],
    prefs: Dict[str, Any],
    meal_log: Dict[str, Any],
    energy_level: str,
) -> Dict[str, Any]:
    """Score a pantry-inferred meal so it can rank next to template candidates."""
    energy = _normalize_energy(energy_level)
    meal_key = str(raw["meal_key"])
    display_name = str(raw["display_name"])
    reasons = list(raw.get("reasons", []))

    try:
        tmpl_min = float(raw.get("max_minutes", 25))
    except (TypeError, ValueError):
        tmpl_min = 25.0

    energy_fit = raw.get("energy_fit") if isinstance(raw.get("energy_fit"), list) else []
    energy_ok = energy in [str(x).lower() for x in energy_fit] if energy_fit else True
    energy_fit_score = 12.0 if energy_ok else 4.0
    if energy_ok:
        reasons.append(f"Fits '{energy}' energy level.")

    cc = prefs.get("cooking_constraints") if isinstance(prefs.get("cooking_constraints"), dict) else {}
    try:
        max_week = float(cc.get("max_weeknight_prep_minutes", 35))
    except (TypeError, ValueError):
        max_week = 35.0
    try:
        low_limit = float(cc.get("low_energy_max_minutes", 20))
    except (TypeError, ValueError):
        low_limit = 20.0
    time_limit = low_limit if energy == "low" else max_week
    time_ok = tmpl_min <= time_limit
    time_score = 8.0 if time_ok else -3.0
    if time_ok:
        reasons.append(f"Prep time {int(tmpl_min)}m within limit ({int(time_limit)}m).")

    repeat_tol = 5
    days_since = _days_since_last_similar_meal(meal_log, display_name, meal_key)
    repeat_penalty = 0.0
    if days_since is not None and days_since < repeat_tol:
        repeat_penalty = (repeat_tol - days_since) * 0.15
        reasons.append(f"Similar meal {days_since}d ago (tolerance {repeat_tol}d).")

    matched_keys = raw.get("matched_keys") if isinstance(raw.get("matched_keys"), list) else []
    confs: List[float] = []
    items_inf = _pantry_items_dict(pantry)
    for k in matched_keys:
        if not isinstance(k, str):
            continue
        ent = items_inf.get(k)
        if isinstance(ent, dict) and is_item_available(ent):
            try:
                confs.append(float(ent.get("confidence", 0.0)))
            except (TypeError, ValueError):
                continue
    avg_conf = sum(confs) / len(confs) if confs else 0.55

    n = len(matched_keys) if matched_keys else 1
    coverage = min(18.0, n * 6.0)

    score = (
        coverage
        + avg_conf * 20.0
        + energy_fit_score
        + time_score
        - repeat_penalty
    )

    matched_labels = [
        _pantry_item_display_name(pantry, k) for k in matched_keys if isinstance(k, str)
    ]

    qs = raw.get("quick_steps")
    if not isinstance(qs, list) or not qs:
        qs = _generic_fallback_steps()
    sh = raw.get("substitution_hints") if isinstance(raw.get("substitution_hints"), list) else []

    try:
        time_minutes = int(round(float(tmpl_min)))
    except (TypeError, ValueError):
        time_minutes = 25

    return {
        "meal_key": meal_key,
        "display_name": display_name,
        "category": "Inferred",
        "candidate_type": "inferred",
        "score": round(score, 4),
        "reason": reasons,
        "matched_items": matched_labels,
        "optional_matched_items": [],
        "missing_items": [],
        "time_minutes": time_minutes,
        "takeout": False,
        "quick_steps": [str(s).strip() for s in qs if str(s).strip()],
        "substitution_hints": [str(s).strip() for s in sh if str(s).strip()],
    }


def infer_meals_from_pantry(
    pantry: Dict[str, Any],
    prefs: Dict[str, Any],
    meal_log: Dict[str, Any],
    energy_level: str,
) -> List[Dict[str, Any]]:
    """Build inferred candidates from simple pantry patterns (not from meal_templates.json)."""
    out: List[Dict[str, Any]] = []
    for fn in _INFERENCE_FUNCS:
        raw = fn(pantry)
        if not raw:
            continue
        out.append(score_inferred_candidate(raw, pantry, prefs, meal_log, energy_level))
    return out


def score_template(
    template: Dict[str, Any],
    pantry: Dict[str, Any],
    prefs: Dict[str, Any],
    meal_log: Dict[str, Any],
    energy_level: str,
) -> Dict[str, Any]:
    """Deterministic score for one structured meal template from meal_templates.json. Does not call Claude."""
    energy = _normalize_energy(energy_level)
    meal_key = str(template.get("meal_key") or "unknown")
    display_name = str(template.get("display_name") or meal_key)

    required = template.get("required_items") if isinstance(template.get("required_items"), list) else []
    optional = template.get("optional_items") if isinstance(template.get("optional_items"), list) else []
    fallback = template.get("fallback_items") if isinstance(template.get("fallback_items"), list) else []

    matched: List[str] = []
    missing: List[str] = []
    reasons: List[str] = []

    required_score = 0.0
    n_req = len(required)
    if n_req == 0:
        required_score = 1.0
        reasons.append("No required ingredients (e.g. takeout / freezer pizza).")
    else:
        met = 0.0
        for r in required:
            if not isinstance(r, str):
                continue
            mk = get_item_match_key(pantry, r)
            if mk:
                matched.append(mk)
                met += 1.0
                continue
            sub = False
            for f in fallback:
                if not isinstance(f, str):
                    continue
                mk_fb = get_item_match_key(pantry, f)
                if mk_fb:
                    matched.append(mk_fb)
                    met += 0.65
                    sub = True
                    reasons.append(f"Used fallback for '{r}' via pantry match.")
                    break
            if not sub:
                missing.append(r)
                if _ingredient_unavailable_in_pantry(pantry, r):
                    reasons.append(f"'{r}' is in pantry but out of stock (not available for cooking).")
        required_score = min(1.0, met / n_req)

    # Optional coverage
    opt_hits: List[str] = []
    for o in optional:
        if not isinstance(o, str):
            continue
        ok = get_item_match_key(pantry, o)
        if ok:
            opt_hits.append(ok)
    optional_bonus = min(12.0, len(opt_hits) * 3.0)

    # Average confidence of matched required pantry keys (available items only)
    matched_keys = list(dict.fromkeys(matched))
    req_conf_sum = 0.0
    req_conf_n = 0
    items_for_conf = _pantry_items_dict(pantry)
    for mk in matched_keys:
        if not isinstance(mk, str):
            continue
        ent = items_for_conf.get(mk)
        if isinstance(ent, dict) and is_item_available(ent):
            try:
                req_conf_sum += float(ent.get("confidence", 0.0))
                req_conf_n += 1
            except (TypeError, ValueError):
                continue
    avg_req_conf = req_conf_sum / req_conf_n if req_conf_n else 0.55

    # Energy fit
    energy_fit_list = template.get("energy_fit") if isinstance(template.get("energy_fit"), list) else []
    energy_ok = energy in [str(x).lower() for x in energy_fit_list]
    energy_fit_score = 15.0 if energy_ok else 4.0
    if energy_ok:
        reasons.append(f"Fits '{energy}' energy level.")

    # Cooking time vs preferences
    cc = prefs.get("cooking_constraints") if isinstance(prefs.get("cooking_constraints"), dict) else {}
    try:
        max_week = float(cc.get("max_weeknight_prep_minutes", 35))
    except (TypeError, ValueError):
        max_week = 35.0
    try:
        low_limit = float(cc.get("low_energy_max_minutes", 20))
    except (TypeError, ValueError):
        low_limit = 20.0
    try:
        tmpl_min = float(template.get("max_minutes", 30))
    except (TypeError, ValueError):
        tmpl_min = 30.0
    time_limit = low_limit if energy == "low" else max_week
    time_ok = tmpl_min <= time_limit
    time_score = 10.0 if time_ok else -5.0 - min(8.0, (tmpl_min - time_limit) * 0.25)
    if time_ok:
        reasons.append(f"Prep time {int(tmpl_min)}m within limit ({int(time_limit)}m).")

    # Repeat penalty
    repeat_tol = int(template.get("repeat_tolerance_days") or 5)
    days_since = _days_since_last_similar_meal(meal_log, display_name, meal_key)
    repeat_penalty = 0.0
    if days_since is not None and days_since < repeat_tol:
        repeat_penalty = (repeat_tol - days_since) * 0.18
        reasons.append(f"Similar meal {days_since}d ago (tolerance {repeat_tol}d).")

    # Kid-friendly
    kid_bonus = 5.0 if template.get("kid_friendly") is True else 0.0
    if kid_bonus:
        reasons.append("Kid-friendly template.")

    # Cuisine vs favorites
    cuisine = str(template.get("cuisine") or "")
    favs = (
        (prefs.get("cuisine_rotation") or {}).get("favorites")
        if isinstance(prefs.get("cuisine_rotation"), dict)
        else []
    )
    cuisine_bonus = 0.0
    if isinstance(favs, list) and cuisine and cuisine in favs:
        cuisine_bonus = 4.0
        reasons.append(f"Household favorite cuisine ({cuisine}).")

    weak_pantry = _avg_pantry_confidence(pantry) < 0.55
    takeout_template = meal_key in ("hibachi_takeout", "pizza_night") or (not required and bool(template.get("takeout_alternative")))

    takeout_bonus = 0.0
    if (energy == "low" or weak_pantry) and (takeout_template or template.get("takeout_alternative")):
        takeout_bonus = 8.0
        reasons.append("Takeout-friendly when energy is low or pantry confidence is mixed.")

    # Weighted score (deterministic) + small bonus for structured template metadata.
    score = (
        required_score * 28.0
        + avg_req_conf * 22.0
        + optional_bonus
        + energy_fit_score
        + time_score
        - repeat_penalty
        + kid_bonus
        + cuisine_bonus
        + takeout_bonus
        + TEMPLATE_STRUCTURE_BONUS
    )

    takeout_flag = bool(
        takeout_template or (meal_key == "pizza_night") or (meal_key == "hibachi_takeout")
    )

    matched_labels = [_pantry_item_display_name(pantry, k) for k in matched_keys]
    opt_hit_keys = list(dict.fromkeys(opt_hits))
    optional_matched_labels = [_pantry_item_display_name(pantry, k) for k in opt_hit_keys]
    missing_labels = [_template_ingredient_display(m) for m in missing if isinstance(m, str)]

    try:
        time_minutes = int(round(float(tmpl_min)))
    except (TypeError, ValueError):
        time_minutes = 30

    tqs = template.get("quick_steps")
    if not isinstance(tqs, list):
        tqs = []
    tsh = template.get("substitution_hints")
    if not isinstance(tsh, list):
        tsh = []

    category_label = str(
        template.get("category") or template.get("cuisine") or "Unknown"
    ).strip() or "Unknown"

    return {
        "meal_key": meal_key,
        "display_name": display_name,
        "category": category_label,
        "candidate_type": "template",
        "score": round(score, 4),
        "reason": reasons,
        "matched_items": matched_labels,
        "optional_matched_items": optional_matched_labels,
        "missing_items": missing_labels,
        "time_minutes": time_minutes,
        "takeout": takeout_flag,
        "quick_steps": [str(s).strip() for s in tqs if str(s).strip()],
        "substitution_hints": [str(s).strip() for s in tsh if str(s).strip()],
    }


def rank_dinner_options(energy_level: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Rank dinner options from (1) meal_templates.json and (2) pantry-inferred meals.
    Templates get a small score bonus so structured meals stay slightly ahead when close.
    """
    root = Path(__file__).resolve().parent
    data_dir = root / "data"

    pantry = load_json(str(data_dir / "pantry_inventory.json"))
    prefs = load_json(str(data_dir / "family_preferences.json"))
    meal_log = load_json(str(data_dir / "meal_log.json"))
    templates_doc = load_json(str(data_dir / "meal_templates.json"))

    templates = templates_doc.get("templates", [])
    if not isinstance(templates, list):
        templates = []

    # Structured templates (known meals) + pantry-inferred options, ranked together.
    combined: List[Dict[str, Any]] = []
    for t in templates:
        if not isinstance(t, dict):
            continue
        combined.append(score_template(t, pantry, prefs, meal_log, energy_level))

    combined.extend(
        infer_meals_from_pantry(pantry, prefs, meal_log, energy_level)
    )

    # If the same meal_key appears twice, keep the higher score.
    by_key: Dict[str, Dict[str, Any]] = {}
    for opt in combined:
        if not isinstance(opt, dict):
            continue
        mk = str(opt.get("meal_key") or "")
        if not mk:
            continue
        prev = by_key.get(mk)
        if prev is None or float(opt.get("score", 0.0)) > float(prev.get("score", 0.0)):
            by_key[mk] = opt

    merged = list(by_key.values())
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[: max(0, max_results)]


def _cap_item_strings(items: Any, max_n: int = 3) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for x in items:
        if x is None:
            continue
        s = str(x).strip()
        if s:
            out.append(s)
        if len(out) >= max_n:
            break
    return out


def _all_item_strings(items: Any) -> List[str]:
    """Full ingredient list for detailed messages (no cap)."""
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for x in items:
        if x is None:
            continue
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def build_mini_steps(selected_option: Dict[str, Any]) -> str:
    """
    One-line cooking summary for a ranked option (top-3 card).
    Uses quick_steps from the ranked option (template JSON or inferred), not hardcoded meal lists.
    """
    if not isinstance(selected_option, dict):
        return ""
    mk = str(selected_option.get("meal_key") or "")
    takeout = bool(selected_option.get("takeout"))

    if mk == "hibachi_takeout":
        return (
            "Order your usual combo for pickup or delivery, set the table while you wait, "
            "and eat hot when it arrives."
        )
    if mk == "pizza_night":
        return (
            "Heat the oven as the box says, bake until the crust is crisp and cheese bubbles, "
            "then slice and serve."
        )

    steps = selected_option.get("quick_steps")
    if isinstance(steps, list) and steps:
        parts = [str(s).strip().rstrip(".") for s in steps[:3] if str(s).strip()]
        if parts:
            segs = [parts[0]]
            for p in parts[1:]:
                segs.append(p[0].lower() + p[1:] if len(p) > 1 else p)
            combined = "; ".join(segs)
            combined = combined[0].upper() + combined[1:] + "."
            if len(combined) > 280:
                combined = combined[:277].rstrip() + "..."
            return combined

    if takeout:
        return "Keep it simple: minimal prep, heat or assemble, and serve."

    return (
        "Cook your main protein or starch first, combine everything in one pan or bowl, "
        "season to taste, and serve warm."
    )


def _use_this_when_line(opt: Dict[str, Any], energy_level: str) -> str:
    """Short situational hint for the top-3 recommendation card."""
    if not isinstance(opt, dict):
        return "Use this when: you need a practical dinner."
    e = _normalize_energy(energy_level)
    reasons = opt.get("reason") if isinstance(opt.get("reason"), list) else []
    rs = " ".join(str(r) for r in reasons)
    try:
        mins = int(opt.get("time_minutes", 20))
    except (TypeError, ValueError):
        mins = 20
    takeout = bool(opt.get("takeout"))

    clauses: List[str] = []
    if takeout:
        clauses.append("you want almost no cooking")
    if e == "low":
        clauses.append("you're wiped and need something straightforward")
    elif e == "high":
        clauses.append("you have a little more bandwidth tonight")
    else:
        clauses.append("you want a normal weeknight pace")

    if "Kid-friendly" in rs or "kid-friendly" in rs.lower():
        clauses.append("kids need something easy to eat")

    if (not takeout) and "Takeout-friendly when energy is low" in rs:
        clauses.append("you might pivot to takeout if cooking still feels like too much")

    if "Similar meal" in rs:
        clauses.append("you're OK repeating something familiar soon")

    # De-duplicate while keeping order
    seen: set[str] = set()
    uniq = []
    for c in clauses:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    context = ", ".join(uniq[:4]) if uniq else "you need a practical dinner"
    return f"Use this when: {context} (about {mins} min)."


def _substitution_lines_for_selected(selected_option: Dict[str, Any]) -> List[str]:
    """Practical subs: template/inferred hints (if present), scorer notes, missing-item nudges."""
    if not isinstance(selected_option, dict):
        return []
    lines: List[str] = []
    reasons = selected_option.get("reason") if isinstance(selected_option.get("reason"), list) else []
    for r in reasons:
        if not isinstance(r, str):
            continue
        rl = r.lower()
        if "fallback" in rl or "used fallback" in rl:
            lines.append(r.strip())

    hints = selected_option.get("substitution_hints")
    if isinstance(hints, list):
        for h in hints:
            if isinstance(h, str) and h.strip() and h.strip() not in lines:
                lines.append(h.strip())

    miss = selected_option.get("missing_items") if isinstance(selected_option.get("missing_items"), list) else []
    for m in miss:
        s = str(m).strip()
        if not s:
            continue
        low = s.lower()
        extra = ""
        if "rice" in low:
            extra = f"If you're short on {s}: microwave rice or cook a small batch, then cool slightly before stir-frying."
        elif "egg" in low:
            extra = f"If you're short on {s}: add extra veg or tofu, or skip and bump soy/sauce flavor."
        elif "tortilla" in low:
            extra = f"If you're short on {s}: use bread for a melt, or lettuce cups for a low-carb wrap."
        elif "cheese" in low:
            extra = f"If you're short on {s}: any melty cheese you have works."
        elif "sauce" in low or "marinara" in low:
            extra = f"If you're short on {s}: tomato sauce + pinch of sugar + dried herbs is a quick stand-in."
        if extra and extra not in lines:
            lines.append(extra)

    return lines


def build_recommendation_message(options: List[Dict[str, Any]], energy_level: str) -> str:
    """Top options with mini-instructions; no numeric scores."""
    energy = _normalize_energy(energy_level)
    lines: List[str] = [
        f"Tonight ({energy} energy) - top picks:",
        "",
    ]
    for opt in options:
        if not isinstance(opt, dict):
            continue
        name = str(opt.get("display_name") or opt.get("meal_key") or "Meal")
        try:
            mins = int(opt.get("time_minutes", 0))
        except (TypeError, ValueError):
            mins = 20
        matched = _cap_item_strings(opt.get("matched_items"), 5)
        optional_m = _cap_item_strings(opt.get("optional_matched_items"), 5)
        missing = _cap_item_strings(opt.get("missing_items"), 5)

        lines.append(f"**{name}** - {mins} min")
        if matched:
            lines.append(f"You already have: {', '.join(matched)}")
        if optional_m:
            lines.append(f"Nice to have: {', '.join(optional_m)}")
        if missing:
            lines.append(f"Missing: {', '.join(missing)}")
        lines.append(_use_this_when_line(opt, energy_level))
        lines.append(f"Quick steps: {build_mini_steps(opt)}")
        lines.append("")

    return "\n".join(lines).strip()


def build_selected_meal_message(selected_option: Dict[str, Any]) -> str:
    """
    Detailed follow-up after the user picks one ranked option:
    grouped ingredients, steps, and substitutions when relevant.
    """
    if not isinstance(selected_option, dict):
        return ""

    name = str(selected_option.get("display_name") or selected_option.get("meal_key") or "Dinner")
    mk = str(selected_option.get("meal_key") or "")
    try:
        mins = int(selected_option.get("time_minutes", 20))
    except (TypeError, ValueError):
        mins = 20

    have = _all_item_strings(selected_option.get("matched_items"))
    extra = _all_item_strings(selected_option.get("optional_matched_items"))
    miss = _all_item_strings(selected_option.get("missing_items"))

    lines: List[str] = [
        f"{name} (~{mins} min)",
        "",
        "Ingredients",
    ]
    if have:
        lines.append(f"You have: {', '.join(have)}")
    if extra:
        lines.append(f"Nice to have: {', '.join(extra)}")
    if miss:
        lines.append(f"Missing: {', '.join(miss)}")
    if not have and not extra and not miss:
        lines.append("(No ingredient matches listed - use the steps below with what you have.)")
    lines.append("")

    steps = selected_option.get("quick_steps")
    if isinstance(steps, list) and steps:
        steps = [str(s).strip() for s in steps if str(s).strip()]
    else:
        steps = _generic_fallback_steps()
    lines.append("Steps:")
    for i, step in enumerate(steps[:5], start=1):
        lines.append(f"{i}. {step}")

    sub_lines = _substitution_lines_for_selected(selected_option)
    if sub_lines:
        lines.append("")
        lines.append("Substitutions (if relevant):")
        for sl in sub_lines[:8]:
            lines.append(f"- {sl}")

    return "\n".join(lines).strip()


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

    # Extract pantry items with sufficient confidence (> 0.5) and in-stock quantity.
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
        if confidence_f > CONFIDENCE_AVAILABILITY_THRESHOLD and is_item_available(item):
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
        else "- (none available with confidence > 0.5)"
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
            "PANTRY (available for cooking, confidence > 0.5):",
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


def _debug_pantry_availability_examples() -> None:
    """Sanity-check is_item_available vs high confidence + zero quantity (run: python -m agent)."""
    pork = {
        "name": "pork chops",
        "confidence": 0.99,
        "estimated_quantity": 0,
    }
    peppers = {
        "name": "frozen mixed peppers",
        "confidence": 0.95,
        "estimated_quantity": "1",
    }
    print("=== Pantry availability (debug) ===")
    print(f"  pork_chops (qty 0, conf 0.99): is_item_available = {is_item_available(pork)}")
    print(
        f"  frozen_mixed_peppers (qty '1', conf 0.95): is_item_available = {is_item_available(peppers)}"
    )


if __name__ == "__main__":
    _debug_pantry_availability_examples()
    print()
    energy_level = "low"
    top_options = rank_dinner_options(energy_level, max_results=3)
    # Debug: full ranked payloads (includes numeric score for sorting only).
    print(json.dumps(top_options, indent=2))
    print()
    # User-facing: top 3 with "Use this when" + one-line quick steps per option.
    print(build_recommendation_message(top_options, energy_level))
    if top_options:
        print()
        # Detailed follow-up as if the user chose the first suggestion.
        print(build_selected_meal_message(top_options[0]))

