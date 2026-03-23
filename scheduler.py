from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state file: {state_path}")
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state_path: Path, state: Dict[str, Any]) -> None:
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def run_daily_checkin() -> None:
    repo_root = Path(__file__).resolve().parent
    state_path = repo_root / "data" / "agent_state.json"

    # 1. Reads data/agent_state.json
    state = _load_state(state_path)

    tonight = state.get("tonight")
    if not isinstance(tonight, dict):
        tonight = {}
        state["tonight"] = tonight

    # 2. Checks if tonight.skip_checkin is True — if so, print and return early
    if bool(tonight.get("skip_checkin")) is True:
        print("Checkin skipped for tonight")
        return

    # 3. Checks if agent_paused is True — if so, print and return early
    if bool(state.get("agent_paused")) is True:
        print("Agent is paused")
        return

    # 4. If neither, import and call these in order:
    #    - load_data() from agent.py
    #    - get_dinner_recommendation(data) from agent.py
    #    - send_sms(recommendation) from sms.py
    from agent import get_dinner_recommendation, load_data
    from sms import send_sms

    data = load_data()
    recommendation = get_dinner_recommendation(data)
    sms_ok = send_sms(recommendation)

    # 5. After sending, update agent_state.json
    now_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    tonight["status"] = "recommendation_sent"
    tonight["decided_at"] = now_ts
    _save_state(state_path, state)

    # 6. Print summary of what happened
    print("Daily check-in summary:")
    print(f"- SMS status: {'success' if sms_ok else 'failed'}")
    print(f"- tonight.status: {tonight.get('status')}")
    print(f"- tonight.decided_at: {tonight.get('decided_at')}")


def schedule_daily(hour: int = 16, minute: int = 30) -> None:
    try:
        import schedule  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "The 'schedule' library is required. Install it with: pip install schedule"
        ) from e

    time_str = f"{hour:02d}:{minute:02d}"
    schedule.every().day.at(time_str).do(run_daily_checkin)

    print(f"Dinner agent scheduler started. Running daily at {time_str}")

    # Continuous loop checking every 60 seconds
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    schedule_daily() # comment out to run immediately

    # run_daily_checkin()  # uncomment to test immediately

