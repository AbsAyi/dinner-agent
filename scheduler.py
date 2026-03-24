from __future__ import annotations

"""
Long-running orchestrator for the dinner agent email loop.

This module is the single entry point to run continuously: it schedules the daily
Gmail energy check-in (4:30 PM local time by default) and repeatedly polls the inbox
via ``inventory.fetch_and_process_emails``. That function handles grocery receipts,
allowlisted command mail (energy replies, meal picks, pantry commands), and applies
labels so work stays idempotent—no need to run ``python inventory.py`` manually.

``inventory.py`` owns all Gmail read/modify/send logic and data side effects; this file
only wires the schedule and prints high-level logs.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# How often to poll Gmail (minutes). Keep between 1 and 5 for responsive replies.
INBOX_POLL_INTERVAL_MINUTES = 3

# How often the main loop wakes to evaluate scheduled jobs (seconds).
SCHEDULER_TICK_SECONDS = 30


def _load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state file: {state_path}")
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state_path: Path, state: Dict[str, Any]) -> None:
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def run_daily_checkin() -> None:
    """Scheduled once per day: send the energy check-in email (skip/pause logic unchanged)."""
    print(f"[scheduler] daily energy check-in triggered at {datetime.now(timezone.utc).isoformat()}")
    repo_root = Path(__file__).resolve().parent
    state_path = repo_root / "data" / "agent_state.json"

    state = _load_state(state_path)

    tonight = state.get("tonight")
    if not isinstance(tonight, dict):
        tonight = {}
        state["tonight"] = tonight

    if bool(tonight.get("skip_checkin")) is True:
        print("[scheduler] check-in skipped for tonight (skip_checkin)")
        return

    if bool(state.get("agent_paused")) is True:
        print("[scheduler] check-in skipped (agent_paused)")
        return

    from gmail_sender import (
        ENERGY_CHECKIN_EMAIL_SUBJECT,
        energy_checkin_email_body,
        send_agent_email,
    )

    body = energy_checkin_email_body()
    email_ok = send_agent_email(ENERGY_CHECKIN_EMAIL_SUBJECT, body)

    now_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    tonight["status"] = "awaiting_energy"
    tonight["energy_level"] = None
    tonight["checkin_sent_at"] = now_ts
    tonight["checkin_channel"] = "gmail"
    tonight.pop("recommendation_sent_at", None)
    tonight.pop("energy_received_at", None)
    _save_state(state_path, state)

    if email_ok:
        print("[scheduler] energy check-in email sent")
    else:
        print("[scheduler] energy check-in email failed to send")

    print("[scheduler] daily check-in summary:")
    print(f"  - Email status: {'success' if email_ok else 'failed'}")
    print(f"  - tonight.status: {tonight.get('status')}")
    print(f"  - tonight.checkin_sent_at: {tonight.get('checkin_sent_at')}")
    print("  - Reply with low, medium, or high; inbox polling will pick it up.")


def run_inbox_poll() -> None:
    """
    Poll Gmail and process matching messages (receipts, commands, replies).

    Idempotency relies on labels and queries in ``inventory.fetch_and_process_emails``;
    safe to call every few minutes.
    """
    print(f"[scheduler] inbox poll start ({datetime.now(timezone.utc).isoformat()})")
    try:
        from inventory import fetch_and_process_emails

        summary = fetch_and_process_emails()
    except Exception as e:
        print(f"[scheduler] inbox poll error: {e}")
        return

    found = int(summary.get("emails_found") or 0)
    proc = int(summary.get("emails_processed") or 0)
    skipped = int(summary.get("emails_skipped") or 0)
    rc = summary.get("route_counts") or {}
    print(
        f"[scheduler] inbox poll done: "
        f"merged_ids={found} processed={proc} skipped={skipped} "
        f"routes={rc}"
    )
    conf = int(summary.get("confirmations_sent") or 0)
    if conf:
        print(
            f"[scheduler]   confirmations_sent={conf} "
            f"energy_replies={int(summary.get('energy_checkin_replies') or 0)} "
            f"meal_selections={int(summary.get('meal_selections_confirmed') or 0)}"
        )


def run_scheduler_loop(
    *,
    checkin_hour: int = 16,
    checkin_minute: int = 30,
    inbox_poll_minutes: int = INBOX_POLL_INTERVAL_MINUTES,
) -> None:
    """
    Long-running loop: daily check-in at ``checkin_hour:checkin_minute`` and
    inbox polling every ``inbox_poll_minutes`` (clamped to 1–5).
    """
    try:
        import schedule  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "The 'schedule' library is required. Install it with: pip install schedule"
        ) from e

    poll_min = max(1, min(5, int(inbox_poll_minutes)))
    time_str = f"{checkin_hour:02d}:{checkin_minute:02d}"

    schedule.every().day.at(time_str).do(run_daily_checkin)
    schedule.every(poll_min).minutes.do(run_inbox_poll)

    print("[scheduler] --- startup ---")
    print("[scheduler] Dinner agent orchestrator started")
    print(f"[scheduler] Daily check-in time: {time_str}")
    print(f"[scheduler] Inbox polling interval: {poll_min} minute(s)")
    print(f"[scheduler] Scheduler tick: every {SCHEDULER_TICK_SECONDS}s (pending jobs)")
    print("[scheduler] Running initial inbox poll immediately...")
    run_inbox_poll()

    while True:
        schedule.run_pending()
        time.sleep(SCHEDULER_TICK_SECONDS)


if __name__ == "__main__":
    # Default: long-running orchestrator (check-in + inbox polling).
    run_scheduler_loop()

    # Manual tests (uncomment one):
    # run_daily_checkin()
    # run_inbox_poll()
