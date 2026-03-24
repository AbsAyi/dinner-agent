"""
Outbound Gmail for the dinner agent (scheduler check-ins, etc.).

Uses the same OAuth files as ``inventory.fetch_and_process_emails``:
``gmail_credentials.json`` and ``token.json`` in the repo root.

Set ``DINNER_AGENT_NOTIFY_EMAIL`` to the destination address for daily recommendations.
"""

from __future__ import annotations

import base64
import os
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


# Subjects used by the scheduler and energy follow-up (thread replies may show as Re: …).
ENERGY_CHECKIN_EMAIL_SUBJECT = "DINNER_AGENT: Dinner energy check"
RECOMMENDATION_EMAIL_SUBJECT = "DINNER_AGENT: Dinner tonight"


def energy_checkin_email_body() -> str:
    """Plain-text body for the scheduled energy prompt (reply with low / medium / high)."""
    return (
        "How's dinner energy tonight?\n\n"
        "Reply with one word: low, medium, or high."
    )


def recommendation_followup_email_subject() -> str:
    """Subject line when sending ranked dinner ideas after energy is known (non-thread sends)."""
    return RECOMMENDATION_EMAIL_SUBJECT


def energy_clarification_email_body() -> str:
    """Short reply when the user's energy level could not be parsed."""
    return (
        "I didn't catch your energy level.\n\n"
        "Please reply with just: low, medium, or high."
    )


def _load_notify_email() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=str(_repo_root() / ".env"), override=False)
    except ImportError:
        pass
    return (os.environ.get("DINNER_AGENT_NOTIFY_EMAIL") or "").strip()


def send_agent_email(subject: str, body: str) -> bool:
    """
    Send a plain-text email from the authenticated Gmail agent inbox.

    Recipient must be set via environment variable ``DINNER_AGENT_NOTIFY_EMAIL``.

    Returns True if the message was accepted by Gmail, False otherwise.
    """
    to_addr = _load_notify_email()
    if not to_addr:
        print(
            "Gmail send skipped: set DINNER_AGENT_NOTIFY_EMAIL in the environment "
            ".env (or your shell) to the destination address."
        )
        return False

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        print(
            "Gmail send failed: missing packages. Install:\n"
            "  pip install google-auth google-auth-oauthlib "
            "google-auth-httplib2 google-api-python-client"
        )
        return False

    scopes = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/gmail.send",
    ]

    root = _repo_root()
    cred_path = root / "gmail_credentials.json"
    token_path = root / "token.json"

    if not cred_path.exists():
        print(f"Gmail send failed: missing OAuth client file: {cred_path}")
        return False

    creds: Optional[Any] = None
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), scopes)
        except Exception as e:
            print(f"Gmail send failed: could not load token.json: {e}")
            creds = None

    if not creds or not creds.valid:
        try:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(cred_path), scopes)
                creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json(), encoding="utf-8")
        except Exception as e:
            print(f"Gmail send failed: OAuth error: {e}")
            return False

    try:
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    except Exception as e:
        print(f"Gmail send failed: could not build service: {e}")
        return False

    mime = MIMEText(body, "plain", "utf-8")
    mime["to"] = to_addr
    mime["subject"] = (subject or "").strip() or "DINNER_AGENT: Dinner tonight"

    try:
        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("utf-8")
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
    except Exception as e:
        print(f"Gmail send failed: {e}")
        return False

    return True
