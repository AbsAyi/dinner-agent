from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables immediately so `send_sms()` can rely on `os.getenv()`.
_REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=str(_REPO_ROOT / ".env"))


def send_sms(message: str) -> bool:
    # Load Twilio credentials from environment variables (loaded from `.env` at module import).
    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")
    my_phone_number = os.getenv("MY_PHONE_NUMBER")

    twilio_account_sid = twilio_account_sid.strip() if isinstance(twilio_account_sid, str) else None
    twilio_auth_token = twilio_auth_token.strip() if isinstance(twilio_auth_token, str) else None
    twilio_phone_number = twilio_phone_number.strip() if isinstance(twilio_phone_number, str) else None
    my_phone_number = my_phone_number.strip() if isinstance(my_phone_number, str) else None

    if not all([twilio_account_sid, twilio_auth_token, twilio_phone_number, my_phone_number]):
        print(
            "SMS send failed: missing required environment variables "
            "(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, MY_PHONE_NUMBER)."
        )
        return False

    try:
        from twilio.rest import Client
    except ImportError:
        print("SMS send failed: twilio library not installed. Run: pip install twilio")
        return False

    try:
        client = Client(twilio_account_sid, twilio_auth_token)
        print(
            f"Attempting to send from {twilio_phone_number} to {my_phone_number}"
        )
        client.messages.create(
            body=message,
            from_=twilio_phone_number,
            to=my_phone_number,
        )
        return True
    except Exception as e:  # noqa: BLE001 - need to catch Twilio/network errors gracefully
        # Twilio errors often expose additional fields like `code` and `status`.
        code = getattr(e, "code", None)
        status = getattr(e, "status", None)
        msg = getattr(e, "msg", None)
        message = getattr(e, "message", None)
        print(
            "SMS send failed: "
            f"type={type(e).__name__}; "
            f"code={code}; status={status}; msg={msg}; message={message}; "
            f"exception={repr(e)}"
        )
        return False


if __name__ == "__main__":
    ok = send_sms("🍳 Dinner agent online. SMS working.")
    print(f"SMS test message sent: {ok}")

