"""Delivery channels for proactive digests: in-app (always), email (SMTP) and Telegram (bot API).

Both external channels are free: Gmail/Outlook SMTP with an app password, and a Telegram bot
created via @BotFather. Each channel reports its own status so a failure never blocks the others.
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any

import httpx

from finpilot.config import Settings, get_settings

log = logging.getLogger(__name__)


def send_email(
    to: str, subject: str, text: str, html_body: str, settings: Settings | None = None
) -> dict[str, Any]:
    s = settings or get_settings()
    if not (s.smtp_host and s.smtp_user and s.smtp_password):
        return {"status": "skipped", "detail": "SMTP not configured"}
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = s.smtp_from or s.smtp_user
    msg["To"] = to
    msg.set_content(text)
    msg.add_alternative(html_body, subtype="html")
    try:
        if s.smtp_port == 465:
            with smtplib.SMTP_SSL(
                s.smtp_host, s.smtp_port, context=ssl.create_default_context(), timeout=20
            ) as srv:
                srv.login(s.smtp_user, s.smtp_password)
                srv.send_message(msg)
        else:
            with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=20) as srv:
                srv.starttls(context=ssl.create_default_context())
                srv.login(s.smtp_user, s.smtp_password)
                srv.send_message(msg)
        return {"status": "sent", "detail": f"to {to}"}
    except Exception as exc:  # noqa: BLE001
        log.warning("email delivery failed: %s", exc)
        return {"status": "failed", "detail": str(exc)[:200]}


def send_telegram(chat_id: str, text: str, settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    if not s.telegram_bot_token:
        return {"status": "skipped", "detail": "Telegram bot token not configured"}
    url = f"https://api.telegram.org/bot{s.telegram_bot_token}/sendMessage"
    # Telegram's legacy Markdown uses single * for bold
    payload = {
        "chat_id": chat_id,
        "text": text.replace("**", "*")[:4000],
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        r = httpx.post(url, json=payload, timeout=15)
        if r.status_code == 400:  # markdown parse error → retry as plain text
            payload.pop("parse_mode")
            r = httpx.post(url, json=payload, timeout=15)
        r.raise_for_status()
        return {"status": "sent", "detail": f"chat {chat_id}"}
    except Exception as exc:  # noqa: BLE001
        log.warning("telegram delivery failed: %s", exc)
        return {"status": "failed", "detail": str(exc)[:200]}


def deliver(
    digest: dict[str, Any],
    *,
    email_to: str | None,
    telegram_chat_id: str | None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    out: dict[str, Any] = {"in_app": {"status": "sent", "detail": "stored in FinPilot inbox"}}
    to = email_to or s.digest_email_to
    out["email"] = (
        send_email(to, digest["subject"], digest["body_markdown"], digest["body_html"], s)
        if to
        else {"status": "skipped", "detail": "no recipient configured"}
    )
    chat = telegram_chat_id or s.telegram_chat_id
    out["telegram"] = (
        send_telegram(chat, f"*{digest['subject']}*\n\n{digest['body_markdown']}", s)
        if chat
        else {"status": "skipped", "detail": "no chat id configured"}
    )
    return out
