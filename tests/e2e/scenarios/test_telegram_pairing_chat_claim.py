"""Regression coverage for #3317 — Telegram pairing chat-claim flow.

The user-visible bug: the Telegram bot's pairing reply said
"Enter this code in IronClaw to pair your Telegram account: <code>" without
naming a specific surface, so users naturally pasted the code into their
TUI/CLI chat. The agent rejected it ("wrong place; send it in Telegram"),
leaving them stuck.

This scenario asserts both halves of the fix:

1. The pairing reply now lists every IronClaw surface explicitly
   (Settings → Channels, agent chat, terminal CLI), so the user knows
   exactly where to type the code.

2. Typing ``approve telegram <code>`` in any chat surface — including
   the gateway's `/api/chat/send` — actually completes the pairing,
   matching the bot reply's instructions.

Without this coverage, the surface-explicit reply could quietly regress
to a generic "Enter this code in IronClaw" wording, or the chat-claim
parser could be unhooked from the bridge handler, and #3317 would
silently come back.
"""

import asyncio
import time

import httpx

from helpers import api_post, auth_headers

from .test_telegram_e2e import (
    PAIRED_USER_ID,
    WEBHOOK_SECRET,
    _next_test_update_id,
    activate_telegram,
    extract_pairing_code,
    post_telegram_webhook,
    reset_fake_tg,
    wait_for_sent_messages,
)


async def test_telegram_pairing_reply_names_every_surface(
    isolated_telegram_e2e_server,
):
    """The bot's pairing reply must name web Settings, agent chat, and CLI.

    A regression that drops any of those three surfaces would re-create
    the ambiguity that #3317 surfaced (user pastes code into TUI, no
    handler matches, agent improvises an unhelpful reply).
    """
    base_url = isolated_telegram_e2e_server["base_url"]
    http_url = isolated_telegram_e2e_server["http_url"]
    fake_tg_url = isolated_telegram_e2e_server["fake_tg_url"]
    channels_dir = isolated_telegram_e2e_server["channels_dir"]

    await activate_telegram(base_url, http_url, fake_tg_url, channels_dir)
    await reset_fake_tg(fake_tg_url)

    # Trigger the pairing reply by DM'ing the bot from an unknown user.
    pairing_resp = await post_telegram_webhook(
        http_url,
        {
            "update_id": _next_test_update_id(),
            "message": {
                "message_id": 5001,
                "from": {
                    "id": PAIRED_USER_ID,
                    "is_bot": False,
                    "first_name": "Pairing Tester",
                },
                "chat": {"id": PAIRED_USER_ID, "type": "private"},
                "date": int(time.time()),
                "text": "hello",
            },
        },
        secret=WEBHOOK_SECRET,
    )
    assert pairing_resp.status_code == 200

    messages = await wait_for_sent_messages(fake_tg_url, min_count=1, timeout=60)
    pairing_text = next(
        (m["text"] for m in reversed(messages) if "pair" in m.get("text", "").lower()),
        None,
    )
    assert pairing_text, f"No pairing-reply text found in: {messages}"

    # Every surface must be named so users know where the code is valid.
    assert "Settings" in pairing_text and "Channels" in pairing_text, (
        f"pairing reply must mention Settings → Channels: {pairing_text}"
    )
    assert "approve telegram" in pairing_text, (
        f"pairing reply must mention chat-surface command 'approve telegram': "
        f"{pairing_text}"
    )
    assert "ironclaw pairing approve telegram" in pairing_text, (
        f"pairing reply must mention CLI fallback: {pairing_text}"
    )

    code = extract_pairing_code(messages)
    assert code, f"Expected pairing code in reply, got: {pairing_text}"
    assert code.isalnum() and code.isupper(), (
        f"pairing code must be alphanumeric uppercase, got: {code!r}"
    )


async def test_chat_surface_approves_pairing_code(
    isolated_telegram_e2e_server,
):
    """Typing `approve telegram CODE` in chat completes pairing end-to-end.

    The chat surface here is the web gateway's `/api/chat/send`, but the
    same parser runs for TUI/CLI/Telegram-itself. We then verify the
    paired user can actually exchange messages — proving the pairing
    propagated to the running WASM channel via
    `complete_pairing_approval`, not just the DB row.
    """
    base_url = isolated_telegram_e2e_server["base_url"]
    http_url = isolated_telegram_e2e_server["http_url"]
    fake_tg_url = isolated_telegram_e2e_server["fake_tg_url"]
    channels_dir = isolated_telegram_e2e_server["channels_dir"]

    await activate_telegram(base_url, http_url, fake_tg_url, channels_dir)
    await reset_fake_tg(fake_tg_url)

    # Step 1 — DM the bot from an unknown user to mint a pairing code.
    pairing_resp = await post_telegram_webhook(
        http_url,
        {
            "update_id": _next_test_update_id(),
            "message": {
                "message_id": 6001,
                "from": {
                    "id": PAIRED_USER_ID,
                    "is_bot": False,
                    "first_name": "Chat-Claim Tester",
                },
                "chat": {"id": PAIRED_USER_ID, "type": "private"},
                "date": int(time.time()),
                "text": "hello",
            },
        },
        secret=WEBHOOK_SECRET,
    )
    assert pairing_resp.status_code == 200

    pairing_messages = await wait_for_sent_messages(
        fake_tg_url, min_count=1, timeout=60
    )
    code = extract_pairing_code(pairing_messages)
    assert code, f"Expected pairing code, got messages: {pairing_messages}"
    await reset_fake_tg(fake_tg_url)

    # Step 2 — Submit the pairing claim through the chat surface that
    # users naturally try first. This is the exact path #3317 said was
    # rejected before the fix.
    thread_r = await api_post(base_url, "/api/chat/thread/new", timeout=15)
    thread_r.raise_for_status()
    thread_id = thread_r.json()["id"]

    send_r = await api_post(
        base_url,
        "/api/chat/send",
        json={
            "content": f"approve telegram {code}",
            "thread_id": thread_id,
        },
        timeout=30,
    )
    assert send_r.status_code in (200, 202), (
        f"chat send for pairing claim failed ({send_r.status_code}): {send_r.text}"
    )

    # Wait for the chat handler to produce the pairing-claim response.
    deadline = time.monotonic() + 30
    pairing_response = None
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            history = await client.get(
                f"{base_url}/api/chat/history",
                params={"thread_id": thread_id},
                headers=auth_headers(),
                timeout=15,
            )
            history.raise_for_status()
            data = history.json()
            for turn in data.get("turns", []):
                response = turn.get("response", "")
                if response and (
                    "Pairing approved" in response
                    or "Pairing was approved" in response
                    or "Invalid or expired pairing code" in response
                ):
                    pairing_response = response
                    break
            if pairing_response:
                break
            await asyncio.sleep(0.5)

    assert pairing_response, (
        f"Pairing claim through chat did not produce a recognizable response "
        f"within 30s for thread {thread_id}"
    )
    assert "Pairing approved" in pairing_response, (
        f"Expected successful pairing, got: {pairing_response}"
    )
    assert "telegram" in pairing_response, (
        f"Pairing response must name the channel: {pairing_response}"
    )

    # Step 3 — Prove the pairing actually propagated: the previously-
    # unknown PAIRED_USER_ID should now exchange messages without
    # triggering another pairing reply.
    await reset_fake_tg(fake_tg_url)
    paired_resp = await post_telegram_webhook(
        http_url,
        {
            "update_id": _next_test_update_id(),
            "message": {
                "message_id": 6002,
                "from": {
                    "id": PAIRED_USER_ID,
                    "is_bot": False,
                    "first_name": "Chat-Claim Tester",
                },
                "chat": {"id": PAIRED_USER_ID, "type": "private"},
                "date": int(time.time()),
                "text": "hello again",
            },
        },
        secret=WEBHOOK_SECRET,
    )
    assert paired_resp.status_code == 200

    follow_up_messages = await wait_for_sent_messages(
        fake_tg_url, min_count=1, timeout=60
    )
    follow_up_text = "\n".join(m.get("text", "") for m in follow_up_messages)
    assert "approve telegram" not in follow_up_text, (
        f"Paired user must not receive another pairing reply, got: {follow_up_text}"
    )
    assert any(
        m.get("chat_id") == PAIRED_USER_ID for m in follow_up_messages
    ), (
        f"Expected at least one reply addressed to PAIRED_USER_ID after pairing, "
        f"got: {follow_up_messages}"
    )


async def test_chat_surface_rejects_invalid_pairing_code(
    isolated_telegram_e2e_server,
):
    """Garbage codes get a clear 'invalid or expired' response, not a stuck thread.

    A second regression class #3317 hinted at: silent rejection. If the
    chat handler just routed bad codes back to the LLM, the user would
    again see an improvised "wrong place" reply. The handler must
    distinguish "valid syntax, unknown code" from "garbage input" and
    surface that as a normal turn response.
    """
    base_url = isolated_telegram_e2e_server["base_url"]
    http_url = isolated_telegram_e2e_server["http_url"]
    fake_tg_url = isolated_telegram_e2e_server["fake_tg_url"]
    channels_dir = isolated_telegram_e2e_server["channels_dir"]

    await activate_telegram(base_url, http_url, fake_tg_url, channels_dir)
    await reset_fake_tg(fake_tg_url)

    thread_r = await api_post(base_url, "/api/chat/thread/new", timeout=15)
    thread_r.raise_for_status()
    thread_id = thread_r.json()["id"]

    send_r = await api_post(
        base_url,
        "/api/chat/send",
        json={
            "content": "approve telegram NOSUCHCODE99",
            "thread_id": thread_id,
        },
        timeout=30,
    )
    assert send_r.status_code in (200, 202)

    deadline = time.monotonic() + 30
    invalid_response = None
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            history = await client.get(
                f"{base_url}/api/chat/history",
                params={"thread_id": thread_id},
                headers=auth_headers(),
                timeout=15,
            )
            history.raise_for_status()
            for turn in history.json().get("turns", []):
                response = turn.get("response", "")
                if response and "Invalid or expired pairing code" in response:
                    invalid_response = response
                    break
            if invalid_response:
                break
            await asyncio.sleep(0.5)

    assert invalid_response, (
        f"Invalid pairing claim must surface a clear rejection "
        f"within 30s for thread {thread_id}"
    )
