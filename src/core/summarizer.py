# core/summarizer.py
#
# Author:  Logicish
# Company: Logic-Ish Designs
# Date:    2/26/2026
#
# ==================================================
# Conversation summarization and context management.
# Generates summary via LLM utility slot, preserves
# recent messages within a token budget, and wipes
# old history.
#
# Token budgets are percentage-based with a static
# system header carve-out. This scales across any
# context window size:
#   remaining = slot_context - header_budget
#   summary cap = remaining * summary_max_percent
#   recent budget = remaining * keep_recent_percent
#
# Lock behavior:
#   flag_crit -> fire immediately as background task
#   flag_warn + is_idle -> background timer picks it up
#   If slot_lock held when message arrives -> wait up
#   to SUMMARIZE_LOCK_WAIT seconds, then drop.
#
# Fallback: if the utility call fails for any reason,
# emergency_trim() does a dumb trim -- keeps recent
# messages within budget, preserves existing summary,
# clears flags. Partial memory beats a bricked user.
#
# Background loop also runs proactive health checks
# on the LLM server and triggers recovery if needed.
#
# Knows about: config (thresholds, lock wait, scheduled
#              settings, idle interval, budgets),
#              slots (User, save), llm (utility call,
#              health, recovery).
# ==================================================

# ==================================================
# Imports
# ==================================================
import asyncio

import structlog

from config import (
    SUMMARIZE_LOCK_WAIT,
    SCHEDULED_SUMMARY,
    IDLE_CHECK_INTERVAL,
    CONTEXT_PER_SLOT,
    SYSTEM_HEADER_BUDGET,
    SUMMARY_MAX_TOKENS,
    KEEP_RECENT_TOKENS,
    CHARS_PER_TOKEN,
)
from core.slots import User, save_profile, get_all_users

log = structlog.get_logger()

# ==================================================
# Token Estimation
# ==================================================

def _estimate_tokens(text: str) -> int:
    # conservative estimate -- fewer chars per token means
    # we assume more tokens, so we stay under budget
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN


def _estimate_message_tokens(msg: dict) -> int:
    # estimate tokens for a single message including role overhead
    # role/formatting adds roughly 4 tokens per message
    return _estimate_tokens(msg.get("content", "")) + 4


# ==================================================
# Message Selection
# ==================================================

def _select_recent_messages(history: list[dict], budget: int) -> list[dict]:
    # walk backwards through history, keep messages that
    # fit within the token budget. always keep at least 1
    # message pair (user + assistant) if possible.
    if not history:
        return []

    selected = []
    tokens_used = 0

    for msg in reversed(history):
        msg_tokens = _estimate_message_tokens(msg)
        if tokens_used + msg_tokens > budget and selected:
            break
        selected.append(msg)
        tokens_used += msg_tokens

    selected.reverse()

    log.info("recent_messages_selected",
             count=len(selected), estimated_tokens=tokens_used,
             budget=budget)
    return selected


# ==================================================
# Scheduled Summary Task
# ==================================================

_scheduler_task: asyncio.Task | None = None


async def start_scheduler():
    global _scheduler_task
    if not SCHEDULED_SUMMARY.get("enabled", False):
        log.info("scheduled_summary_disabled")
        return

    cron_str = SCHEDULED_SUMMARY.get("cron", "0 2 * * *")
    _scheduler_task = asyncio.create_task(_scheduler_loop(cron_str))
    log.info("scheduled_summary_started", cron=cron_str)


async def stop_scheduler():
    global _scheduler_task
    if _scheduler_task and not _scheduler_task.done():
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
    _scheduler_task = None
    log.info("scheduled_summary_stopped")


async def _scheduler_loop(cron_str: str):
    # simple cron-like loop -- parses "M H * * *" format
    # only supports hour and minute for daily scheduling
    parts = cron_str.split()
    target_minute = int(parts[0]) if parts[0] != "*" else 0
    target_hour   = int(parts[1]) if parts[1] != "*" else 2

    while True:
        try:
            from datetime import datetime, timedelta
            dt = datetime.now()

            target = dt.replace(
                hour=target_hour, minute=target_minute, second=0, microsecond=0
            )
            if target <= dt:
                target += timedelta(days=1)

            wait_seconds = (target - dt).total_seconds()
            log.info("scheduler_next_run",
                     target=target.isoformat(), wait_seconds=int(wait_seconds))

            await asyncio.sleep(wait_seconds)
            await _run_scheduled_summary()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("scheduler_error", error=str(e))
            await asyncio.sleep(60)


async def _run_scheduled_summary():
    log.info("scheduled_summary_triggered")
    users = get_all_users()

    for uid, user in users.items():
        if uid == "utility":
            continue
        try:
            async with user.slot_lock:
                await _do_summarize(user)
        except Exception as e:
            log.error("scheduled_summary_user_failed",
                      user_id=uid, error=str(e))

    log.info("scheduled_summary_complete")

    if SCHEDULED_SUMMARY.get("restart_llm", False):
        log.info("scheduled_llm_restart")
        from core import llm
        await llm.restart()


# ==================================================
# Public API
# ==================================================

async def summarize_if_needed(user: User):
    # fire-and-forget background task
    # acquires slot_lock, summarizes, saves profile
    if user.slot_lock.locked():
        log.info("slot_lock_held_skip", user_id=user.user_id)
        return

    try:
        async with user.slot_lock:
            log.info("summarizing", user_id=user.user_id, slot=user.slot)
            await _do_summarize(user)
            log.info("summarization_complete", user_id=user.user_id)
    except Exception as e:
        log.error("summarization_failed", user_id=user.user_id, error=str(e))


async def emergency_summarize(user: User):
    # called by main.py when context overflow is detected
    # during a call. acquires lock, summarizes, so the
    # retry has room to work.
    log.warning("emergency_summarize_triggered", user_id=user.user_id)
    try:
        async with user.slot_lock:
            await _do_summarize(user)
            log.info("emergency_summarize_complete", user_id=user.user_id)
    except Exception as e:
        log.error("emergency_summarize_failed",
                   user_id=user.user_id, error=str(e))


async def check_idle_summarize(user: User):
    # called periodically -- if flag_warn and user is idle, summarize
    if user.flag_warn and user.is_idle():
        log.info("idle_summarize_triggered", user_id=user.user_id)
        await summarize_if_needed(user)


async def wait_for_lock(user: User) -> bool:
    if not user.slot_lock.locked():
        return True

    log.info("waiting_for_slot_lock", user_id=user.user_id)
    for _ in range(SUMMARIZE_LOCK_WAIT):
        await asyncio.sleep(1)
        if not user.slot_lock.locked():
            return True

    log.warning("slot_lock_timeout",
                user_id=user.user_id, timeout=SUMMARIZE_LOCK_WAIT)
    return False


# ==================================================
# Background Loop -- Idle Checks + Health Monitor
# ==================================================

_background_task: asyncio.Task | None = None


async def start_background_loop():
    global _background_task
    _background_task = asyncio.create_task(
        _background_loop(IDLE_CHECK_INTERVAL)
    )
    log.info("background_loop_started", interval=IDLE_CHECK_INTERVAL)


async def stop_background_loop():
    global _background_task
    if _background_task and not _background_task.done():
        _background_task.cancel()
        try:
            await _background_task
        except asyncio.CancelledError:
            pass
    _background_task = None
    log.info("background_loop_stopped")


async def _background_loop(interval: int):
    while True:
        try:
            await asyncio.sleep(interval)

            # proactive LLM health check
            await _check_llm_health()

            # idle summarization checks
            users = get_all_users()
            for uid, user in users.items():
                if user.flag_warn and user.is_idle():
                    await summarize_if_needed(user)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("background_loop_error", error=str(e))


async def _check_llm_health():
    from core import llm

    if not llm.is_running():
        log.warning("background_health_llm_not_running")
        await llm.attempt_recovery()
        return

    healthy = await llm.health_check()
    if not healthy:
        log.warning("background_health_check_failed")
        await llm.attempt_recovery()


# ==================================================
# Internal -- Summarization Logic
# ==================================================

async def _do_summarize(user: User):
    if len(user.conversation_history) < 2:
        log.info("not_enough_history", user_id=user.user_id)
        return

    # split history: messages to summarize vs messages to keep
    recent = _select_recent_messages(
        user.conversation_history, KEEP_RECENT_TOKENS
    )
    # everything before the recent window gets summarized
    keep_count = len(recent)
    to_summarize = user.conversation_history[:-keep_count] if keep_count else user.conversation_history

    if not to_summarize:
        log.info("nothing_to_summarize", user_id=user.user_id,
                 recent_count=keep_count)
        return

    # build the summarization prompt using only the old messages
    history_text = _format_history(to_summarize)
    existing_summary = user.summary or "(No previous summary)"

    # estimate if the prompt fits in the utility slot
    prompt_content = (
        f"Previous summary:\n{existing_summary}\n\n"
        f"New conversation:\n{history_text}\n\n"
        "Produce an updated summary."
    )
    system_text = (
        "You are a summarization assistant. Produce a concise summary "
        "of the conversation below. Preserve key facts, decisions, and "
        "context the user would need if the conversation continued."
    )

    estimated_prompt_tokens = (
        _estimate_tokens(system_text) +
        _estimate_tokens(prompt_content) + 8  # message framing overhead
    )

    # check if the prompt itself fits in the utility slot
    # utility slot has the same per-slot context as everyone else
    utility_budget = CONTEXT_PER_SLOT - SUMMARY_MAX_TOKENS - 16  # leave room for response + framing
    if estimated_prompt_tokens > utility_budget:
        # the old messages are too long even after removing recent --
        # truncate the history text to fit
        max_history_chars = (utility_budget - _estimate_tokens(system_text) -
                             _estimate_tokens(existing_summary) - 50) * CHARS_PER_TOKEN
        if max_history_chars > 0:
            history_text = history_text[:max_history_chars]
            log.warning("summarize_prompt_truncated",
                         user_id=user.user_id,
                         estimated=estimated_prompt_tokens,
                         budget=utility_budget)
        else:
            # can't fit anything -- fall back to emergency trim
            log.warning("summarize_prompt_too_large_fallback",
                         user_id=user.user_id)
            _emergency_trim(user, recent)
            return

        # rebuild prompt with truncated history
        prompt_content = (
            f"Previous summary:\n{existing_summary}\n\n"
            f"New conversation:\n{history_text}\n\n"
            "Produce an updated summary."
        )

    summary_messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": prompt_content},
    ]

    try:
        from core import llm
        result = await llm.call_utility(
            messages=summary_messages,
            temperature=0.3,
            max_tokens=SUMMARY_MAX_TOKENS,
        )
        new_summary = result.content
    except Exception as e:
        log.error("summarize_llm_error", user_id=user.user_id, error=str(e))
        # utility call failed -- do emergency trim so user isn't bricked
        _emergency_trim(user, recent)
        return

    # apply the new summary and keep only recent messages
    user.summary = new_summary
    user.conversation_history = recent
    user.flag_warn = False
    user.flag_crit = False

    save_profile(user)
    log.info("summary_updated",
             user_id=user.user_id,
             kept_recent=len(recent),
             summary_tokens=_estimate_tokens(new_summary))


def _emergency_trim(user: User, recent: list[dict] | None = None):
    # last resort -- keep recent messages, preserve whatever
    # summary already exists, clear flags. user loses some
    # context but can keep talking.
    if recent is None:
        recent = _select_recent_messages(
            user.conversation_history, KEEP_RECENT_TOKENS
        )

    old_count = len(user.conversation_history)
    user.conversation_history = recent
    user.flag_warn = False
    user.flag_crit = False

    save_profile(user)
    log.warning("emergency_trim_applied",
                 user_id=user.user_id,
                 old_messages=old_count,
                 kept_messages=len(recent))


def _format_history(history: list[dict]) -> str:
    lines = []
    for msg in history:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)