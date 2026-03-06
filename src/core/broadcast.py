# core/broadcast.py
#
# Author:  Logicish
# Company: Logic-Ish Designs
# Date:    3/3/2026
#
# ==================================================
# Lightweight pub/sub broadcast bus.
# Allows multiple clients to subscribe to a user's
# output stream. Disabled by default — a module must
# call enable() at runtime to activate.
#
# When disabled, publish() is a no-op with zero
# overhead. When enabled, events are pushed to all
# subscriber queues for the target user.
#
# Subscribers are identified by asyncio.Queue objects.
# Transport creates a queue per SSE listener connection
# and cleans up on disconnect.
#
# Security: subscribe() only allows same-user access.
# This is enforced at the transport layer (Gate 1),
# not here — the bus itself is user-keyed but dumb.
#
# Knows about: nothing — this is a leaf dependency.
# ==================================================

# ==================================================
# Imports
# ==================================================
import asyncio

import structlog

log = structlog.get_logger()

# ==================================================
# State
# ==================================================

_enabled: bool = False
_subscribers: dict[str, set[asyncio.Queue]] = {}

# ==================================================
# Runtime Toggle
# ==================================================

def enable():
    global _enabled
    _enabled = True
    log.info("broadcast_enabled")


def disable():
    global _enabled
    _enabled = False
    log.info("broadcast_disabled")


def is_enabled() -> bool:
    return _enabled


# ==================================================
# Subscribe / Unsubscribe
# ==================================================

def subscribe(user_id: str) -> asyncio.Queue:
    # creates a new queue for this subscriber and
    # registers it under the user_id. caller is
    # responsible for calling unsubscribe on disconnect.
    queue: asyncio.Queue = asyncio.Queue()
    if user_id not in _subscribers:
        _subscribers[user_id] = set()
    _subscribers[user_id].add(queue)
    log.info("broadcast_subscribe",
             user_id=user_id,
             listeners=len(_subscribers[user_id]))
    return queue


def unsubscribe(user_id: str, queue: asyncio.Queue):
    if user_id in _subscribers:
        _subscribers[user_id].discard(queue)
        if not _subscribers[user_id]:
            del _subscribers[user_id]
    log.info("broadcast_unsubscribe",
             user_id=user_id,
             listeners=len(_subscribers.get(user_id, set())))


# ==================================================
# Publish
# ==================================================

def publish(user_id: str, event: dict):
    # push an event to all subscribers for this user.
    # no-op if broadcast is disabled or no listeners.
    # event format: {"event": "token|final|init|done", "data": ...}
    if not _enabled:
        return

    queues = _subscribers.get(user_id)
    if not queues:
        return

    for queue in queues:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            log.warning("broadcast_queue_full", user_id=user_id)


def has_subscribers(user_id: str) -> bool:
    # quick check so callers can skip building events
    # when nobody is listening
    if not _enabled:
        return False
    return bool(_subscribers.get(user_id))