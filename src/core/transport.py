# core/transport.py
#
# Author:  Logicish
# Company: Logic-Ish Designs
# Date:    2/26/2026
#
# ==================================================
# FastAPI HTTP server.
# Receives requests, authenticates, identifies user,
# calls handle_message or handle_stream, returns
# response. Supports both JSON and SSE endpoints.
# GATE 1 — first security checkpoint.
#
# Broadcast listener endpoint:
#   GET /channel/listen/{user_id} — SSE subscription.
#   Requires Gate 1 auth via query param. Same-user
#   only. Returns 503 if broadcast is disabled.
#
# Admin endpoints:
#   GET /admin/dump — full prompt dump for all users.
#   GET /admin/dump/{user_id} — dump for one user.
#   Both require ADMIN-level Gate 1.
#
# Knows about: config (SecurityLevel), slots (resolve,
#              get_user), llm (health, restart, session),
#              broadcast (subscribe, enabled check).
# ==================================================

# ==================================================
# Imports
# ==================================================
import asyncio
from datetime import datetime
from typing import Callable

import structlog
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from config import SecurityLevel
from core import slots, llm, broadcast

log = structlog.get_logger()


# ==================================================
# Payload Schemas
# ==================================================

class MessagePayload(BaseModel):
    user_id:         str        = "guest"
    message:         str        = Field(..., min_length=1, max_length=4096)
    conversation_id: str | None = None
    input_type:      str        = "text"    # text, voice, image
    extra:           dict | None = None

    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
    }


class AdminPayload(BaseModel):
    # lightweight payload for admin-only endpoints
    user_id: str = Field(..., min_length=1)

    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
    }


# ==================================================
# Route Factory
# ==================================================

def create_routes(
    app: FastAPI,
    handle_message: Callable,
    handle_stream: Callable,
):
    # --- GATE 1 helper ---
    def _gate1(user_id: str):
        # resolve user identity — unknown maps to guest or None
        resolved = slots.resolve_user(user_id)
        if resolved is None:
            log.warning("gate1_reject_unknown", user_id=user_id)
            return None, JSONResponse(
                status_code=403,
                content={"error": "unknown_user", "detail": "Not in slot map"},
            )

        user = slots.get_user(resolved)
        if user is None or user.security_level < SecurityLevel.GUEST:
            log.warning("gate1_reject_access", user_id=resolved)
            return None, JSONResponse(
                status_code=403,
                content={"error": "access_denied"},
            )

        return user, None

    # --- ADMIN gate helper ---
    def _gate_admin(user_id: str):
        user, error = _gate1(user_id)
        if error:
            return None, error
        if user.security_level < SecurityLevel.ADMIN:
            log.warning("gate_admin_denied", user_id=user_id,
                         level=user.security_level)
            return None, JSONResponse(
                status_code=403,
                content={"error": "access_denied",
                         "detail": "ADMIN level required"},
            )
        return user, None

    # --------------------------------------------------
    # POST /channel/chat — JSON request/response
    # --------------------------------------------------
    @app.post("/channel/chat")
    async def chat_json(payload: MessagePayload):
        user_id = payload.user_id.lower().strip()
        preview = payload.message[:60] + ("..." if len(payload.message) > 60 else "")
        log.info("request_chat", user_id=user_id, preview=preview)

        user, error = _gate1(user_id)
        if error:
            return error

        try:
            result = await handle_message(
                user.user_id,
                payload.message,
                conversation_id=payload.conversation_id,
            )
        except Exception as e:
            log.error("handle_message_failed", user_id=user.user_id, error=str(e))
            return JSONResponse(
                status_code=500,
                content={"error": "internal", "detail": str(e)},
            )

        return {
            "response":        result,
            "user_id":         user.user_id,
            "conversation_id": payload.conversation_id,
            "timestamp":       datetime.now().isoformat(),
        }

    # --------------------------------------------------
    # POST /channel/chat/stream — SSE streaming
    # --------------------------------------------------
    @app.post("/channel/chat/stream")
    async def chat_stream(payload: MessagePayload):
        user_id = payload.user_id.lower().strip()
        preview = payload.message[:60] + ("..." if len(payload.message) > 60 else "")
        log.info("request_stream", user_id=user_id, preview=preview)

        user, error = _gate1(user_id)
        if error:
            return error

        conv_id = payload.conversation_id

        async def event_generator():
            # init event — self-describing stream metadata
            yield {
                "event": "init",
                "data": f'{{"user_id": "{user.user_id}", "conversation_id": "{conv_id or ""}"}}',
            }
            try:
                async for chunk in handle_stream(
                    user.user_id,
                    payload.message,
                    conversation_id=conv_id,
                ):
                    yield {"event": "token", "data": chunk}
                yield {
                    "event": "done",
                    "data": conv_id or "",
                }
            except Exception as e:
                log.error("stream_failed", user_id=user.user_id, error=str(e))
                yield {"event": "error", "data": str(e)}

        return EventSourceResponse(event_generator())

    # --------------------------------------------------
    # GET /channel/listen/{user_id} — broadcast listener
    # --------------------------------------------------
    @app.get("/channel/listen/{target_user_id}")
    async def listen_stream(
        target_user_id: str,
        user_id: str = Query(..., description="Authenticated user_id"),
    ):
        # check if broadcast is enabled
        if not broadcast.is_enabled():
            return JSONResponse(
                status_code=503,
                content={"error": "broadcast_disabled",
                         "detail": "Broadcast is not enabled. A module must enable it."},
            )

        # gate 1 — authenticate the requesting user
        user, error = _gate1(user_id.lower().strip())
        if error:
            return error

        # same-user enforcement — can only listen to your own stream
        target = target_user_id.lower().strip()
        if user.user_id != target:
            log.warning("listen_denied_wrong_user",
                         requesting=user.user_id, target=target)
            return JSONResponse(
                status_code=403,
                content={"error": "access_denied",
                         "detail": "Can only listen to your own stream"},
            )

        # subscribe and stream events
        queue = broadcast.subscribe(user.user_id)

        async def listener_generator():
            try:
                yield {
                    "event": "init",
                    "data": f'{{"user_id": "{user.user_id}", "listening": true}}',
                }
                while True:
                    event = await queue.get()
                    yield event
            except asyncio.CancelledError:
                pass
            finally:
                broadcast.unsubscribe(user.user_id, queue)

        return EventSourceResponse(listener_generator())

    # --------------------------------------------------
    # POST /llm/restart — ADMIN-gated
    # --------------------------------------------------
    @app.post("/llm/restart")
    async def llm_restart(payload: AdminPayload):
        user_id = payload.user_id.lower().strip()
        log.info("llm_restart_requested", user_id=user_id)

        user, error = _gate_admin(user_id)
        if error:
            return error

        success = await llm.restart()
        return {
            "success":   success,
            "user_id":   user.user_id,
            "timestamp": datetime.now().isoformat(),
        }

    # --------------------------------------------------
    # GET /admin/dump — all users prompt dump
    # --------------------------------------------------
    @app.get("/admin/dump")
    async def admin_dump_all(
        user_id: str = Query(..., description="ADMIN user_id for auth"),
    ):
        admin, error = _gate_admin(user_id.lower().strip())
        if error:
            return error

        users = slots.get_all_users()
        dump = {}
        for uid, user in users.items():
            if uid == "utility":
                continue
            dump[uid] = {
                "slot":       user.slot,
                "security":   user.security_level,
                "persona":    user.persona,
                "summary":    user.summary,
                "messages":   user.build_messages(),
                "flag_warn":  user.flag_warn,
                "flag_crit":  user.flag_crit,
                "is_idle":    user.is_idle(),
                "history_len": len(user.conversation_history),
            }
        return {"dump": dump, "timestamp": datetime.now().isoformat()}

    # --------------------------------------------------
    # GET /admin/dump/{target_user_id} — single user dump
    # --------------------------------------------------
    @app.get("/admin/dump/{target_user_id}")
    async def admin_dump_user(
        target_user_id: str,
        user_id: str = Query(..., description="ADMIN user_id for auth"),
    ):
        admin, error = _gate_admin(user_id.lower().strip())
        if error:
            return error

        target = target_user_id.lower().strip()
        user = slots.get_user(target)
        if user is None:
            return JSONResponse(
                status_code=404,
                content={"error": "user_not_found", "detail": f"No user '{target}'"},
            )

        return {
            "user_id":    user.user_id,
            "slot":       user.slot,
            "security":   user.security_level,
            "persona":    user.persona,
            "summary":    user.summary,
            "messages":   user.build_messages(),
            "flag_warn":  user.flag_warn,
            "flag_crit":  user.flag_crit,
            "is_idle":    user.is_idle(),
            "history_len": len(user.conversation_history),
            "timestamp":  datetime.now().isoformat(),
        }

    # --------------------------------------------------
    # GET /health
    # --------------------------------------------------
    @app.get("/health")
    async def health():
        return {
            "status":      "ok",
            "llm_running": llm.is_running(),
            "llm_pid":     llm.get_pid(),
            "broadcast":   broadcast.is_enabled(),
            "timestamp":   datetime.now().isoformat(),
        }

    # --------------------------------------------------
    # GET /slots — show active user slot info
    # --------------------------------------------------
    @app.get("/slots")
    async def slot_status():
        users = slots.get_all_users()
        info = {}
        for uid, user in users.items():
            info[uid] = {
                "slot":       user.slot,
                "security":   user.security_level,
                "flag_warn":  user.flag_warn,
                "flag_crit":  user.flag_crit,
                "is_idle":    user.is_idle(),
                "history_len": len(user.conversation_history),
                "has_summary": bool(user.summary),
            }
        return {"slots": info}

    # --------------------------------------------------
    # Catch-all 404
    # --------------------------------------------------
    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    )
    async def catch_all(request: Request, path: str):
        log.warning("route_not_found", path=f"/{path}")
        return JSONResponse(
            status_code=404,
            content={"error": f"Route /{path} not found"},
        )