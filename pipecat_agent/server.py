import argparse
import sys
import uuid

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from loguru import logger
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from contextlib import asynccontextmanager

from bot import run_bot
from llm import get_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading vLLM Gemma engine...")
    await get_engine()
    logger.info("vLLM engine ready")
    yield


app = FastAPI(lifespan=lifespan)

app.mount("/prebuilt", SmallWebRTCPrebuiltUI)

small_webrtc_handler = SmallWebRTCRequestHandler()


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/prebuilt/")


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    async def on_connection(connection):
        background_tasks.add_task(run_bot, connection)

    answer = await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=on_connection,
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


# -- Pipecat Cloud-style /start + session proxy (needed by prebuilt UI) ------

active_sessions: dict[str, dict] = {}


@app.post("/start")
async def start(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    session_id = str(uuid.uuid4())
    active_sessions[session_id] = body

    result: dict = {"sessionId": session_id}
    if body.get("enableDefaultIceServers"):
        result["iceConfig"] = {
            "iceServers": [{"urls": "stun:stun.l.google.com:19302"}]
        }
    return result


@app.api_route(
    "/sessions/{session_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def session_proxy(
    session_id: str, path: str, request: Request, background_tasks: BackgroundTasks
):
    if session_id not in active_sessions:
        return Response(content="Unknown session", status_code=404)

    if not path.endswith("api/offer"):
        return Response(status_code=200)

    body = await request.json()

    if request.method == "POST":
        webrtc_request = SmallWebRTCRequest(
            sdp=body["sdp"],
            type=body["type"],
            pc_id=body.get("pc_id"),
            restart_pc=body.get("restart_pc"),
            request_data=body,
        )
        return await offer(webrtc_request, background_tasks)

    if request.method == "PATCH":
        patch_request = SmallWebRTCPatchRequest(
            pc_id=body["pc_id"],
            candidates=[IceCandidate(**c) for c in body.get("candidates", [])],
        )
        return await ice_candidate(patch_request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Gemma dialogue agent")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logger.remove(0)
    if args.verbose:
        logger.add(sys.stderr, level="TRACE")
    else:
        logger.add(sys.stderr, level="DEBUG")

    uvicorn.run(app, host=args.host, port=args.port)
