"""SGIC – FastAPI server"""
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from auth import (
    delete_session,
    fetch_emergent_session,
    get_user_from_request,
    store_session,
    upsert_user,
)
import sgic

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

app = FastAPI(title="SGIC API")
api = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

class SessionRequest(BaseModel):
    session_id: str


@api.post("/auth/session")
async def create_session(payload: SessionRequest, response: Response):
    profile = await fetch_emergent_session(payload.session_id)
    user = await upsert_user(db, profile)
    expires = await store_session(db, user["user_id"], profile["session_token"])
    response.set_cookie(
        key="session_token",
        value=profile["session_token"],
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        expires=expires,
    )
    return {"user": user}


@api.get("/auth/me")
async def auth_me(request: Request):
    user = await get_user_from_request(db, request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@api.post("/auth/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("session_token")
    if token:
        await delete_session(db, token)
    response.delete_cookie("session_token", path="/")
    return {"ok": True}


# ---------------------------------------------------------------------------
# SGIC endpoints (auth-protected)
# ---------------------------------------------------------------------------

async def _require_user(request: Request) -> dict:
    user = await get_user_from_request(db, request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@api.post("/embed")
async def embed(
    request: Request,
    image: UploadFile = File(...),
    message: str = Form(...),
    secret_key: str = Form(...),
):
    await _require_user(request)
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if not secret_key.strip():
        raise HTTPException(status_code=400, detail="Secret key cannot be empty")
    image_bytes = await image.read()
    try:
        result = sgic.run_embed(image_bytes, message, secret_key)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Embed failed: {exc}")
    return result


@api.post("/extract")
async def extract(
    request: Request,
    image: UploadFile = File(...),
    secret_key: str = Form(...),
    original_message: str = Form(""),
):
    await _require_user(request)
    image_bytes = await image.read()
    return sgic.run_extract(image_bytes, secret_key, original_message)


class RobustnessRequest(BaseModel):
    stego_image: str  # base64 data URL
    secret_key: str
    original_message: str


@api.post("/robustness")
async def robustness(req: RobustnessRequest, request: Request):
    await _require_user(request)
    return {"results": sgic.run_robustness(req.stego_image, req.secret_key, req.original_message)}


@api.post("/ablation")
async def ablation(
    request: Request,
    image: UploadFile = File(...),
    message: str = Form(...),
    secret_key: str = Form(...),
):
    await _require_user(request)
    image_bytes = await image.read()
    return {"results": sgic.run_ablation(image_bytes, message, secret_key)}


@api.get("/comparison")
async def comparison():
    return {"methods": sgic.COMPARISON_TABLE}


@api.get("/graphs")
async def graphs():
    return {
        "epochs": sgic.EPOCHS,
        "psnr_vs_epoch": sgic.PSNR_VS_EPOCH,
        "ssim_vs_epoch": sgic.SSIM_VS_EPOCH,
        "timing_ms": sgic.TIMING_MS,
        "steganalysis": sgic.STEGANALYSIS,
        "loss_curves": {
            "epochs": sgic.LOSS_CURVES["epochs"],
            "train": [round(float(v), 6) for v in sgic.LOSS_CURVES["train"]],
            "val": [round(float(v), 6) for v in sgic.LOSS_CURVES["val"]],
        },
    }


@api.get("/")
async def root():
    return {"service": "SGIC", "status": "ok"}


app.include_router(api)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sgic")


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
