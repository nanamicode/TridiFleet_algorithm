from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .auth import login, logout, require_admin, websocket_is_admin
from .intelligence import HybridRetentionIntelligence
from .models import Ad, ContextEvent, Decision, DecisionRequest, Feedback, PosteriorView
from .service import RetentionService
from .sim.domain import SimConfig
from .sim.engine import SimulationManager
from .store import MemoryStore


def build_store():
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        from .redis_store import RedisStore
        return RedisStore(redis_url)
    return MemoryStore()


store = build_store()
intelligence = HybridRetentionIntelligence(store)
service = RetentionService(store, intelligence)
simulation = SimulationManager()

app = FastAPI(
    title="TridiFleet Retention Engine",
    version="0.2.0",
    description="Contextual retention intelligence + local digital-twin laboratory",
)

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class LoginBody(BaseModel):
    username: str
    password: str


class ConfigureBody(BaseModel):
    n_totems: int = Field(ge=1, le=500)
    radius_km: float = Field(ge=0.5, le=25)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)


class PauseBody(BaseModel):
    paused: bool


class SpeedBody(BaseModel):
    speed: float


@app.get("/")
def root():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    engine = simulation.get()
    return {
        "status": "ok",
        "store": type(store).__name__,
        "active_ads": len(store.active_ads()),
        "lab_configured": engine is not None,
        "lab_running": bool(engine and engine.running),
    }


@app.post("/api/auth/login")
def auth_login(body: LoginBody, response: Response):
    if not login(body.username, body.password, response):
        raise HTTPException(status_code=401, detail="invalid credentials")
    return {"ok": True, "username": body.username}


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response):
    logout(request, response)
    return {"ok": True}


@app.get("/api/auth/me")
def auth_me(admin: str = Depends(require_admin)):
    return {"username": admin, "role": "admin"}


# Standalone retention API ----------------------------------------------------

@app.post("/api/v1/ads", response_model=Ad)
def upsert_ad(ad: Ad, _: str = Depends(require_admin)):
    store.put_ad(ad)
    return ad


@app.post("/api/v1/context", response_model=ContextEvent)
def ingest_context(event: ContextEvent, _: str = Depends(require_admin)):
    store.put_context(event)
    return event


@app.post("/api/v1/decision", response_model=Decision)
def decide(req: DecisionRequest, _: str = Depends(require_admin)):
    try:
        return intelligence.choose(req.totem_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/v1/feedback")
def feedback(payload: Feedback, _: str = Depends(require_admin)):
    try:
        return service.apply_feedback(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/posteriors/{ad_id}", response_model=list[PosteriorView])
def posteriors(ad_id: str, _: str = Depends(require_admin)):
    return [
        PosteriorView(
            ad_id=ad_id,
            context_key=key,
            alpha=p.alpha,
            beta=p.beta,
            observations=p.observations,
            mean=p.mean,
        )
        for key, p in store.list_posteriors(ad_id)
    ]


@app.get("/api/v1/intelligence")
def intelligence_diagnostics(_: str = Depends(require_admin)):
    return intelligence.diagnostics()


# Digital twin laboratory ----------------------------------------------------

def _engine():
    engine = simulation.get()
    if engine is None:
        raise HTTPException(status_code=409, detail="simulation not configured")
    return engine


@app.get("/api/lab/status")
def lab_status(_: str = Depends(require_admin)):
    engine = simulation.get()
    if engine is None:
        return {"configured": False}
    snap = engine.snapshot()
    return {
        "configured": True,
        "running": snap["running"],
        "paused": snap["paused"],
        "speed": snap["speed"],
        "sim_time": snap["sim_time"],
        "n_totems": engine.config.n_totems,
        "radius_km": engine.config.radius_km,
        "seed": engine.config.seed,
    }


@app.post("/api/lab/configure")
def lab_configure(body: ConfigureBody, _: str = Depends(require_admin)):
    try:
        engine = simulation.configure(
            SimConfig(
                n_totems=body.n_totems,
                radius_km=body.radius_km,
                seed=body.seed,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "ok": True,
        "n_totems": engine.config.n_totems,
        "radius_km": engine.config.radius_km,
        "seed": engine.config.seed,
    }


@app.post("/api/lab/start")
def lab_start(_: str = Depends(require_admin)):
    engine = _engine()
    engine.start_background()
    return {"ok": True, "running": True}


@app.post("/api/lab/pause")
def lab_pause(body: PauseBody, _: str = Depends(require_admin)):
    engine = _engine()
    engine.set_paused(body.paused)
    return {"ok": True, "paused": engine.paused}


@app.post("/api/lab/speed")
def lab_speed(body: SpeedBody, _: str = Depends(require_admin)):
    engine = _engine()
    speed = engine.set_speed(body.speed)
    return {"ok": True, "speed": speed, "max_speed": 1.0}


@app.get("/api/lab/map")
def lab_map(_: str = Depends(require_admin)):
    return _engine().city.serialize()


@app.get("/api/lab/state")
def lab_state(_: str = Depends(require_admin)):
    return _engine().snapshot()


@app.get("/api/lab/metrics")
def lab_metrics(_: str = Depends(require_admin)):
    return _engine().metrics()


@app.get("/api/lab/creatives")
def lab_creatives(_: str = Depends(require_admin)):
    return _engine().creative_stats()


@app.post("/api/lab/creatives", response_model=Ad)
def lab_add_creative(ad: Ad, _: str = Depends(require_admin)):
    engine = _engine()
    if ad.ad_id in engine.store.ads:
        raise HTTPException(status_code=409, detail="ad_id already exists")
    return engine.add_creative(ad)


@app.get("/api/lab/totems/{totem_id}")
def lab_totem(totem_id: str, _: str = Depends(require_admin)):
    detail = _engine().totem_detail(totem_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="totem not found")
    return detail


@app.websocket("/ws/lab")
async def lab_socket(websocket: WebSocket):
    if not websocket_is_admin(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    try:
        while True:
            engine = simulation.get()
            if engine is None:
                await websocket.send_json({"configured": False})
            else:
                await websocket.send_json(engine.snapshot())
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        return
