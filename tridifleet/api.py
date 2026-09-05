from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException

from .bandit import HierarchicalThompsonBandit
from .models import Ad, ContextEvent, Decision, DecisionRequest, Feedback, PosteriorView
from .service import RetentionService
from .store import MemoryStore


def build_store():
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        from .redis_store import RedisStore
        return RedisStore(redis_url)
    return MemoryStore()


store = build_store()
bandit = HierarchicalThompsonBandit(store)
service = RetentionService(store, bandit)

app = FastAPI(
    title="TridiFleet Retention Engine",
    version="0.1.0",
    description="MVP contextual retention intelligence API",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "store": type(store).__name__,
        "active_ads": len(store.active_ads()),
    }


@app.post("/api/v1/ads", response_model=Ad)
def upsert_ad(ad: Ad):
    store.put_ad(ad)
    return ad


@app.post("/api/v1/context", response_model=ContextEvent)
def ingest_context(event: ContextEvent):
    store.put_context(event)
    return event


@app.post("/api/v1/decision", response_model=Decision)
def decide(req: DecisionRequest):
    try:
        return bandit.choose(req.totem_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/v1/feedback")
def feedback(payload: Feedback):
    try:
        return service.apply_feedback(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/posteriors/{ad_id}", response_model=list[PosteriorView])
def posteriors(ad_id: str):
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
