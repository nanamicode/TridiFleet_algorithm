from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field, model_validator


class Ad(BaseModel):
    ad_id: str
    name: str
    duration_seconds: float = Field(gt=0)
    active: bool = True
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    daily_budget: float = Field(default=250.0, gt=0)
    cost_per_play: float = Field(default=0.06, gt=0)


class ContextEvent(BaseModel):
    totem_id: str
    timestamp: datetime
    location_id: str | None = None
    region: str | None = None
    x_km: float | None = None
    y_km: float | None = None
    reach_window: int = Field(default=0, ge=0)
    impressions_window: int = Field(default=0, ge=0)
    female_share: float | None = Field(default=None, ge=0, le=1)
    mean_age: float | None = Field(default=None, ge=0, le=120)
    age_std: float | None = Field(default=None, ge=0, le=60)
    age_distribution: dict[str, float] = Field(default_factory=dict)
    flow_per_minute: float = Field(default=0.0, ge=0)
    crowd_density: float = Field(default=0.0, ge=0)


class DecisionRequest(BaseModel):
    totem_id: str
    timestamp: datetime | None = None


class Decision(BaseModel):
    decision_id: str
    ad_id: str
    totem_id: str
    sampled_score: float
    context_keys: list[str]
    policy: str = "hybrid_contextual_thompson"
    model_score: float | None = None
    residual_score: float | None = None


class Feedback(BaseModel):
    decision_id: str
    reach: int = Field(ge=0)
    impressions: int = Field(ge=0)
    avg_view_seconds: float = Field(ge=0)
    ad_duration_seconds: float = Field(gt=0)
    completion_rate: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def impressions_cannot_exceed_reach(self):
        if self.reach and self.impressions > self.reach:
            raise ValueError("impressions cannot exceed reach")
        return self


class PosteriorView(BaseModel):
    ad_id: str
    context_key: str
    alpha: float
    beta: float
    observations: float
    mean: float
