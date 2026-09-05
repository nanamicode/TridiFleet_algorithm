from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    max_evidence_weight: float = 12.0
    shrinkage_k: float = 20.0
    dwell_exponent: float = 0.60
    impression_exponent: float = 0.40


settings = Settings()
