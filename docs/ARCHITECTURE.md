# Architecture — TridiFleet Retention Engine

## Objective

Transform aggregated computer-vision telemetry from physical advertising totens into an online decision system that chooses the creative with the highest expected visual retention for the current context.

This repository intentionally focuses on the retention MVP. Auction, billing, advertiser bidding and commercial optimization are outside the current scope.

## Online loop

1. Totem publishes current context (POST /api/v1/context).
2. Totem requests the next creative (POST /api/v1/decision).
3. The engine samples every active creative with contextual Thompson Sampling.
4. The chosen creative is displayed.
5. Totem sends delayed observed outcome (POST /api/v1/feedback).
6. Posterior state is updated immediately and influences the next decision.

## Signals

Current MVP consumes:
- totem and location identity;
- timestamp / daypart;
- region;
- reach;
- impressions;
- female share;
- estimated mean age;
- continuous average viewing time;
- creative duration.

Raw images are not required by this backend.

## Reward

The primary objective is visual attention. We combine attention capture (impressions / reach) and retention depth (average continuous viewing time / creative duration).

Current reward:

    reward = impression_rate^0.40 * dwell_ratio^0.60

The result is clipped to [0, 1].

The exponent choice deliberately gives slightly more importance to sustained attention than to the first look. These values are configuration, not immutable product truth; once real traffic exists, offline replay and online experiments should tune them.

### Evidence weighting

One aggregate window must not move the model as much as an entire day. Posterior updates therefore use a bounded evidence weight:

    weight = min(12, sqrt(reach))

A window with zero reach has zero evidence and never punishes a creative.

## Context hierarchy

A naive implementation would create one independent bucket such as female_18_24_afternoon_north_totem_55. That creates severe sparsity.

Instead, the engine learns several overlapping levels:
- global
- daypart
- demographic bucket
- daypart + demographic
- region
- region + daypart
- location
- individual totem

The sampling function applies shrinkage. New/sparse contexts stay close to broad/global knowledge; mature contexts increasingly control their own decisions.

## Thompson Sampling

Each (creative, context) maintains a Beta posterior:
- alpha: accumulated positive evidence
- beta: accumulated negative evidence

New creatives start at Beta(1,1), which is deliberately uncertain. This naturally creates cold-start exploration without a fixed 10% exploration rule.

For a decision:
1. sample each context posterior;
2. blend from most-specific to broadest using confidence n / (n + k);
3. use the global posterior once as the fallback;
4. pick the creative with the highest sampled value.

This continuously balances exploration and exploitation.

## State

### Development

MemoryStore — zero infrastructure, deterministic tests and simulations.

### Deployment

RedisStore — shared state for multiple API workers, restart-safe posterior state, current context and delayed decision lookup.

Set REDIS_URL=redis://host:6379/0.

## Data lake / big-data boundary

Redis is not the historical warehouse. Production should duplicate immutable telemetry and decisions to a cheap append-only analytical path, initially:

    API -> buffered event stream -> object storage (Parquet) -> offline analytics/training

On AWS this can become Kinesis/Firehose -> S3 -> Athena, but the decision engine is intentionally not coupled to AWS-specific APIs.

## DLRM relationship and evolution

DLRM ideas are useful here, but a full DLRM is intentionally not the first model.

The MVP already separates conceptually:

### Sparse/categorical features
- totem id
- location
- region
- creative id/category
- time bucket

### Dense features
- female share
- mean age
- reach
- impression rate
- dwell ratio

The evolution path is:

### Phase 1 — current
Hierarchical Contextual Thompson Sampling. Cheap, explainable, online, excellent cold start for a small daily creative inventory.

### Phase 2
Add continuous context with Linear Thompson Sampling / Bayesian logistic reward model. This reduces manual bucketing while preserving online uncertainty.

### Phase 3
Train embeddings and a DLRM-like contextual reward predictor from the accumulated historical lake. The predictor should provide a prior/expected reward to the bandit, not replace exploration.

Conceptually:

    DLRM-like predictor -> contextual prior -> Thompson/uncertainty layer -> decision

This keeps deep learning responsible for generalization and the bandit responsible for online exploration.

## Why not pure DLRM now?

The problem has dozens/hundreds of active creatives, not billions of feed candidates. Retrieval is therefore unnecessary in the MVP. A large neural ranker would add infrastructure cost, calibration work and cold-start risk before enough labels exist.

The best initial architecture is an online Bayesian decision engine that creates the training data required by future deep models.

## Production milestones

1. Validate telemetry contract with real totens.
2. Replay synthetic + recorded traffic offline.
3. Run shadow mode: compute decisions without controlling displays.
4. Compare baseline rotation vs bandit expected reward.
5. Canary on a small subset of totens.
6. Add append-only event lake and observability.
7. Tune reward and evidence weighting from real outcomes.
8. Only then introduce learned embeddings / DLRM-like priors.
