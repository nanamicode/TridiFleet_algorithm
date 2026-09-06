# TridiFleet Digital Twin Lab

## Purpose

The lab is not a visual mock. It is a local, continuously running experimental environment for the same retention intelligence that can later serve real totens.

The browser is only a control/observation surface. World evolution and learning run in the Python server process, so closing the browser tab does not pause the simulation.

## Experimental integrity

The simulator is intentionally split into three trust boundaries.

### 1. Hidden world

Files under `tridifleet/sim/ground_truth.py` define the latent response of simulated people.

Hidden variables include:
- personal interests;
- attention propensity;
- creative intrinsic quality;
- hidden demographic fit;
- hidden time-of-day fit;
- proximity effects;
- crowd distraction;
- stochastic gaze and dwell noise.

The intelligence is never given these values.

### 2. Simulated edge sensor

`tridifleet/sim/sensor.py` converts the world into the type of aggregated telemetry a physical totem could send.

It includes measurement error for:
- pedestrian detection;
- age estimation;
- apparent gender classification.

The model receives only this sensor output plus delayed creative feedback.

### 3. Intelligence

`tridifleet/intelligence.py` has no import or reference to the ground-truth model.

It sees:
- time;
- location;
- recent flow;
- recent reach / impressions;
- observed demographic aggregates;
- creative metadata/tags;
- delayed retention feedback.

It does not see:
- hidden interests of individuals;
- hidden creative quality;
- expected reward of unplayed creatives;
- oracle ranking;
- baseline score.

The simulator may compute random-baseline and oracle metrics for evaluation, but those numbers are never used as training features.

## City

A city is generated from:
- radius in km;
- number of totens;
- deterministic seed.

The road system has local, collector and arterial roads. Intersections are denser around the core. Zones include downtown, residential, commercial, offices, transit, leisure, school and health areas.

Totems are preferentially placed at higher-value intersections rather than uniformly in empty space.

## Population

People:
- enter and leave the city;
- follow the generated road grid;
- have walking speeds in realistic pedestrian ranges;
- follow a mixed age distribution;
- carry hidden interest sets;
- have individual attention propensity.

Population volume follows a smooth urban day curve with morning, lunch and evening peaks.

Each visual dot is an individual simulated pedestrian, not an aggregate KPI invented after the fact.

## Exposure windows

A totem does not count only people present at the decision instant.

During an active creative slot it accumulates unique pedestrian IDs that pass inside its detection radius. Each person is kept once using the nearest observed distance during that slot.

At slot completion:
1. hidden physics decides which people actually looked;
2. continuous dwell time is generated per impression;
3. completion is generated;
4. the sensor aggregates the outcome;
5. only then does the intelligence receive feedback.

This keeps decision and outcome causally ordered.

## Nominal clock

At 1x, the laboratory currently advances 120 simulated seconds per real second.

This is the lab's nominal clock, not a claim that real hardware can run faster than itself.

Controls are intentionally limited to:
- 0.25x
- 0.5x
- 1x

There is no >1x turbo mode. Returning to 1x only undoes a previous slowdown.

## Creative inventory and cold start

The initial inventory includes many local-city categories:
- bakery;
- supermarket;
- coffee;
- salon;
- barbershop;
- pharmacy;
- gym;
- restaurants;
- pet shop;
- dental;
- fashion;
- electronics;
- school;
- automotive;
- real estate;
- delivery;
- cosmetics;
- laundry;
- furniture;
- tourism;
- finance;
- cleaning/hygiene;
- entertainment.

An admin can add a creative while the simulation is running with:
- ID;
- name;
- category;
- arbitrary tags;
- duration;
- daily budget.

The hybrid Bayesian model hashes arbitrary tags into a shared feature space. A new soap/supermarket creative can therefore borrow prior statistical structure from related tags while Thompson uncertainty still forces honest exploration.

## Evaluation

The metrics drawer compares:
- observed AI reward;
- random-policy expected baseline;
- hidden oracle ceiling;
- uplift vs random;
- posterior/model uncertainty.

Oracle and random expectations are evaluation-only and cannot influence a decision.

## Login

The local default is:

- user: `admin`
- password: `tridifleet-local`

Docker binds the UI to `127.0.0.1:8000` by default. Change both environment variables before exposing the service to any other machine:

- `TRIDIFLEET_ADMIN_USER`
- `TRIDIFLEET_ADMIN_PASSWORD`


## Reward contract

The optimizer currently uses **visual_attention_v2**:

    capture = impressions / reach
    dwell = average_view_seconds / creative_duration
    reward = capture^0.40 * dwell^0.60

Completion rate is still measured and displayed, but it is intentionally not an
optimization term. In a physical low-volume slot, completion can be based on one
or two people and creates excessive reward variance. Continuous dwell carries
more stable information and is the primary retention signal.

The CI runs a deterministic full-city benchmark and requires the learned policy
to beat random rotation by at least 8% expected reward after the quick training
window. This is a regression floor, not a claimed production performance number.
