# TridiFleet Algorithm

Motor de inteligência de retenção para a rede de mídia física TridiFleet / TridiAudience.

O objetivo deste repositório é responder continuamente:

> Qual anúncio deve tocar neste totem, neste momento, para maximizar a atenção visual esperada?

O backend recebe apenas telemetria já produzida pela visão computacional dos totens: alcance, impressões reais, distribuição demográfica, horário/local e tempo contínuo de visualização. O processamento de câmera/edge fica fora deste projeto.

## MVP implementado

- FastAPI para ingestão, decisão e feedback;
- Contextual Multi-Armed Bandit;
- Thompson Sampling;
- posterior Beta por anúncio/contexto;
- contexto hierárquico com backoff para evitar fragmentação;
- cold start automático para criativos novos;
- recompensa combinando taxa de olhar + profundidade de retenção;
- peso de evidência limitado por volume de alcance;
- proteção contra falsos negativos quando reach=0;
- armazenamento em memória para desenvolvimento;
- Redis para execução real com múltiplos workers/restarts;
- Docker + Docker Compose;
- testes unitários;
- simulador sintético de cidade.

## Arquitetura

Veja docs/ARCHITECTURE.md.

A decisão atual é proposital: começar com um bandit Bayesiano barato e explicável e usar os dados gerados por ele para evoluir depois para modelos com embeddings / princípios de DLRM.

## Rodar localmente

### Docker

~~~bash
docker compose up --build
~~~

API: http://localhost:8000

Swagger: http://localhost:8000/docs

### Sem Docker

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn main:app --reload
~~~

Sem REDIS_URL, o servidor usa memória local. Com Redis:

~~~bash
export REDIS_URL=redis://localhost:6379/0
uvicorn main:app --host 0.0.0.0 --port 8000
~~~

## Fluxo mínimo

### 1. Cadastrar criativo

~~~bash
curl -X POST http://localhost:8000/api/v1/ads \
  -H "content-type: application/json" \
  -d '{"ad_id":"bakery-01","name":"Pães da Padaria","duration_seconds":10,"category":"food"}'
~~~

### 2. Enviar contexto atual do totem

~~~bash
curl -X POST http://localhost:8000/api/v1/context \
  -H "content-type: application/json" \
  -d '{
    "totem_id":"55",
    "timestamp":"2026-09-05T17:00:00-03:00",
    "location_id":"mall-north",
    "region":"north",
    "reach_window":42,
    "impressions_window":12,
    "female_share":0.65,
    "mean_age":24
  }'
~~~

### 3. Pedir próximo anúncio

~~~bash
curl -X POST http://localhost:8000/api/v1/decision \
  -H "content-type: application/json" \
  -d '{"totem_id":"55"}'
~~~

Guarde o decision_id.

### 4. Enviar feedback observado

~~~bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "content-type: application/json" \
  -d '{
    "decision_id":"DECISION_ID",
    "reach":20,
    "impressions":9,
    "avg_view_seconds":6.2,
    "ad_duration_seconds":10
  }'
~~~

A atualização afeta imediatamente as próximas decisões.

## Simulação

~~~bash
python scripts/simulate_city.py
~~~

O ambiente sintético possui preferências diferentes por manhã/tarde/noite e serve para verificar se o bandit aprende a concentrar exibições nos criativos com melhor recompensa contextual.

## Testes

~~~bash
pytest -q
~~~

## Escopo atual

Este é o motor de retenção. Ainda não entram:
- leilão de mídia;
- orçamento do anunciante;
- pacing financeiro;
- cobrança;
- atribuição de venda;
- bidding.

Essas camadas devem ser construídas depois que o motor de atenção estiver validado em shadow mode e tráfego real.
