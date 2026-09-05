# TridiFleet Algorithm

Motor de inteligência de retenção + laboratório de gêmeo digital para a rede TridiFleet / TridiAudience.

A pergunta central é:

> **Qual criativo deve tocar neste totem, neste momento, para maximizar a atenção visual esperada?**

## Estado atual

O repositório já contém duas partes integradas:

### 1. Inteligência real

O motor atual é um **Hybrid Contextual Thompson Sampling**:

- Thompson Sampling hierárquico por criativo/contexto;
- backoff global -> horário -> demografia -> região -> localização -> totem;
- modelo Bayesiano compartilhado por features;
- hashing de tags arbitrárias de criativo;
- generalização para criativos novos;
- incerteza explícita para exploração;
- feedback online contínuo;
- recompensa de retenção limitada a [0,1].

O objetivo atual permanece retenção. Leilão, bidding e cobrança não fazem parte do aprendizado.

### 2. Digital Twin Lab

Uma cidade 2D procedural roda localmente com:

- ruas arteriais, coletoras e locais;
- zonas urbanas;
- pessoas entrando, saindo e caminhando pelas ruas;
- distribuição demográfica;
- fluxo variável durante o dia;
- dezenas ou centenas de totens;
- exposição real por janelas;
- gaze/impressões simulados;
- tempo contínuo de visualização;
- orçamento diário por criativo;
- troca autônoma de criativos;
- passagem de dias;
- pausa e velocidades 0,25x / 0,5x / 1x;
- painel web em tempo real;
- inspeção individual de totens;
- lista e criação de novos criativos;
- gráficos de aprendizado.

A simulação continua rodando no processo Python mesmo que a aba seja fechada.

Documentação de integridade experimental: `docs/DIGITAL_TWIN.md`.

Arquitetura do motor: `docs/ARCHITECTURE.md`.

## Rodar

### Docker — recomendado

~~~bash
docker compose up --build
~~~

Abra:

~~~text
http://localhost:8000
~~~

Login local padrão:

~~~text
usuário: admin
senha: tridifleet-local
~~~

O Docker expõe a aplicação somente em `127.0.0.1` por padrão.

Antes de expor a outra máquina, altere:

~~~bash
TRIDIFLEET_ADMIN_USER=...
TRIDIFLEET_ADMIN_PASSWORD=...
~~~

### Sem Docker

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn main:app
~~~

## Começando uma simulação

Depois do login:

1. escolha a quantidade de totens;
2. escolha o raio da cidade em km;
3. escolha a seed;
4. gere a cidade;
5. a simulação e o aprendizado começam no servidor local.

O mapa mostra as pessoas se movendo e os totens tomando decisões. Clique em qualquer totem para ver anúncio atual, alcance, impressões, retenção, demografia e reward.

## Criativo novo

Abra **Criativos** e cadastre algo como:

~~~text
ID: SOAP-001
Nome: Sabonete Mercado Centro
Categoria: higiene
Tags: hygiene, supermarket, family, home
Duração: 10
Gasto diário: 400
~~~

Ele entra imediatamente no conjunto de opções. Não existe uma regra manual dizendo onde mostrar esse anúncio: a incerteza Bayesiana e o aprendizado por tags/contexto determinam como ele será explorado.

## Integridade

A IA não controla nem conhece a função que gera os resultados.

~~~text
mundo oculto
    |
    v
sensor simulado
    |
    v
telemetria agregada
    |
    v
INTELIGÊNCIA
    |
    v
escolha do criativo
    |
    v
mundo oculto gera o resultado
    |
    v
feedback atrasado
~~~

O oracle e a baseline aleatória existem somente no avaliador do laboratório e não entram nas features da IA.

## Testes

~~~bash
pytest -q
~~~

O GitHub Actions executa os testes automaticamente a cada push.
