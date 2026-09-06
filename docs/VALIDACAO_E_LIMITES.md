# Validação e limites do laboratório

## O que está implementado

Bandit contextual híbrido com Thompson Sampling, regressão bayesiana diagonal,
interações entre tags e demografia/horário/fluxo, e históricos por criativo e contexto.
O componente Beta usa pseudocontagens de recompensa fracionária: é uma aproximação,
não um posterior Bernoulli exato para retenção contínua. A regressão também aproxima
as correlações entre parâmetros por uma covariância diagonal.

Não há uma cópia do Meta Ads nem uma rede DLRM treinada neste projeto. A conexão
com DLRM é o uso de atributos numéricos, categóricos e interações para generalizar.
Referências primárias:
- https://arxiv.org/abs/1707.02038 — Thompson Sampling.
- https://arxiv.org/abs/1906.00091 — DLRM.

## Experimento reproduzível

Execute `python scripts/validate_learning.py`. O relatório inclui **todas** as seeds,
com três horas por seed e sensor de 6 m. `docs/validation.json` contém o resultado
medido nesta revisão. Não selecionamos apenas a melhor execução.

O ganho usa escores contrafactuais no mesmo público e conjunto elegível do slot.
A fórmula é aplicada às taxas médias da função oculta: por ser não linear, esse
escore **não é a esperança matemática exata da recompensa amostral**. A linha de
recompensa observada, sujeita a amostras pequenas e ruído, não deve ser comparada
diretamente com a baseline calculada. A baseline não é uma campanha aleatória
independente com histórico próprio de fadiga e orçamento. Logo o ganho é um
diagnóstico local, não uma estimativa causal de ganho de campanha inteira.

Para comprovação em campo: holdout aleatório por totem/período, análise por clusters,
intervalos de confiança, definição prévia da métrica e avaliação fora do treino.
A simulação testa o software e a capacidade de aprender um mundo artificial;
não valida as hipóteses demográficas como fatos sobre uma cidade brasileira.

## Integridade corrigida

- Abrir a interface não consome RNG da população.
- Iterações de conjuntos que consomem RNG são ordenadas entre processos.
- Física em passos fixos de 1 s, independente de agrupamento de chamadas/velocidade.
- Tempo de olhar limitado pela maior permanência consecutiva observada no sensor.
  É um limite físico discretizado; gaze ainda é uma amostra agregada por slot,
  não uma reconstrução quadro a quadro de cada olhar.
- Feedback repetido não retreina. Payload conflitante é rejeitado.
- Decisões concluídas liberam memória, inclusive com alcance zero.
- Até 10.000 decisões pendentes; ao atingir a capacidade, novos pedidos são recusados
  explicitamente. Os últimos 10.000 resultados aceitos suportam repetição idempotente.
- Orçamento reservado na seleção, por custo fixo de exibição. Não há cobrança
  adicional fictícia por impressão nem ocultação de gastos usando `min(budget, gasto)`.
- Hash encadeado registra contexto, candidatos, escolha e resultados. Detecta alterações
  acidentais, mas não protege contra um administrador que reescreva toda a cadeia.

## Relógio e operação

1× significa 120 segundos **simulados** por segundo de execução, e não acelerar o
hardware. Os botões oferecem 0,25×, 0,5× e retorno a 1×; não existe opção acima de 1×.
Se o computador não acompanhar, a simulação fica mais lenta. Nenhum passo físico é
pulado para fabricar progresso. Pausa interrompe mundo e aprendizado juntos.

Fechar a aba não para o servidor. Fechar o processo Python, desligar ou suspender o
computador interrompe o processamento. Docker pode manter o servidor em segundo
plano com `docker compose up -d --build`. O histórico SQLite fica no diretório `data`.
**Reiniciar o servidor preserva o histórico, mas não restaura automaticamente a cidade
nem os pesos em memória.** Iniciar novamente cria outro experimento.

## Escala e integração

A API `/api/v1` recebe contextos e feedbacks agregados, e retorna decisões. Ela usa
um aprendiz separado do laboratório para não contaminar os dados reais com sintéticos.
Use um único processo/worker por aprendiz nesta versão. Redis armazena parte do estado,
mas não torna os parâmetros do modelo compartilhados entre processos nem reiniciáveis.
Não use múltiplos workers como se fossem uma IA única.

A cidade aceita 1–500 totens, raio de 0,5–25 km e sensor configurável de 1–30 m.
Esse limite de configuração não é um SLA de desempenho. A população visual é uma
amostra sintética limitada, não um censo populacional. Big data distribuído, ingestão
com fila durável, particionamento, restauração integral do aprendiz e testes de carga
em hardware real ainda são etapas de produção. O código atual é um laboratório local
funcional; não representa uma infraestrutura comprovada para milhões de eventos/s.
