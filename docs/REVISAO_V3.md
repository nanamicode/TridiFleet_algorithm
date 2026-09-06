# Laboratório v3 — avaliação pareada e retomada

## Correções executadas

O avaliador antigo aplicava uma fórmula não linear a probabilidades médias. Ele foi
substituído pela média de 64 resultados amostrados por candidato, incluindo janelas
com zero impressões. A implementação usa os mesmos limites de permanência, ruído,
regra de conclusão e fórmula de recompensa do feedback do aprendiz.

Um sorteio separado fornece os resultados da janela efetivamente observada. Todas as
alternativas usam os mesmos números aleatórios exógenos; sua geração não depende do
anúncio escolhido. O aprendiz recebe somente o feedback da escolha feita. Os resultados
das alternativas ficam no avaliador/auditoria. Perfis e fadiga da audiência são copiados
na exposição para evitar que alterações posteriores de outros totens reescrevam o
passado daquele slot.

O ganho principal do painel compara a política com a média dos resultados potenciais
de uma rotação uniforme no mesmo slot. A média dessa rotação não exige escolher apenas
um anúncio aleatório e reduz variância. Ambos incluem os mesmos slots com público,
inclusive zero olhares. Janelas sem público não treinam nem entram na comparação.

O gráfico de expectativa usa Monte Carlo; o máximo entre estimativas de candidatos
é rotulado como teto estimado, pois tem erro amostral e viés de seleção do máximo.

## Evidência e limites explícitos

O painel agrega diferenças por totem/hora e mostra um intervalo descritivo após 20
blocos. Isso NÃO é uma garantia estatística, teste sequencial ou A/B independente.
Pessoas podem atravessar múltiplos totens; blocos podem permanecer correlacionados.
A comparação controla a audiência do slot, mas condiciona no histórico de fadiga e
orçamento produzido pela política real. Não estima o efeito de duas campanhas
independentes rodando durante dias.

O ganho do cabeçalho é acumulado; os escores do gráfico são janelas recentes.
Não misture números dos dois horizontes. O JSON inclui a configuração, versão do
avaliador, evidência pareada e limites da interpretação.

## Salvamento automático

A cada 30 segundos de execução ativa e no encerramento normal, o servidor grava
`data/lab-checkpoint.json.gz` por substituição atômica. Inclui cidade, pessoas, relógio,
RNGs, criativos, orçamentos, parâmetros bayesianos, decisões pendentes, exposições,
métricas e estado de pausa. Ao iniciar o servidor, a simulação retoma esse estado.
Uma queda abrupta pode perder o intervalo desde o último checkpoint. Pausa e suspensão
não avançam o mundo fora do processo.

O formato é JSON com tipos explicitamente permitidos, sem pickle/importação dinâmica.
Uma restauração abre uma nova execução de auditoria e referencia a execução e hash de
origem. Isso preserva o registro anterior sem reescrever o histórico após recuperação.
Checkpoints incompatíveis são rejeitados; o servidor não inventa uma cidade substituta.
Não há conversão automática de uma execução antiga v2 que existia somente em RAM.

## Validação executada

- Controle nulo: candidatos equivalentes geram exatamente o mesmo resultado pareado;
  não existe ganho artificial por escolher outro ID.
- Caso de uma pessoa: a expectativa reproduz a recompensa amostrada (~29%), em vez do
  antigo escore plug-in (~51%) para o cenário controlado do teste.
- Permanência limita tempo observado; audiência vazia produz zero evidência.
- Checkpoint em meio a decisões pendentes: continuação restaurada coincide com a
  original em população, métricas, média e variância dos parâmetros.
- Cinco seeds, três horas cada: relatório em `validation-v3.json`.

O relatório `validation.json` anterior fica como referência histórica v2, não deve
ser usado como comprovação desta revisão. As hipóteses de comportamento continuam
sintéticas e precisam de calibração com medições para representar uma cidade real.
