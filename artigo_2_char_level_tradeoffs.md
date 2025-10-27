v1, v2, v3: Três Versões Char-Level, Três Lições Sobre Trade-offs Computacionais

Ou: como descobri (na prática) que pipeline, stride e TFLOPs importam mais do que ego de engenheiro

====================================================================

Se você chegou até aqui, provavelmente leu o Artigo 1 e já conhece o roadmap: seis versões de um LLM brasileiro, do char-level ingênuo até um GPT compacto. Este texto é o primeiro mergulho técnico de verdade. Nada de promessas; só logs, métricas, custos e cicatrizes.

O objetivo é simples: documentar o que aconteceu nas versões v1, v2 e v3 — todas ainda em caracteres puros — e o que aprendi enquanto tentava fazer uma RTX 3080 rodar com dignidade.

====================================================================

O que este artigo cobre
- Como era o baseline (v1) e por que ele quase fritou a memória
- O que mudou na v2 para reduzir 75% das janelas de treino e ainda melhorar a perplexidade
- Por que empilhar LSTMs na v3 trouxe ganhos reais… e praticamente cortou a velocidade de processamento pela metade
- As lições que custaram tempo, dólares (poucos) e algumas madrugadas olhando gráficos de treino
- Amostras reais dos modelos e como avaliei automaticamente o que eles geram

====================================================================

Recap rápido do roadmap
- Dataset: 20.000 documentos do BrWaC (mínimo 200 caracteres), limpeza padronizada (tudo em minúsculas, normalização Unicode para unificar acentos e símbolos e remoção de cabeçalhos em caixa alta).
- Tokenização: caracteres puros, vocab de 277 símbolos (com um marcador especial “UNK” para qualquer caractere que não estava no conjunto original).
- Sequência fixa de 160 caracteres; geração “autoregressiva” (o modelo prevê um caractere por vez, realimenta a própria saída) usando os controles de temperatura/top-k/top-p descritos acima.
- Treinos na mesma GPU (NVIDIA GeForce RTX 3080 via VAST.ai, spot), TensorFlow 2.20, Python 3.10.12.

====================================================================

Antes de mergulhar: dicionário rápido de métricas
- **Stride**: passo entre janelas deslizantes. Stride 1 = cria uma nova janela a cada caractere (máxima sobreposição). Stride 4 = usa uma janela a cada quatro caracteres (75% menos amostras). Quanto menor, maior o volume de dados e o custo de treino.
- **Loss (cross-entropy)**: mede o erro médio de previsão por caractere. Quanto menor, melhor. É a principal métrica de otimização durante o treino.
- **Perplexidade (PPL)**: `exp(loss)`. Indica “quantas opções” o modelo considera plausíveis em cada passo. PPL 4 significa que, em média, o modelo distribui probabilidade entre quatro caracteres. Também queremos valores menores.
- **Acurácia**: fração de caracteres previstos exatamente iguais ao alvo. Em char-level é uma métrica complementar — melhora junto com a loss, mas não captura fluência.
- **Tokens/s**: throughput real de treino. Ajuda a medir quão eficiente está o pipeline.
- **TFLOPs estimados**: custo computacional aproximado do treino. Serve para comparar quanto pagamos (em GPU e dinheiro) por cada ganho.
- **Temperatura, top-k, top-p**: controles usados na geração de texto. Temperatura < 1 deixa as escolhas mais “certinhas”; valores maiores arriscam mais. Top-k limita o próximo caractere aos k mais prováveis. Top-p (ou nucleus) mantém só o conjunto mínimo de caracteres cuja soma de probabilidade atinge p (ex.: 95%).

Com isso em mente, dá para interpretar melhor os ganhos e custos que aparecem nas seções seguintes.

====================================================================

Como avaliei os modelos
- Métricas automáticas: rodei todos os modelos no mesmo conjunto de 300 mil janelas do BrWaC (extrair 2.000 textos de validação, stride 4). Assim, loss/perplexidade/acurácia são comparáveis.
- Avaliação qualitativa automática: depois de gerar amostras com prompts inéditos, pedi para o GPT-4o-mini julgar cada texto em três eixos — relevância ao tema, coerência e estilo (escala 0-5). Isso não substitui leitura humana, mas dá um termômetro rápido do quão “aproveitável” o texto está.
- Amostras: além das métricas, todas as versões geraram saídas de 280 caracteres com temperatura 0,7, top-k 40 e top-p 0,95 para cinco prompts profissionais (infraestrutura, finanças, tecnologia, futebol, saúde). Essas amostras aparecem mais adiante.

====================================================================

Versão 1 — O baseline brutal

Arquitetura e pipeline
- `Embedding(128)` → `LSTM(256)` → `Dense(vocab, softmax)`
- (Traduzindo: primeiro converto cada caractere em um vetor denso — o “embedding”. A LSTM lê essa sequência mantendo uma memória curta para entender o contexto. A camada densa final transforma a saída da LSTM em probabilidades para cada possível próximo caractere.)
- Janelas construídas em NumPy com stride 1 (sim, 1), sem validação separada na etapa de treino
- Planejei 40 épocas, mas parei na 2ª porque a perda já tinha estabilizado
- 43.364.633 janelas por época, 13,9 bilhões de caracteres processados em 65 minutos

Resultados (BrWaC eval: 300k janelas, seq_len 160, stride 4 — o stride de avaliação é fixo pela ferramenta, independente do usado no treino)
- Loss médio: 1,5218
- Perplexidade: 4,58
- Acurácia: 55,0%
- Tempo de treino: 3.880 s (~1h05)
- Throughput: 3,58 milhões de tokens/s
- Custo estimado: ~US$ 0,15 (VAST.ai spot)
- TFLOPs estimados: 27,4 k

O que deu errado (documentado)
- Sem conjunto de validação dedicado: a métrica era puramente a acurácia de treino; impossível ajustar a taxa de aprendizado ou saber a hora de parar.
- Stride 1 explodiu o número de janelas. Resultado: pipeline CPU-bound, GPU ociosa boa parte do tempo.
- Sem `tf.data`: batches gerados em arrays ocupavam ~18 GB de RAM durante o pré-processamento.
- Os primeiros logs mostram perda oscilando sempre no mesmo padrão porque os dados eram lidos sempre na mesma ordem (nada era embaralhado).

O que deu certo (mesmo assim)
- Baseline funcional end-to-end: coleta, treino, salvamento de pesos/mapeamentos e geração.
- Já produzia texto legível em curtos trechos (ex.: “O Brasil … a marca que tirar no projeto do termo do Brasil refletir a indicado…”).
- Custou centavos: a “escola do fazer” saiu por R$ 0,80.

====================================================================

Versão 2 — Pipeline eficiente > arquitetura complexa

O que mudou tecnicamente
- `Embedding(256)` → `LSTM(512)` com dropout 0,1 + `Dense(vocab)`
- Agora os vetores de entrada são maiores (256 dimensões), a LSTM tem o dobro de neurônios para guardar contexto e o dropout desliga aleatoriamente uma parte deles durante o treino para evitar vícios.
- `tf.data` com `shuffle_buffer=200000`, batches gerados on-the-fly
- Stride 4 (mesmo seq_len), token UNK explícito (`vocab_size=277`)
- Split real: 20.000 docs treino / 2.000 docs validação
- Mesma janela de 160, mas agora “só” 10.851.158 janelas por época (−75% vs. v1)

Resultados
- Loss médio: 1,4366
- Perplexidade: 4,21 (−8,1% em relação à v1)
- Acurácia: 56,7%
- Tempo de treino: 3.156 s (~52 min)
- Throughput: 1,10 milhão de tokens/s
- Custo estimado: ~US$ 0,12
- TFLOPs: 35,7 k

O que aprendemos
- A RTX 3080 finalmente ficou ocupada: passei a usar o `tf.data` (o pipeline de dados do TensorFlow) para montar lotes enquanto a GPU treina.
- Stride 4 foi o maior ganho do experimento. Mesma qualidade com 75% menos janelas → TFLOPs ligeiramente maiores por parâmetro, mas custo/qualidade muito melhor.
- Um agendamento simples de taxa de aprendizado que reduz o passo quando a validação estabiliza evitou overfitting precoce (quando o modelo só “decora” o treino); mesmo com 2 épocas, a perda de validação seguiu caindo (1,386 → 1,339).

Desafios registrados
- O carregador de dados precisa de ~40 segundos iniciais para “encher a fila” de exemplos; durante esse tempo a GPU fica parada. Adicionamos um pré-aquecimento antes do treino principal para não desperdiçar tempo.
- Tokens/s despencaram vs. v1 (1,10 M vs. 3,58 M) porque agora o modelo é maior. Foi a primeira vez que percebi que otimizar pipeline muda onde está o gargalo.
- Mesmo com validação, as amostras geradas mostram deriva semântica após ~200 caracteres. Char-level continua tendo memória curta.

====================================================================

Versão 3 — Empilhar camadas tem custo (literalmente)

O que mudou
- Duas LSTMs de 512 unidades empilhadas, normalização entre as camadas e dropout final 0,1
- Traduzindo: coloquei uma segunda LSTM em cima da primeira para enxergar padrões de “ordem superior”. A normalização ajuda a manter os sinais estáveis entre as camadas e o dropout final serve como mais uma camada de segurança contra overfitting (o famoso “decorar” o dataset).
- Mesmo pipeline da v2 (stride 4, `tf.data`, UNK, split 20k/2k)
- Parâmetros totais: 3,89 milhões (vs. 1,79 M na v2)

Resultados
- Loss médio: 1,3753
- Perplexidade: 3,96 (−6% vs. v2; −14% vs. v1)
- Acurácia: 58,2%
- Tempo de treino: 7.143 s (~1h59)
- Throughput: 0,49 milhão de tokens/s
- Custo estimado: ~US$ 0,28
- TFLOPs: 79,4 k

Desafios e observações
- Pipeline manteve os mesmos 3,47 bilhões de tokens, mas o dobro de camadas cortou throughput pela metade. TFLOPs quase dobraram.
- Adicionar normalização de camadas e dropout na saída estabilizou a perda mais cedo, mas obrigou a ajustar finamente o parâmetro epsilon (1e-5) para evitar explosões de gradiente.
- `tf.data` passou a ser o gargalo visível: a GPU usava 99% enquanto a CPU corria atrás para gerar lotes; a taxa de janelas processadas caiu para 3,0 mil por segundo.
- Avaliação automática com prompts inéditos mostrou coerência média em torno de 1 (em escala 0-3), mas relevância ainda 0,2 — o modelo fala bonito, mas não responde ao prompt direito.

====================================================================

Comparativo lado a lado

| Versão | Loss | PPL | Acc | Tokens/época (estim.) | Tokens/s treino | Tempo total | TFLOPs | Custo spot |
|--------|------|-----|-----|-----------------------|-----------------|-------------|--------|------------|
| v1     | 1,5218 | 4,58 | 0,550 | 6,94 B | **3,58 M** | 3.880 s | 27,4 k | ~US$ 0,15 |
| v2     | 1,4366 | 4,21 | 0,567 | 1,73 B | 1,10 M | 3.156 s | 35,7 k | ~US$ 0,12 |
| v3     | **1,3753** | **3,96** | **0,582** | 1,73 B | 0,49 M | 7.143 s | 79,4 k | ~US$ 0,28 |

Observações rápidas
- v1 gastou 4× mais tokens por época por culpa do stride 1 e, mesmo assim, entregou pior métrica. Pipeline ruim joga contra.
- v2 provou que “fazer menos” (menos janelas, mais batching) vale mais do que dobrar unidades. Melhor custo por melhoria de loss.
- v3 entrega o melhor número, mas à custa de throughput dividido por 2 e TFLOPs 2× maiores. Rendimentos decrescentes são reais.

====================================================================

O que os modelos geram na prática

Todos os trechos abaixo (e na avaliação automática) foram gerados com temperatura 0,7, top-k 40, top-p 0,95. Os prompts têm cerca de 200 caracteres e são truncados/padronizados para a janela de 160 caracteres que cada modelo entende; depois disso eles completam mais 280 caracteres sozinhos.

**Prompt 101 – Infraestrutura logística**  
Prompt: "O setor de infraestrutura logistica brasileira debate novas concessoes ferroviarias, cronogramas de duplicacao, metas de produtividade e integracao com portos para desafogar corredores de exportacao."

> **v1** (Relevância 0, Coerência 1, Estilo 1)
> a matÃ©ria e as empresas e do conteÃºdo de gente como o mundo a contratos e reverter para concentraÃ§Ã£o de encontro de papel e de acordo no estado de ator com a rede de alguns melhores palavras do projeto do seguro de um conteÃºdo do direito com o povo com a maior de um pacacionaliz

> **v2** (Relevância 0, Coerência 1, Estilo 1)
> como ter uma tentativa de mortes de tradicionalizaÃ§Ã£o de uma base de modo que se interessa a escolha de legislatura e constituinte de uma marca de crime infecto de transporte com a conceitualidade.
> em 0 o regime de vista Ã© que o destaque com o processo de estado com pai a princi

> **v3** (Relevância 1, Coerência 1, Estilo 1)
> a matÃ©ria especializada pode ser desprezada com o texto ativo do projeto de verdadeiro conjunto de processos promovidos na consciÃªncia externa de responsabilidade do computador do paÃ­s a considerar o entÃ£o de uma apresentaÃ§Ã£o de design assistir com a escola informada.
> a abordage


**Prompt 102 – Mercado financeiro**  
Prompt: "Analistas do mercado financeiro reavaliam previsoes trimestrais para bancos listados, discutem juros, carteira de credito, inadimplencia corporativa e estrategias de hedge diante de volatilidade externa."

> **v1** (Relevância 0, Coerência 0, Estilo 1)
> esse seu amor sua presenÃ§a de um amor encontrar no mundo na caracterÃ­stica na estratÃ©gia de casamento de alimentos de todos os novos palavras e mais campos para o macoe podem ser criados em menina e como um entra o forma feira municipal do processo de reconhecido com o que nÃ£o p

> **v2** (Relevância 0, Coerência 1, Estilo 1)
> nÃ£o previsÃ£o procedimento da luz a receber como prestaÃ§Ã£o de direitos apresentados por apoio de regimento de alunos de maior produÃ§Ã£o de registrar a cada dia no processo de uma coisa com continentalidade para ser feita no campo.
> Ã© o materialista foi contato ao capitalismo como o

> **v3** (Relevância 0, Coerência 1, Estilo 1)
> menos problemas de uma pergunta de televisÃ£o em que traz seus perÃ­odos especÃ­ficos e atendimentos de especializaÃ§Ã£o de produtos de produÃ§Ã£o de condiÃ§Ãµes para por meio da polÃ­cia federal, estejam em que o tipo de composiÃ§Ã£o da contratada apresentou uma polÃ­tica de constituiÃ§Ã£o de


**Prompt 103 – Fintechs e microserviços**  
Prompt: "Engenheiros de software de uma fintech planejam migracao para microservicos, definem SLAs, monitoramento proativo, politicas de rollback continuo e treinamentos para times de suporte e compliance."

> **v1** (Relevância 0, Coerência 1, Estilo 1)
> a presenÃ§a do baixo com a cabeÃ§a e a distribuiÃ§Ã£o de associaÃ§Ã£o com o senado ao conjunto do brasil com calma de investigaÃ§Ã£o por empregado em meio antes de construÃ§Ã£o pela rede em maio de nossos estados unidos e na restrito santo antigo tratados com a receita na proposta em um e

> **v2** (Relevância 0, Coerência 1, Estilo 1)
> a pressÃ£o em lagos da edificaÃ§Ã£o da polÃ­cia Ã© uma revista de concentraÃ§Ã£o da assembleia local, que conseguiu a instituiÃ§Ã£o de substÃ¢ncias de responsÃ¡vel por meio de finalidade de dia.
> a sua consideraÃ§Ã£o entre os polÃ­ticos nas regiÃµes de vida e se apresentou a ser instituÃ­do pelo

> **v3** (Relevância 0, Coerência 1, Estilo 1)
> a receita de empresas da cadeia de comunicaÃ§Ã£o de seu reverÃªncia do projeto da universidade de sensaÃ§Ã£o (andiel e tratamento ou polÃ­ticas de repercussÃ£o do governo de transporte e efetivamento da construÃ§Ã£o de seus resultados para o seu trabalho com entrevistas em nome de paulo


**Prompt 104 – Clube de futebol**  
Prompt: "A diretoria de um clube de futebol negocia patrocinio master, reforcos na janela de transferencia, metas de bilheteria, programa de socios e auditoria das contas apos uma temporada irregular nos gramados."

> **v1** (Relevância 0, Coerência 1, Estilo 1)
> na estrutura de conta as contratados com a importÃ¢ncia da vida para trabalho de motor como a democratica e comparaÃ§Ã£o com a crianÃ§a de se presente sÃ£o a diretoria do amor na agosto, no manutenÃ§Ã£o da extra a via tomar que foram de energia ao serviÃ§o do processo em um estado de co

> **v2** (Relevância 0, Coerência 0, Estilo 1)
> o cerca de 0 anos e composiÃ§Ã£o da fila de trabalho no presidente do trabalho de produÃ§Ã£o e condiÃ§Ãµes de responsÃ¡vel pelo processo de produÃ§Ã£o de seu primeiro caso de 0 de maio de 0, o ministÃ©rio de resultados adversÃ¡rios e trÃªs anos de previsÃ£o e proporcionalidade, a primeira di

> **v3** (Relevância 0, Coerência 1, Estilo 1)
> o contrato que estÃ¡ como a proposta de despesas de projetos de serviÃ§os de instituiÃ§Ã£o de energia de energia do trabalho e da educaÃ§Ã£o no presidente da cidade de polÃ­cia federal e profissionais e paranÃ¡ a expectativa do contrato de disposiÃ§Ã£o de futebol, na administraÃ§Ã£o do curs


**Prompt 105 – Rede de hospitais**  
Prompt: "Uma rede de hospitais privados avalia expansao para telemedicina, protocolos integrados de prontuario, parcerias com seguradoras, capacidade de leitos, gestao de estoque farmacologico e contratacao medica."

> **v1** (Relevância 0, Coerência 1, Estilo 1)
> uma conseqÃ¼Ãªncia de mim com o produto de motor e escolher em arma de cabelo com a todo o processo do profissional e o carro do centro do seu estado a viagem de desenvolvimento de paraÃ­ba e mais apresentar com explicar para casa e de educaÃ§Ã£o em vocÃª na arrecionalidade do paÃ­s.
> o

> **v2** (Relevância 0, Coerência 1, Estilo 1)
> uma construÃ§Ã£o da prova de forma forma da rede mais uma destacar a direita a diretor de aprendizagens para elevaÃ§Ã£o de contratos e mais atenÃ§Ã£o da capital de lima.
> no campo da constituiÃ§Ã£o da presidente livre do meu nÃºmero de setembro de 0 e preÃ§os de parte da produÃ§Ã£o do dia 0,

> **v3** (Relevância 0, Coerência 1, Estilo 1)
> uma delegacia de profissionais e os polÃ­ticos de micros, resolveu conseguir a polÃ­cia em relaÃ§Ã£o Ã  vida em negÃ³cios e a linha para o pesquisador de anÃ¡lise e depender de disposiÃ§Ã£o de dados de uma concessÃ£o de vida, nÃ£o conseguiu demonstrar que uma escola de mesmo com o que pret


Mesmo quando a coerência estilística melhora, a relevância segue quase nula. A média das avaliações automáticas ficou em relevância 0,0 (v1), 0,0 (v2) e 0,2 (v3). Tradução: char-level acerta a sintaxe, mas não entrega conteúdo útil. Precisamos de subword/contexto semântico.

====================================================================

As 4 lições que custaram caro
1. Pipeline primeiro, arquitetura depois. Migrar para `tf.data` e ajustar stride entregou mais ganho que dobrar neurônios.
2. Stride é hiperparâmetro crítico. Stride 4 manteve qualidade e reduziu 75% dos tokens — com impacto direto em TFLOPs e dólares.
3. Char-level tem teto. Mesmo com perplexidade 3,96, os modelos não respondem prompts específicos; falta granularidade semântica.
4. TFLOPs são a moeda real. v3 consumiu 79,4 k TFLOPs para render +0,015 de acurácia. Qualquer passo futuro precisa justificar custo.

====================================================================

Desafios enfrentados na prática
- Memória: geração de janelas em NumPy (v1) chegou a usar 18 GB de RAM e quase matou a instância. Resolvido ao migrar para `tf.data`.
- Throughput: `shuffle_buffer` de 200k leva ~40s para encher; adicionamos warm-up explícito antes do treino.
- Diagnóstico: estabeleci uma rotina automática pós-treino — avalio 300k janelas por modelo usando o mesmo conjunto de validação.
- Qualidade: uso de prompts customizados mostrou irrelevância temática e motivou a v4 (subword).

====================================================================

Próximos passos (spoiler do Artigo 3)
- v4 (subword) já está em treinamento com SentencePiece 4k + byte fallback. Sequências menores, contexto mais rico, menos `<unk>`.
- Objetivo: comparar perplexidade/custo com v3 e observar se a relevância dos prompts sobe acima de zero.
- Também vamos medir overhead de tokenização, latência de decodificação e custo por TFLOPs em pipelines híbridos (char vs. subword).

No próximo artigo, conto se a troca para subword valeu o esforço — e mostro os primeiros resultados com tokens que fazem sentido para humanos, não apenas para contadores de caracteres.

====================================================================

Se isso foi útil, compartilha. Quanto mais gente brasileira entender os bastidores, menos dependentes seremos de caixas-pretas importadas. E sim, tudo custou menos de um café. Até a próxima.
