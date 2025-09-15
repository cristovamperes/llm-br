# v1 - LLM simples por caracteres

Guia didatico: modelo gerador de texto por caracteres em Keras (LSTM). Mantem a arquitetura simples para focar nos fundamentos (tokenizacao por caracteres, janelas deslizantes e geracao autoregressiva).

## Estrutura
- Notebook: `versions/v1-char-rnn/notebooks/llm.ipynb`
- Modelos: `versions/v1-char-rnn/models/`
- Mapeamentos: `versions/v1-char-rnn/mappings/`

## 1. Setup
- Execute a celula SETUP_ARTIFACT_PATHS no inicio do notebook. Ela define:
  - `MODELO_OUT` -> `versions/v1-char-rnn/models/modelo_char_rnn.keras`
  - `MAPEAMENTOS_OUT` -> `versions/v1-char-rnn/mappings/mapeamentos.pkl`

## 2. Dados e Pre-processamento
- Corpus: Dom Casmurro (Project Gutenberg).
- Normalizacao recomendada: `texto = texto.lower()` e `texto = ' '.join(texto.split())` para reduzir vocabulario e ruido.

## 3. Vocabulario e Janelas de Treino
- Cria mapeamentos char->id e id->char e pares (X,y) por janelas fixas.
- Parametros base: `TAMANHO_SEQUENCIA = 160`.

## 4. Arquitetura do Modelo
- `LSTM(256)` seguida de `Dense(vocab, activation='softmax')`.
- Otimizador: `Adam(lr=2e-3, clipnorm=1.0)`.

## 5. Treinamento
- `BATCH_SIZE = 256` (ajuste conforme memoria GPU) e `EPOCAS_TREINO = 40`.
- Callbacks: `ModelCheckpoint(save_best_only=True)`, `ReduceLROnPlateau`, `EarlyStopping`.
- Opcional: `validation_split=0.05` no `fit` para melhor ajuste de LR/early stop.

## 6. Salvamento e Carregamento
- Modelo salvo em `.keras` e mapeamentos em `.pkl` nos caminhos de versao.
- Carregamento: `tf.keras.models.load_model(NOME_ARQUIVO_MODELO)` e `pickle.load` para mapeamentos.

## 7. Geracao de Texto
- Use a celula de geracao com top-k/nucleus adicionada ao notebook:
  - `gerar_texto(model, char_para_int, int_para_char, seed, comprimento=400, k=20, temperatura=0.8)`
- Dicas:
  - Seed ~160 chars do proprio corpus melhora a fluencia.
  - Temperatura 0.7–0.9 tende a gerar texto mais natural que 0.2.

## Passo a passo rapido
1) Setup -> 2) Dados -> 3) Vocab -> 4) Modelo -> 5) Treino -> 6) Salvar/Carregar -> 7) Geracao.

## Solucao de problemas
- OOM: reduza `BATCH_SIZE` e/ou `TAMANHO_SEQUENCIA`.
- Saida repetitiva: aumente temperatura e/ou use top-k (k=20) ou nucleus (p=0.9).
