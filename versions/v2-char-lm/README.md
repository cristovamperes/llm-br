# v2 - LLM de caracteres (evolucao)

Versao evoluida do LM por caracteres para comparar hiperparametros e treino com callbacks. Mantem a abordagem char-level, com maior capacidade.

## Estrutura
- Notebook: `versions/v2-char-lm/notebooks/llm_v2.ipynb`
- Modelos: `versions/v2-char-lm/models/`
- Mapeamentos: `versions/v2-char-lm/mappings/`

## 1. Setup
- Execute a celula SETUP_ARTIFACT_PATHS para definir `MODELO_OUT` e `MAPEAMENTOS_OUT` dentro da versao.

## 2. Dados e Pre-processamento
- Mesmo corpus base (ex.: Dom Casmurro) ou outro texto.
- Normalizacao: lowercase + compactacao de espacos.

## 3. Vocabulario e Janelas de Treino
- Parametros base: `SEQ_LEN = 160`.
- Mapeamentos char->id, id->char e criacao de (X,y).

## 4. Arquitetura do Modelo
- LSTM com mais unidades: `LSTM_UNITS = 512` (ajuste conforme GPU).
- `Dense(vocab, softmax)` no topo.
- Otimizador: `Adam(lr=2e-3, clipnorm=1.0)`.

## 5. Treinamento
- `BATCH_SIZE = 256` e `EPOCHS = 40` (ajuste para sua GPU).
- Callbacks: `ModelCheckpoint(save_best_only=True)`, `ReduceLROnPlateau`, `EarlyStopping`.
- `validation_split=0.05` no `fit` para controlar LR/early stop.

## 6. Salvamento e Carregamento
- Artefatos em `models/` e `mappings/` dentro da versao.
- Reuso: `tf.keras.models.load_model(...)` e `pickle.load(...)` para mapeamentos.

## 7. Geracao de Texto
- Mesmas funcoes de geracao (top-k/nucleus) estao no notebook.
- Use seed ~160 chars do corpus e temperatura 0.7–0.9 para avaliar qualidade.

## Dicas
- Se overfitting: aumente dropout (se houver), reduza unidades, ou aumente validation_split.
- Se falta memoria: reduza `BATCH_SIZE` ou `LSTM_UNITS`.
