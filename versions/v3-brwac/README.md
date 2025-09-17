# v3 - Treino com BrWaC

LM por caracteres treinado no corpus BrWaC via Hugging Face Datasets. Usa a classe `LLMSimplesGenerico` em `llm_generico.py` para preparo de dados e treino.

## Estrutura
- Script (baseline): `versions/v3-brwac/scripts/treinar_com_brwac.py`
- Script (recomendado, tf.data + limpeza): `versions/v3-brwac/scripts/treinar_com_brwac_tfdata.py`
- Modelos: `versions/v3-brwac/models/` (ex.: `modelo_brwac.keras`)
- Mapeamentos: `versions/v3-brwac/mappings/` (ex.: `mapeamentos_brwac.pkl`)

## 1. Setup
- Por padrao, o script salva artefatos nas pastas desta versao.
- Pode customizar com `--modelo_saida` e `--mapeamentos_saida`.

## 2. Dados e Pre-processamento
- Dataset: `nlpufg/brwac` (Hugging Face Datasets), split `train`.
- Limpeza (script recomendado): normaliza Unicode (NFKC), remove controles, preserva quebras de linha e trata marcadores `<END>` como separadores de documento.
- Split por documento (treino/validacao) sem usar `validation_split` por janelas.
- Amostragem: `--max_textos` limita a quantidade de textos por conjunto (treino/val). Use `--min_len` para filtrar textos muito curtos.

## 3. Vocabulario e Janelas de Treino
- Implementado internamente em `LLMSimplesGenerico` (char->id, id->char, janelas (X,y)).

## 4. Arquitetura do Modelo
- Embedding -> `LSTM(tamanho_lstm)` -> `Dense(vocab, softmax)`.
- Parametros comuns: `tamanho_sequencia=160`, `tamanho_lstm=256..512`.

## 5. Treinamento
- Baseline: usa callbacks (ModelCheckpoint, ReduceLROnPlateau, EarlyStopping). Se `--validacao_split > 0`, monitora `val_loss`.
- Recomendado (tf.data): gera janelas sob demanda com `--stride`, reduz custo de memoria e permite split por documento.
- Exemplo baseline (10k textos):
```
python versions/v3-brwac/scripts/treinar_com_brwac.py \
  --epocas 10 --max_textos 10000 \
  --tamanho_sequencia 160 --tamanho_lstm 256 --batch_size 256
```
- Exemplo recomendado (tf.data + limpeza + split doc-level):
```
python versions/v3-brwac/scripts/treinar_com_brwac_tfdata.py \
  --epocas 10 --max_textos 10000 \
  --tamanho_sequencia 160 --tamanho_lstm 256 --embedding_dim 128 \
  --batch_size 256 --stride 3 --dropout 0.2 --clipnorm 1.0 \
  --validacao_split 0.1
```
- Em GPUs (ex.: RTX 3070), experimente `--tamanho_lstm 512` e `--batch_size 256`.

## 6. Salvamento e Carregamento
- Modelo `.keras` e mapeamentos `.pkl` sao salvos nos caminhos informados (ou padrao da versao).
- Para gerar texto posteriormente, carregue modelo e mapeamentos e use a funcao de geracao (vide v1/v2) ou um dos utilitarios: `compare_generate.py` (simples) ou `compare_generate_v2.py` (com top-k/top-p).

## 7. Geracao de Texto
- Opcao 1: `compare_generate.py` na raiz:
  - `python compare_generate.py --only v3 --prompt "Seu prompt" --length 400 --temperature 0.8`
- Opcao 2: `compare_generate_v2.py` com top-k/top-p:
  - `python compare_generate_v2.py --only v3 --prompt "Seu prompt" --length 400 --temperature 0.8 --top_k 40 --top_p 0.95`
- Opcao 3: implementar uma rotina de geracao similar a v1/v2 a partir dos mapeamentos.

> Mais detalhes do script recomendado em `versions/v3-brwac/README_v2.md`.

## Dicas
- `--max_textos` e `--epocas` controlam tempo/qualidade; aumente gradualmente.
- Se faltar memoria: reduza `--batch_size` e/ou `--tamanho_lstm`.
