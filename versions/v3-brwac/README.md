# v3 - Treino com BrWaC

LM por caracteres treinado no corpus BrWaC via Hugging Face Datasets. Usa a classe `LLMSimplesGenerico` em `llm_generico.py` para preparo de dados e treino.

## Estrutura
- Script: `versions/v3-brwac/scripts/treinar_com_brwac.py`
- Modelos: `versions/v3-brwac/models/` (ex.: `modelo_brwac.keras`)
- Mapeamentos: `versions/v3-brwac/mappings/` (ex.: `mapeamentos_brwac.pkl`)

## 1. Setup
- Por padrao, o script salva artefatos nas pastas desta versao.
- Pode customizar com `--modelo_saida` e `--mapeamentos_saida`.

## 2. Dados e Pre-processamento
- Dataset: `nlpufg/brwac` (Hugging Face Datasets), split `train`.
- Limpeza simples: remover quebras de linha extras e normalizar espacos.
- Amostragem: `--max_textos` limita a quantidade de textos concatenados.

## 3. Vocabulario e Janelas de Treino
- Implementado internamente em `LLMSimplesGenerico` (char->id, id->char, janelas (X,y)).

## 4. Arquitetura do Modelo
- Embedding -> `LSTM(tamanho_lstm)` -> `Dense(vocab, softmax)`.
- Parametros comuns: `tamanho_sequencia=160`, `tamanho_lstm=256..512`.

## 5. Treinamento
- Exemplo (amostra de 10k textos):
```
python versions/v3-brwac/scripts/treinar_com_brwac.py \
  --epocas 10 --max_textos 10000 \
  --tamanho_sequencia 160 --tamanho_lstm 256 --batch_size 256
```
- Em GPUs (ex.: RTX 3070), experimente `--tamanho_lstm 512` e `--batch_size 256`.

## 6. Salvamento e Carregamento
- Modelo `.keras` e mapeamentos `.pkl` sao salvos nos caminhos informados (ou padrao da versao).
- Para gerar texto posteriormente, carregue modelo e mapeamentos e use a funcao de geracao (vide v1/v2) ou o utilitario `compare_generate.py`.

## 7. Geracao de Texto
- Opcao 1: usar `compare_generate.py` na raiz:
  - `python compare_generate.py --only v3 --prompt "Seu prompt" --length 400 --temperature 0.8`
- Opcao 2: implementar uma rotina de geracao similar a v1/v2 com top-k/nucleus a partir dos mapeamentos.

## Dicas
- `--max_textos` e `--epocas` controlam tempo/qualidade; aumente gradualmente.
- Se faltar memoria: reduza `--batch_size` e/ou `--tamanho_lstm`.
