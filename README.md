# LLM Brasileiro — Projeto do TCC em Ciência de Dados

Este repositório reúne implementações de Modelos de Linguagem (LLMs) com diferentes níveis de complexidade, desenvolvidas como parte do TCC em Ciência de Dados. A organização está separada por versões, para facilitar a referência no texto do TCC e o reuso do código.

## Estrutura do repositório

- `versions/v1-char-rnn`: primeira versão (RNN/LSTM de caracteres) com notebook demonstrativo.
- `versions/v2-char-lm`: segunda versão, evolução do LM de caracteres, também em notebook.
- `versions/v3-brwac`: terceira versão com script de treino usando o corpus BrWaC.
- `data/`: arquivos de exemplo pequenos usados nos testes (ex.: `exemplo_texto.txt`).

Os artefatos de treino (modelos `.keras` e mapeamentos `.pkl`) ficam dentro de cada versão, em `models/` e `mappings/`. Por padrão, esses arquivos são ignorados no Git para manter o repositório leve; consulte abaixo como reproduzir o treino.

## Como usar

### v1 — LLM simples (caracteres)
- Notebook: `versions/v1-char-rnn/notebooks/llm.ipynb`
- Objetivo: demonstrar os conceitos fundamentais de um LM por caracteres.
- Requisitos: Python 3.12+ e dependências do notebook (TensorFlow/Keras e NumPy — veja notas de ambiente).

### v2 — LLM de caracteres (evolução)
- Notebook: `versions/v2-char-lm/notebooks/llm_v2.ipynb`
- Objetivo: melhorias de arquitetura/treino em relação ao v1.

### v3 — Treino com BrWaC
- Script (baseline): `versions/v3-brwac/scripts/treinar_com_brwac.py`
- Script recomendado (tf.data + limpeza + split por documento): `versions/v3-brwac/scripts/treinar_com_brwac_tfdata.py`
- Exemplo recomendado:
  - `python versions/v3-brwac/scripts/treinar_com_brwac_tfdata.py --epocas 10 --max_textos 10000 --tamanho_sequencia 160 --tamanho_lstm 256 --embedding_dim 128 --batch_size 256 --stride 3 --clipnorm 1.0 --validacao_split 0.1 --end_inline_sep newline`
- Geração comparativa com top‑k/top‑p: `compare_generate_v2.py`
  - `python compare_generate_v2.py --only v3 --prompt "Seu prompt" --length 400 --temperature 0.8 --top_k 40 --top_p 0.95`
- Artefatos gerados: `versions/v3-brwac/models/modelo_brwac.keras` e `versions/v3-brwac/mappings/mapeamentos_brwac.pkl`.

## Notas de ambiente
- Recomenda-se criar um ambiente virtual (ex.: `python -m venv .venv`) e instalar as dependências necessárias (TensorFlow/Keras, NumPy, etc.).
- Os diretórios `.venv/` e equivalentes estão no `.gitignore` para não irem ao repositório.

### GPU (NVIDIA) — Dicas de desempenho
- Verifique a GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- Mixed precision (recomendado):
  - Linux/macOS: `export TF_ENABLE_AUTO_MIXED_PRECISION=1`
  - Windows PowerShell: `$env:TF_ENABLE_AUTO_MIXED_PRECISION="1"`
- XLA JIT (opcional; pode ajudar pouco em LSTM):
  - Linux/macOS: `export TF_XLA_FLAGS="--tf_xla_auto_jit=2"`
  - Windows PowerShell: `$env:TF_XLA_FLAGS="--tf_xla_auto_jit=2"`
- Perfis sugeridos (RTX 3080):
  - Throughput equilibrado:
    - `python versions/v3-brwac/scripts/treinar_com_brwac_tfdata.py --epocas 8 --max_textos 20000 --tamanho_sequencia 160 --tamanho_lstm 512 --embedding_dim 256 --batch_size 512 --stride 2 --dropout 0.0 --recurrent_dropout 0.0 --clipnorm 0 --shuffle_buffer 100000 --validacao_split 0.1 --end_inline_sep newline`
  - Mais steps/capacidade (se couber em VRAM):
    - `python versions/v3-brwac/scripts/treinar_com_brwac_tfdata.py --epocas 6 --max_textos 50000 --tamanho_sequencia 160 --tamanho_lstm 512 --embedding_dim 256 --batch_size 768 --stride 1 --dropout 0.0 --recurrent_dropout 0.0 --clipnorm 0 --shuffle_buffer 200000 --validacao_split 0.1 --end_inline_sep newline`
  - Se ocorrer OOM, reduza primeiro `--batch_size`, depois `--tamanho_lstm`.

### Célula de setup nos notebooks (v1/v2)
- Ambos os notebooks começam com uma célula “SETUP_ARTIFACT_PATHS” que define caminhos padrão para artefatos:
  - `MODELO_OUT`: arquivo `.keras` dentro de `versions/<versão>/models/`
  - `MAPEAMENTOS_OUT`: arquivo `.pkl` dentro de `versions/<versão>/mappings/`
- As variáveis de uso ao longo do notebook já apontam para esses caminhos padronizados:
  - `NOME_ARQUIVO_MODELO = str(MODELO_OUT)`
  - `NOME_ARQUIVO_MAPS = str(MAPEAMENTOS_OUT)`

## Dados e artefatos
- Arquivos grandes de modelo (`.keras`) e mapeamentos (`.pkl`) são mantidos fora do Git por padrão.
- Opcionalmente, use Git LFS se desejar versionar os artefatos treinados.
