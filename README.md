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
- Script: `versions/v3-brwac/scripts/treinar_com_brwac.py`
- Exemplo de execução:
  - `python versions/v3-brwac/scripts/treinar_com_brwac.py --epocas 5 --max_textos 10000`
- Artefatos gerados: `versions/v3-brwac/models/modelo_brwac.keras` e `versions/v3-brwac/mappings/mapeamentos_brwac.pkl`.

## Notas de ambiente
- Recomenda-se criar um ambiente virtual (ex.: `python -m venv .venv`) e instalar as dependências necessárias (TensorFlow/Keras, NumPy, etc.).
- Os diretórios `.venv/` e equivalentes estão no `.gitignore` para não irem ao repositório.

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

## Autor
[Seu nome]

## Licença
[A definir]

