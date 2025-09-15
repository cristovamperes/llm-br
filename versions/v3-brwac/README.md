# v3 - Treino com BrWaC

Nesta versao treinamos o LM por caracteres usando o corpus BrWaC (colecao de portugues do Brasil) via Hugging Face Datasets.

## Estrutura
- Script de treino: `scripts/treinar_com_brwac.py`
- Modelos salvos: `models/` (ex.: `modelo_brwac.keras`)
- Mapeamentos: `mappings/` (ex.: `mapeamentos_brwac.pkl`)
- Observacao: artefatos binarios (.keras, .pkl) sao ignorados no Git.

## Pre-requisitos
- Python 3.11+ (recomendado 3.12).
- Instale dependencias: `pip install -r requirements.txt` na raiz.
- Conexao para baixar o dataset `nlpufg/brwac` (Hugging Face Datasets faz o download na primeira execucao).

## Execucao
Exemplo rapido (amostra de 10k textos). Por padrao, os artefatos ja sao salvos nas pastas desta versao:
```
python versions/v3-brwac/scripts/treinar_com_brwac.py \
  --epocas 5 --max_textos 10000 \
  --tamanho_sequencia 100 --tamanho_lstm 256 --batch_size 128
```
Se desejar, voce pode trocar os caminhos de saida com `--modelo_saida` e `--mapeamentos_saida`.

## O que o script faz
1) Baixa/carrega `nlpufg/brwac`.
2) Pre-processa os textos (limpeza simples e concatenacao em um arquivo temporario).
3) Cria e treina o modelo via `LLMSimplesGenerico` (definido em `llm_generico.py`).
4) Salva artefatos em `models/` e `mappings/`.

## Dicas e cuidados
- Reduza `--max_textos` e `--epocas` para testes rapidos.
- Se faltar memoria, reduza `--batch_size` e/ou `--tamanho_lstm`.
- Primeira execucao baixa o dataset e pode demorar; as proximas usam o cache local do Datasets.

## Solucao de problemas
- ImportError `llm_generico`: confirme que `llm_generico.py` esta na raiz do repositorio.
- Latencia alta/sem GPU: TensorFlow roda em CPU; considere reduzir hiperparametros.
- Acentuacao: usamos `utf-8` no processo; mantenha esse padrao em leituras/gravacoes.

