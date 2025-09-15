# v2 — LLM de caracteres (evolução)

Esta versão aprofunda o LM por caracteres com melhorias de arquitetura/treino em relação ao v1. É indicada para comparar escolhas de hiperparâmetros e callbacks.

## Estrutura
- Notebook principal: `notebooks/llm_v2.ipynb`
- Modelos salvos: `models/`
- Mapeamentos (char↔idx): `mappings/`
- Observação: artefatos binários (.keras, .pkl) são ignorados no Git.

## Pré‑requisitos
- Python 3.11+ (recomendado 3.12).
- Dependências: `pip install -r requirements.txt` na raiz.

## O que muda em relação ao v1
- Possível uso de camadas adicionais, regularização (Dropout), ou embeddings de caracteres.
- Callbacks do Keras: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` para treinos mais estáveis.
- Ajustes no preparo de dados (sequências, balanceamento, split de validação).

## Passo a passo
1) Abra `notebooks/llm_v2.ipynb` e execute as células.
2) Monitore as curvas de perda/validação; ajuste `épocas`, `batch_size`, `tamanho_lstm` conforme necessário.
3) Gere texto e compare qualitativamente com o v1.

## Dicas
- Se houver overfitting, aumente dropout ou use early stopping.
- Se o treino estiver lento, reduza `tamanho_lstm` e `batch_size`.
- Experimente diferentes estratégias de amostragem durante a geração (top‑k/top‑p se implementadas).

## Solução de problemas
- “No module named tensorflow”: instale o `requirements.txt` e verifique versão do Python compatível.
- Problemas de acentuação: padronize `utf-8` na leitura do corpus.

## Boas práticas
- Registre resultados (tabelas/figuras) para o relatório do TCC.
- Mantenha `models/` e `mappings/` fora do Git (já ignorados).
