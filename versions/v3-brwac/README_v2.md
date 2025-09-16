# v3 - BrWaC (tf.data) — Script V2

Treinamento por caracteres com BrWaC usando tf.data (janelas com stride) e limpeza de texto mais robusta (tratando marcadores `<END>` como separadores de documento).

## Scripts
- Treino (novo): `versions/v3-brwac/scripts/treinar_com_brwac_tfdata.py`
- Comparação com top-k/top-p (novo): `compare_generate_v2.py`

## Principais melhorias
- Split de validação por documento (sem `validation_split` por janelas).
- `tf.data` para gerar janelas sob demanda, com `--stride`, `--shuffle_buffer` e `prefetch`.
- Hiperparâmetros extras: `--embedding_dim`, `--dropout`, `--recurrent_dropout`, `--clipnorm`.
- Limpeza: Unicode NFKC, remoção de controles, preserva quebras e trata `<END>` como separador.
- Seeds fixos para reprodutibilidade.

## Exemplo — Treinamento
```
python versions/v3-brwac/scripts/treinar_com_brwac_tfdata.py \
  --epocas 10 --max_textos 10000 \
  --tamanho_sequencia 160 --tamanho_lstm 256 --embedding_dim 128 \
  --batch_size 256 --stride 3 --dropout 0.2 --clipnorm 1.0 \
  --validacao_split 0.1
```

Artefatos gerados:
- Modelo: `versions/v3-brwac/models/modelo_brwac.keras`
- Mapeamentos: `versions/v3-brwac/mappings/mapeamentos_brwac.pkl`

## Exemplo — Geração com top‑k/top‑p
```
python compare_generate_v2.py \
  --only v3 --prompt "Seu prompt" --length 400 \
  --temperature 0.8 --top_k 40 --top_p 0.95
```

## Dicas
- Aumente `--stride` para reduzir custo (ex.: 5–8) se o corpus for grande.
- Se overfitting: aumente `--dropout`, use `--tamanho_lstm 512` apenas com GPU.
- Se memória for limite: reduza `--batch_size` e/ou `--tamanho_lstm`.

