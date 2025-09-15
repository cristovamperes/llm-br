# v1 — LLM simples por caracteres

Guia didático para estudantes: esta versão implementa um modelo gerador de texto por caracteres usando Keras (LSTM). É ótima para entender conceitos fundamentais: tokenização por caracteres, janelas deslizantes, one-hot/embedding, arquitetura sequencial e geração autoregressiva.

## Estrutura
- Notebook principal: `notebooks/llm.ipynb`
- Modelos salvos: `models/`
- Mapeamentos (char↔idx): `mappings/`
- Observação: artefatos binários (.keras, .pkl) são ignorados no Git.

## Pré‑requisitos
- Python 3.11+ (recomendado 3.12) e um ambiente virtual ativo.
- Instale as dependências: `pip install -r requirements.txt` na raiz do repositório.

## Passo a passo (rápido)
1) Abra o notebook `notebooks/llm.ipynb` no Jupyter.
2) Execute as células sequencialmente. A primeira execução treina o modelo com o corpus configurado no notebook.
3) Após o treino, use a célula de “geração de texto” e teste prompts curtos.

## Conceitos‑chave explicados
- Tokenização por caracteres: cada caractere do corpus recebe um índice; o mapeamento é salvo em `mappings/`.
- Janelas de sequência: cria pares (entrada, próximo caractere) de comprimento fixo para treinar a próxima predição.
- Arquitetura: LSTM com camada densa final para distribuição sobre o vocabulário de caracteres.
- Temperatura: parâmetro que controla aleatoriedade na amostragem de caracteres durante a geração.

## Dicas de treino
- Aumente `épocas` para melhor qualidade (cuidado com overfitting).
- Aumente `tamanho_sequencia` se o corpus for grande; reduz para treinos rápidos.
- Salve checkpoints: o notebook pode incluir callbacks (EarlyStopping/ModelCheckpoint) do Keras.

## Reprodutibilidade
- Registre semente aleatória (NumPy/TensorFlow) para resultados semelhantes entre execuções.
- Salve o `.keras` e o `.pkl` correspondentes no mesmo diretório da versão.

## Geração de texto
- Forneça um texto inicial (“seed”) com caracteres presentes no vocabulário.
- Ajuste a “temperatura” e o comprimento de saída para experimentar estilos.

## Solução de problemas
- TensorFlow sem GPU: tudo funciona em CPU, apenas mais lento.
- Erros de memória: reduza `batch_size` e/ou `tamanho_sequencia`.
- Unicode/acentos: garanta leitura com `encoding="utf-8"` no preparo do texto.

## Boas práticas
- Não versione `models/` e `mappings/` (já ignorados no `.gitignore`).
- Documente hiperparâmetros utilizados em cada treino (útil para o TCC).
