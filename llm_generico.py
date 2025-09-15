import os
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


@dataclass
class TreinoConfig:
    tamanho_sequencia: int = 100
    tamanho_lstm: int = 256
    epocas_treino: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    validacao_split: float = 0.1


class LLMSimplesGenerico:
    """
    LLM simples por caracteres usando Keras (Embedding + LSTM + Softmax).

    Fluxo:
    - Carrega texto único a partir de um arquivo.
    - Cria vocabulário e mapeamentos char↔idx.
    - Gera pares (X: janelas de índices, y: próximo caractere índice).
    - Treina um modelo para prever o próximo caractere.
    - Salva modelo (.keras) e mapeamentos (.pkl).
    """

    def __init__(
        self,
        tamanho_sequencia: int = 100,
        tamanho_lstm: int = 256,
        epocas_treino: int = 10,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        validacao_split: float = 0.1,
    ) -> None:
        self.cfg = TreinoConfig(
            tamanho_sequencia=tamanho_sequencia,
            tamanho_lstm=tamanho_lstm,
            epocas_treino=epocas_treino,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validacao_split=validacao_split,
        )
        self.model: tf.keras.Model | None = None
        self.char_to_idx: Dict[str, int] | None = None
        self.idx_to_char: Dict[int, str] | None = None

    # ------------------------------
    # Público
    # ------------------------------
    def treinar(
        self,
        caminho_texto: str,
        nome_arquivo_modelo: str,
        nome_arquivo_maps: str,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ) -> None:
        texto = self._ler_texto(caminho_texto)
        self.char_to_idx, self.idx_to_char = self._criar_mapeamentos(texto)
        X, y = self._criar_janelas(texto, self.char_to_idx, self.cfg.tamanho_sequencia)

        vocab_size = len(self.char_to_idx)
        self.model = self._construir_modelo(vocab_size)

        self.model.fit(
            X,
            y,
            batch_size=self.cfg.batch_size,
            epochs=self.cfg.epocas_treino,
            validation_split=self.cfg.validacao_split,
            callbacks=callbacks if callbacks else None,
            verbose=2,
        )

        # Garantir diretórios de saída
        os.makedirs(os.path.dirname(nome_arquivo_modelo) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(nome_arquivo_maps) or ".", exist_ok=True)

        # Salvar artefatos
        self.model.save(nome_arquivo_modelo)
        self._salvar_mapeamentos(nome_arquivo_maps)

    # ------------------------------
    # Internos
    # ------------------------------
    @staticmethod
    def _ler_texto(caminho: str) -> str:
        with open(caminho, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _criar_mapeamentos(texto: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        vocab = sorted(list(set(texto)))
        char_to_idx = {ch: i for i, ch in enumerate(vocab)}
        idx_to_char = {i: ch for ch, i in char_to_idx.items()}
        return char_to_idx, idx_to_char

    @staticmethod
    def _criar_janelas(
        texto: str, char_to_idx: Dict[str, int], tamanho_sequencia: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Converter texto para índices
        seq = np.array([char_to_idx[ch] for ch in texto if ch in char_to_idx], dtype=np.int32)
        if len(seq) <= tamanho_sequencia:
            raise ValueError("Texto muito curto para o tamanho de sequência escolhido.")

        X_list = []
        y_list = []
        for i in range(0, len(seq) - tamanho_sequencia):
            X_list.append(seq[i : i + tamanho_sequencia])
            y_list.append(seq[i + tamanho_sequencia])
        X = np.stack(X_list)
        y = np.array(y_list, dtype=np.int32)
        return X, y

    def _construir_modelo(self, vocab_size: int) -> tf.keras.Model:
        model = models.Sequential([
            layers.Input(shape=(self.cfg.tamanho_sequencia,)),
            layers.Embedding(input_dim=vocab_size, output_dim=64),
            layers.LSTM(self.cfg.tamanho_lstm),
            layers.Dense(vocab_size, activation="softmax"),
        ])
        opt = optimizers.Adam(learning_rate=self.cfg.learning_rate)
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def _salvar_mapeamentos(self, caminho_maps: str) -> None:
        assert self.char_to_idx is not None and self.idx_to_char is not None
        payload = {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "tamanho_sequencia": self.cfg.tamanho_sequencia,
        }
        with open(caminho_maps, "wb") as f:
            pickle.dump(payload, f)
