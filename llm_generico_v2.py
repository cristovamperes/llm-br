# -*- coding: utf-8 -*-
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


@dataclass
class TreinoConfigV2:
    tamanho_sequencia: int = 160
    tamanho_lstm: int = 256
    embedding_dim: int = 128
    epocas_treino: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    stride: int = 3
    shuffle_buffer: int = 10000
    dropout: float = 0.2
    recurrent_dropout: float = 0.0
    clipnorm: float | None = 1.0
    unk_token: str = "~"


class LLMSimplesGenericoV2:
    """LLM simples por caracteres usando Keras (Embedding + LSTM + Softmax).

    Usa tf.data para gerar janelas deslizantes de forma eficiente e oferece
    utilidades para registrar estatísticas do treinamento.
    """

    def __init__(
        self,
        tamanho_sequencia: int = 160,
        tamanho_lstm: int = 256,
        embedding_dim: int = 128,
        epocas_treino: int = 10,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        stride: int = 3,
        shuffle_buffer: int = 10000,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.0,
        clipnorm: float | None = 1.0,
        unk_token: str = "~",
    ) -> None:
        self.cfg = TreinoConfigV2(
            tamanho_sequencia=tamanho_sequencia,
            tamanho_lstm=tamanho_lstm,
            embedding_dim=embedding_dim,
            epocas_treino=epocas_treino,
            batch_size=batch_size,
            learning_rate=learning_rate,
            stride=stride,
            shuffle_buffer=shuffle_buffer,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            clipnorm=clipnorm,
            unk_token=unk_token,
        )
        self.unk_token = unk_token
        self.model: tf.keras.Model | None = None
        self.char_to_idx: Dict[str, int] | None = None
        self.idx_to_char: Dict[int, str] | None = None
        # Stats
        self.vocab_size: int = 0
        self.train_windows: int = 0
        self.val_windows: int = 0
        self.train_batches: int = 0
        self.val_batches: int = 0

    # ------------------------------
    # Público
    # ------------------------------
    def treinar(
        self,
        caminho_texto: str,
        nome_arquivo_modelo: str,
        nome_arquivo_maps: str,
        callbacks: Optional[list[tf.keras.callbacks.Callback]] = None,
        caminho_texto_validacao: Optional[str] = None,
    ) -> None:
        texto_train = self._ler_texto(caminho_texto)
        self.char_to_idx, self.idx_to_char = self._criar_mapeamentos(texto_train)

        unk_idx = self._unk_idx()
        seq_train = self._texto_para_indices(texto_train, self.char_to_idx, unk_idx=unk_idx)
        ds_train = self._seq_para_dataset(seq_train, treino=True)

        # Stats: vocab/windows/batches
        self.vocab_size = len(self.char_to_idx)
        self.train_windows = max(0, int((len(seq_train) - self.cfg.tamanho_sequencia) // max(1, self.cfg.stride)))
        self.train_batches = int(np.ceil(self.train_windows / max(1, self.cfg.batch_size)))

        ds_val = None
        if caminho_texto_validacao:
            texto_val = self._ler_texto(caminho_texto_validacao)
            seq_val = self._texto_para_indices(texto_val, self.char_to_idx, unk_idx=unk_idx)
            ds_val = self._seq_para_dataset(seq_val, treino=False)
            self.val_windows = max(0, int((len(seq_val) - self.cfg.tamanho_sequencia) // max(1, self.cfg.stride)))
            self.val_batches = int(np.ceil(self.val_windows / max(1, self.cfg.batch_size)))

        self.model = self._construir_modelo(self.vocab_size)

        self.model.fit(
            ds_train,
            epochs=self.cfg.epocas_treino,
            validation_data=ds_val if ds_val is not None else None,
            callbacks=callbacks if callbacks else None,
            verbose=1,
        )

        # Garantir diretórios de saída
        os.makedirs(os.path.dirname(nome_arquivo_modelo) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(nome_arquivo_maps) or ".", exist_ok=True)

        # Salvar artefatos
        self.model.save(nome_arquivo_modelo)
        self._salvar_mapeamentos(nome_arquivo_maps)

    def stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do treino/modelo até o momento."""
        params = int(self.model.count_params()) if self.model is not None else 0
        return {
            "vocab_size": self.vocab_size,
            "tamanho_sequencia": self.cfg.tamanho_sequencia,
            "embedding_dim": self.cfg.embedding_dim,
            "tamanho_lstm": self.cfg.tamanho_lstm,
            "batch_size": self.cfg.batch_size,
            "stride": self.cfg.stride,
            "train_windows": self.train_windows,
            "val_windows": self.val_windows,
            "train_batches": self.train_batches,
            "val_batches": self.val_batches,
            "params": params,
            "unk_token": self.cfg.unk_token,
        }

    # ------------------------------
    # Internos
    # ------------------------------
    @staticmethod
    def _ler_texto(caminho: str) -> str:
        with open(caminho, "r", encoding="utf-8") as f:
            return f.read()

    def _criar_mapeamentos(self, texto: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        vocab = sorted(set(texto))
        if self.cfg.unk_token not in vocab:
            vocab.append(self.cfg.unk_token)
        vocab = sorted(vocab)
        char_to_idx = {ch: i for i, ch in enumerate(vocab)}
        idx_to_char = {i: ch for ch, i in char_to_idx.items()}
        return char_to_idx, idx_to_char

    @staticmethod
    def _texto_para_indices(texto: str, char_to_idx: Dict[str, int], unk_idx: int | None = None) -> np.ndarray:
        seq: list[int] = []
        for ch in texto:
            if ch in char_to_idx:
                seq.append(char_to_idx[ch])
            elif unk_idx is not None:
                seq.append(unk_idx)
        return np.array(seq, dtype=np.int32)

    def _seq_para_dataset(self, seq: np.ndarray, treino: bool) -> tf.data.Dataset:
        if len(seq) <= self.cfg.tamanho_sequencia:
            raise ValueError("Texto muito curto para o tamanho de sequência escolhido.")
        ds = tf.data.Dataset.from_tensor_slices(seq)
        ds = ds.window(self.cfg.tamanho_sequencia + 1, shift=self.cfg.stride, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.cfg.tamanho_sequencia + 1))
        ds = ds.map(lambda w: (w[:-1], w[-1]), num_parallel_calls=tf.data.AUTOTUNE)
        if treino:
            ds = ds.shuffle(self.cfg.shuffle_buffer, reshuffle_each_iteration=True)
        ds = ds.batch(self.cfg.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def _construir_modelo(self, vocab_size: int) -> tf.keras.Model:
        model = models.Sequential([
            layers.Input(shape=(self.cfg.tamanho_sequencia,)),
            layers.Embedding(input_dim=vocab_size, output_dim=self.cfg.embedding_dim),
            layers.LSTM(self.cfg.tamanho_lstm, dropout=self.cfg.dropout, recurrent_dropout=self.cfg.recurrent_dropout),
            layers.Dense(vocab_size, activation="softmax"),
        ])
        if self.cfg.clipnorm is not None and self.cfg.clipnorm > 0:
            opt = optimizers.Adam(learning_rate=self.cfg.learning_rate, clipnorm=self.cfg.clipnorm)
        else:
            opt = optimizers.Adam(learning_rate=self.cfg.learning_rate)
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def _unk_idx(self) -> int:
        # Fallback para caractere desconhecido
        assert self.char_to_idx is not None
        if self.cfg.unk_token in self.char_to_idx:
            return self.char_to_idx[self.cfg.unk_token]
        if " " in self.char_to_idx:
            return self.char_to_idx[" "]
        return next(iter(self.char_to_idx.values()))

    def _salvar_mapeamentos(self, caminho_maps: str) -> None:
        assert self.char_to_idx is not None and self.idx_to_char is not None
        payload = {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "tamanho_sequencia": self.cfg.tamanho_sequencia,
            "embedding_dim": self.cfg.embedding_dim,
            "tamanho_lstm": self.cfg.tamanho_lstm,
            "stride": self.cfg.stride,
            "vocab_size": self.vocab_size,
            "unk_token": self.cfg.unk_token,
        }
        with open(caminho_maps, "wb") as f:
            pickle.dump(payload, f)

