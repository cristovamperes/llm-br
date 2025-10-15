# -*- coding: utf-8 -*-
"""Pipeline de treinamento para o LM de caracteres com BrWaC usando tf.data."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from datasets import load_dataset

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_generico_v2 import LLMSimplesGenericoV2
from preprocessamento import preparar_texto, resolve_end_inline_sep


def _processar_documentos(
    dataset_split,
    limite: int,
    *,
    min_len: int,
    clean_kwargs: Dict[str, object],
) -> Tuple[int, str]:
    count = 0
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        caminho = tmp.name

    with open(caminho, "w", encoding="utf-8") as handle:
        for exemplo in dataset_split:
            if count >= limite:
                break
            texto = exemplo["text"]
            texto_p = preparar_texto(texto, **clean_kwargs)
            if len(texto_p) >= min_len:
                handle.write(texto_p + "\n\n")
                count += 1
                if count % 5000 == 0:
                    print(f"Processados {count} textos...")

    return count, caminho


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinar LLM (char) com BrWaC usando tf.data")
    # Dados/limpeza
    parser.add_argument("--max_textos", type=int, default=50000, help="Número máximo de textos de treino")
    parser.add_argument("--min_len", type=int, default=50, help="Comprimento mínimo por documento")
    parser.add_argument("--min_line_chars", type=int, default=40, help="Comprimento mínimo por linha após limpeza")
    parser.add_argument("--min_alpha_ratio", type=float, default=0.4, help="Proporção mínima de caracteres alfabéticos por linha")
    parser.add_argument("--keep_numbers", action="store_true", help="Não normalizar números (mantém dígitos originais)")
    parser.add_argument("--keep_upper_metadata", action="store_true", help="Manter linhas em caixa alta (metadados)")
    parser.add_argument("--seed", type=int, default=42, help="Seed de reprodução")
    parser.add_argument("--no_lowercase", action="store_true", help="Não converter para minúsculas")
    parser.add_argument("--end_inline_sep", type=str, choices=["space", "newline"], default="newline", help="Substituir token <END> por espaço ('space') ou quebra ('newline')")
    parser.add_argument("--unk_token", type=str, default="~", help="Caractere para representar tokens desconhecidos (1 caractere)")
    parser.add_argument("--batch_log_freq", type=int, default=100, help="Frequência de logging por batch (0 desativa)")
    # Modelo/treino
    parser.add_argument("--epocas", type=int, default=10, help="Épocas de treino")
    parser.add_argument("--tamanho_sequencia", type=int, default=160, help="Comprimento da janela")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--tamanho_lstm", type=int, default=256, help="Unidades LSTM")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimensão do embedding")
    parser.add_argument("--stride", type=int, default=3, help="Passo entre janelas")
    parser.add_argument("--shuffle_buffer", type=int, default=100000, help="Buffer de shuffle para o tf.data")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout LSTM")
    parser.add_argument("--recurrent_dropout", type=float, default=0.0, help="Recurrent dropout LSTM")
    parser.add_argument("--clipnorm", type=float, default=1.0, help="Clip norm (0 desabilita)")
    parser.add_argument("--validacao_split", type=float, default=0.1, help="Proporção doc-level para validação")
    # Saída
    parser.add_argument(
        "--modelo_saida",
        type=str,
        default="versions/v3-brwac/models/modelo_brwac.keras",
        help="Arquivo de modelo (.keras)",
    )
    parser.add_argument(
        "--mapeamentos_saida",
        type=str,
        default="versions/v3-brwac/mappings/mapeamentos_brwac.pkl",
        help="Arquivo de mapeamentos (.pkl)",
    )
    parser.add_argument(
        "--log_json_saida",
        type=str,
        default="",
        help="Arquivo JSON com detalhes do treino (se vazio, cria em versions/v3-brwac/logs/)",
    )

    args = parser.parse_args()

    if not args.unk_token or len(args.unk_token) != 1:
        parser.error("--unk_token deve ser um único caractere")

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    logs_dir = _HERE.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_json_path = Path(args.log_json_saida) if args.log_json_saida else logs_dir / f"train_{ts}.json"
    csv_log_path = logs_dir / f"history_{ts}.csv"
    tb_log_dir = logs_dir / f"tb_{ts}"
    batch_log_path = logs_dir / f"batches_{ts}.log"

    print("Carregando dataset BrWaC (split 'train')...")
    dataset = load_dataset("nlpufg/brwac")
    ds_train_full = dataset["train"].shuffle(seed=args.seed)
    print(f"Dataset carregado com {len(ds_train_full)} exemplos")

    if args.validacao_split and 0.0 < args.validacao_split < 1.0:
        split = ds_train_full.train_test_split(test_size=args.validacao_split, seed=args.seed)
        ds_train = split["train"]
        ds_val = split["test"]
        print(f"Split doc-level -> train: {len(ds_train)}, val: {len(ds_val)}")
    else:
        ds_train = ds_train_full
        ds_val = None

    max_train = min(args.max_textos, len(ds_train))
    max_val = min(int(args.max_textos * args.validacao_split), len(ds_val)) if ds_val else 0
    print(f"Preparando até {max_train} textos de treino e {max_val} de validação...")

    end_inline_sep = resolve_end_inline_sep(args.end_inline_sep)
    clean_kwargs = dict(
        lowercase=not args.no_lowercase,
        end_inline_sep=end_inline_sep,
        min_line_chars=args.min_line_chars,
        min_alpha_ratio=args.min_alpha_ratio,
        normalize_numbers=not args.keep_numbers,
        drop_uppercase_metadata=not args.keep_upper_metadata,
    )

    count_t, caminho_texto_train = _processar_documentos(
        ds_train,
        max_train,
        min_len=args.min_len,
        clean_kwargs=clean_kwargs,
    )
    print(f"Treino: processados {count_t} textos no total")

    count_v = 0
    caminho_texto_val = None
    if ds_val is not None and max_val > 0:
        count_v, caminho_texto_val = _processar_documentos(
            ds_val,
            max_val,
            min_len=args.min_len,
            clean_kwargs=clean_kwargs,
        )
        print(f"Validação: processados {count_v} textos no total")

    started_at = time.time()
    try:
        llm = LLMSimplesGenericoV2(
            tamanho_sequencia=args.tamanho_sequencia,
            tamanho_lstm=args.tamanho_lstm,
            embedding_dim=args.embedding_dim,
            epocas_treino=args.epocas,
            batch_size=args.batch_size,
            learning_rate=1e-3,
            stride=args.stride,
            shuffle_buffer=args.shuffle_buffer,
            dropout=args.dropout,
            recurrent_dropout=args.recurrent_dropout,
            clipnorm=args.clipnorm if args.clipnorm and args.clipnorm > 0 else None,
            unk_token=args.unk_token,
        )

        print("Iniciando treinamento...")
        monitor = "val_loss" if caminho_texto_val else "loss"

        def _batch_logger_factory(path: Path, freq: int) -> tf.keras.callbacks.Callback:
            if freq <= 0:
                return tf.keras.callbacks.LambdaCallback()

            path.parent.mkdir(parents=True, exist_ok=True)

            def _on_batch_end(batch, logs=None):
                if logs is None or batch % freq != 0:
                    return
                loss = logs.get("loss")
                msg = f"batch {batch} - loss {loss:.4f}" if isinstance(loss, float) else f"batch {batch} - loss {loss}"
                print(msg, flush=True)
                try:
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(f"{int(time.time())},{msg}\n")
                except Exception:
                    pass

            return tf.keras.callbacks.LambdaCallback(on_batch_end=_on_batch_end)

        callbacks: list[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=args.modelo_saida,
                save_best_only=True,
                monitor=monitor,
                mode="min",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, min_lr=5e-5),
            tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=5, restore_best_weights=True),
            tf.keras.callbacks.CSVLogger(csv_log_path),
            tf.keras.callbacks.TensorBoard(log_dir=str(tb_log_dir), update_freq=100),
        ]
        if args.batch_log_freq > 0:
            callbacks.append(_batch_logger_factory(batch_log_path, freq=args.batch_log_freq))

        llm.treinar(
            caminho_texto=caminho_texto_train,
            nome_arquivo_modelo=args.modelo_saida,
            nome_arquivo_maps=args.mapeamentos_saida,
            callbacks=callbacks,
            caminho_texto_validacao=caminho_texto_val,
        )
        ended_at = time.time()
        elapsed = ended_at - started_at
        print(f"Treinamento concluído! Modelo salvo em '{args.modelo_saida}'")

        stats = llm.stats()
        best_metric = None
        for cb in callbacks:
            if isinstance(cb, tf.keras.callbacks.ModelCheckpoint):
                best_metric = getattr(cb, "best", None)
                break

        gpu_devices = []
        for device in tf.config.list_physical_devices("GPU"):
            try:
                details = tf.config.experimental.get_device_details(device)
                if isinstance(details, dict):
                    gpu_devices.append(details.get("device_name", str(device)))
                else:
                    gpu_devices.append(str(device))
            except Exception:
                gpu_devices.append(str(device))

        payload = {
            "timestamps": {
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at)),
                "ended_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ended_at)),
                "elapsed_sec": elapsed,
            },
            "env": {
                "python": platform.python_version(),
                "tensorflow": tf.__version__,
                "os": platform.platform(),
                "cpu": platform.processor(),
                "cpu_count": os.cpu_count(),
                "gpu_devices": gpu_devices,
            },
            "dataset": {
                "train_docs": count_t,
                "val_docs": count_v,
                "min_len": args.min_len,
            },
            "cleaning": {
                "lowercase": not args.no_lowercase,
                "end_inline_sep": args.end_inline_sep,
                "min_line_chars": args.min_line_chars,
                "min_alpha_ratio": args.min_alpha_ratio,
                "normalize_numbers": not args.keep_numbers,
                "drop_uppercase_metadata": not args.keep_upper_metadata,
            },
            "train_config": {
                "epocas": args.epocas,
                "batch_size": args.batch_size,
                "tamanho_sequencia": args.tamanho_sequencia,
                "tamanho_lstm": args.tamanho_lstm,
                "embedding_dim": args.embedding_dim,
                "stride": args.stride,
                "shuffle_buffer": args.shuffle_buffer,
                "dropout": args.dropout,
                "recurrent_dropout": args.recurrent_dropout,
                "clipnorm": args.clipnorm,
                "batch_log_freq": args.batch_log_freq,
                "seed": args.seed,
                "unk_token": args.unk_token,
            },
            "model": {
                "params": stats.get("params"),
                "vocab_size": stats.get("vocab_size"),
                "unk_token": stats.get("unk_token"),
            },
            "windows": {
                "train_windows": stats.get("train_windows"),
                "val_windows": stats.get("val_windows"),
                "train_batches": stats.get("train_batches"),
                "val_batches": stats.get("val_batches"),
            },
            "artifacts": {
                "modelo": args.modelo_saida,
                "mapeamentos": args.mapeamentos_saida,
                "csv_history": str(csv_log_path),
                "tensorboard_log_dir": str(tb_log_dir),
                "batch_log": str(batch_log_path) if args.batch_log_freq > 0 else "",
            },
            "best_monitor_value": best_metric,
        }
        try:
            payload["git"] = {"commit": os.popen("git rev-parse HEAD").read().strip()}
        except Exception:
            pass
        try:
            with open(log_json_path, "w", encoding="utf-8") as jf:
                json.dump(payload, jf, ensure_ascii=False, indent=2)
            print(f"Log de treino salvo em {log_json_path}")
        except Exception as exc:
            print(f"[WARN] Falha ao salvar log JSON: {exc}")

        monitor_value = best_metric if best_metric is not None else "n/a"
        print(
            f"[Resumo] duração: {elapsed / 60:.2f} min | docs treino: {count_t} | "
            f"docs val: {count_v} | melhor {monitor}: {monitor_value}"
        )
    finally:
        for caminho in filter(None, [caminho_texto_train, caminho_texto_val]):
            try:
                os.unlink(caminho)
            except OSError:
                pass


if __name__ == "__main__":
    main()
