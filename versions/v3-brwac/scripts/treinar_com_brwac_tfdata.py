import argparse
import os
import random
import tempfile
import unicodedata
import re
import time
import json
import platform

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from datasets import load_dataset

# Permitir importar módulos do raiz do repo quando executado via caminho
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_generico_v2 import LLMSimplesGenericoV2


def preparar_texto(
    texto: str,
    lowercase: bool = True,
    end_marker: str = "<END>",
    end_inline_sep: str = "\n",
) -> str:
    """Limpeza mais robusta para BrWaC.
    - Normaliza Unicode (NFKC) e remove caracteres de controle
    - Trata <END> como separador IN-LINE dentro da mesma linha (não quebra documento)
    - Consolida espaços mantendo quebras de linha
    - Lowercase opcional
    """
    # Unicode normalize
    texto = unicodedata.normalize("NFKC", texto)
    # Remover controles (exceto \n, \t)
    texto = "".join(ch for ch in texto if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    # <END> separa campos na MESMA linha -> substituir por separador inline (ex.: espaço)
    if end_marker:
        # normalizar ocorrências com espaços ao redor
        texto = re.sub(r"\s*" + re.escape(end_marker) + r"\s*", end_inline_sep, texto)
    # Normalizar quebras
    texto = texto.replace("\r", "")

    def _squash_spaces(line: str) -> str:
        while "  " in line:
            line = line.replace("  ", " ")
        return line.strip()

    linhas = [l for l in ( _squash_spaces(l) for l in texto.split("\n") ) if l != ""]
    texto = "\n".join(linhas)
    if lowercase:
        texto = texto.lower()
    return texto


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinar LLM (char) com BrWaC usando tf.data")
    # Dados/limpeza
    parser.add_argument("--max_textos", type=int, default=50000, help="Número máximo de textos de treino")
    parser.add_argument("--min_len", type=int, default=50, help="Comprimento mínimo por documento")
    parser.add_argument("--seed", type=int, default=42, help="Seed de reprodução")
    parser.add_argument("--no_lowercase", action="store_true", help="Não converter para minúsculas")
    parser.add_argument("--end_inline_sep", type=str, choices=["space", "newline"], default="newline", help="Substituir token <END> por espaco ('space') ou quebra ('newline')")
    # Modelo/treino
    parser.add_argument("--epocas", type=int, default=10, help="Épocas de treino")
    parser.add_argument("--tamanho_sequencia", type=int, default=160, help="Comprimento da janela")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--tamanho_lstm", type=int, default=256, help="Unidades LSTM")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimensão do embedding")
    parser.add_argument("--stride", type=int, default=3, help="Passo entre janelas")
    parser.add_argument("--shuffle_buffer", type=int, default=10000, help="Buffer de shuffle")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout LSTM")
    parser.add_argument("--recurrent_dropout", type=float, default=0.0, help="Recurrent dropout LSTM")
    parser.add_argument("--clipnorm", type=float, default=1.0, help="Clip norm (0 desabilita)")
    parser.add_argument("--validacao_split", type=float, default=0.1, help="Proporção doc-level validação")
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
        help="Arquivo JSON para registrar detalhes do treino (se vazio, gera em versions/v3-brwac/logs/)",
    )

    args = parser.parse_args()

    # Seeds
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Paths de log
    ts = time.strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join("versions", "v3-brwac", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_json_path = args.log_json_saida or os.path.join(logs_dir, f"train_{ts}.json")
    csv_log_path = os.path.join(logs_dir, f"history_{ts}.csv")
    tb_log_dir = os.path.join(logs_dir, f"tb_{ts}")

    print("Carregando dataset BrWaC (split 'train')...")
    dataset = load_dataset("nlpufg/brwac")
    ds_train_full = dataset["train"].shuffle(seed=args.seed)
    print(f"Dataset carregado com {len(ds_train_full)} exemplos")

    # Split por documento
    if args.validacao_split and 0.0 < args.validacao_split < 1.0:
        split = ds_train_full.train_test_split(test_size=args.validacao_split, seed=args.seed)
        ds_train = split["train"]
        ds_val = split["test"]
        print(f"Split doc-level -> train: {len(ds_train)}, val: {len(ds_val)}")
    else:
        ds_train = ds_train_full
        ds_val = None

    # Amostragem limitada
    max_train = min(args.max_textos, len(ds_train))
    max_val = min(int(args.max_textos * args.validacao_split), len(ds_val)) if ds_val else 0
    print(f"Preparando até {max_train} textos de treino e {max_val} de validação...")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as ftrain:
        caminho_texto_train = ftrain.name
        count_t = 0
        for exemplo in ds_train:
            if count_t >= max_train:
                break
            texto = exemplo["text"]
            sep = " " if args.end_inline_sep == "space" else "\n"
            texto_p = preparar_texto(texto, lowercase=(not args.no_lowercase), end_inline_sep=sep)
            if len(texto_p) >= args.min_len:
                ftrain.write(texto_p + "\n\n")
                count_t += 1
                if count_t % 5000 == 0:
                    print(f"Treino: processados {count_t} textos...")
        print(f"Treino: processados {count_t} textos no total")

    caminho_texto_val = None
    count_v = 0
    if ds_val is not None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as fval:
            caminho_texto_val = fval.name
            for exemplo in ds_val:
                if count_v >= max_val:
                    break
                texto = exemplo["text"]
                sep = " " if args.end_inline_sep == "space" else "\n"
                texto_p = preparar_texto(texto, lowercase=(not args.no_lowercase), end_inline_sep=sep)
                if len(texto_p) >= args.min_len:
                    fval.write(texto_p + "\n\n")
                    count_v += 1
                    if count_v % 2000 == 0:
                        print(f"Val: processados {count_v} textos...")
            print(f"Val: processados {count_v} textos no total")

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
        )

        print("Iniciando treinamento...")
        monitor = "val_loss" if caminho_texto_val else "loss"
        cb = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=args.modelo_saida,
                save_best_only=True,
                monitor=monitor,
                mode="min",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, min_lr=5e-5),
            tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=5, restore_best_weights=True),
            tf.keras.callbacks.CSVLogger(csv_log_path),
            tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir),
        ]

        started_at = time.time()
        llm.treinar(
            caminho_texto=caminho_texto_train,
            nome_arquivo_modelo=args.modelo_saida,
            nome_arquivo_maps=args.mapeamentos_saida,
            callbacks=cb,
            caminho_texto_validacao=caminho_texto_val,
        )
        ended_at = time.time()
        elapsed = ended_at - started_at

        # Montar e salvar log JSON com ambiente, config e stats
        def _env_info() -> dict:
            gpus = tf.config.list_physical_devices("GPU")
            gpu_names = []
            for g in gpus:
                try:
                    det = tf.config.experimental.get_device_details(g)
                    name = det.get("device_name") or str(g)
                except Exception:
                    name = str(g)
                gpu_names.append(name)
            return {
                "python": platform.python_version(),
                "tensorflow": tf.__version__,
                "os": platform.platform(),
                "cpu": platform.processor(),
                "cpu_count": os.cpu_count(),
                "gpu_devices": gpu_names,
            }

        stats = llm.stats()
        payload = {
            "timestamps": {
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at)),
                "ended_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ended_at)),
                "elapsed_sec": elapsed,
            },
            "env": _env_info(),
            "dataset": {"train_docs": count_t, "val_docs": count_v, "min_len": args.min_len},
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
                "end_inline_sep": "newline" if args.end_inline_sep != "space" else "space",
                "seed": args.seed,
            },
            "model": {"params": stats.get("params"), "vocab_size": stats.get("vocab_size")},
            "windows": {
                "train_windows": stats.get("train_windows"),
                "val_windows": stats.get("val_windows"),
                "train_batches": stats.get("train_batches"),
                "val_batches": stats.get("val_batches"),
            },
            "artifacts": {
                "modelo": args.modelo_saida,
                "mapeamentos": args.mapeamentos_saida,
                "csv_history": csv_log_path,
                "tensorboard_log_dir": tb_log_dir,
            },
        }
        try:
            payload["git"] = {"commit": os.popen("git rev-parse HEAD").read().strip()}
        except Exception:
            pass
        try:
            with open(log_json_path, "w", encoding="utf-8") as jf:
                json.dump(payload, jf, ensure_ascii=False, indent=2)
            print(f"Log de treino salvo em {log_json_path}")
        except Exception as e:
            print(f"[WARN] Falha ao salvar log JSON: {e}")
        print(f"Treinamento concluído! Modelo salvo em '{args.modelo_saida}'")
    finally:
        # Limpar arquivos temporários
        for p in [caminho_texto_train, caminho_texto_val]:
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
