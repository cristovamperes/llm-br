# -*- coding: utf-8 -*-
"""Utilitário para analisar o vocabulário após a limpeza do BrWaC."""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path

from datasets import load_dataset

_HERE = Path(__file__).resolve().parent

from preprocessamento import preparar_texto, resolve_end_inline_sep  # noqa: F401


def _repr_char(ch: str) -> str:
    if ch == " ":
        return "<space>"
    if ch == "\n":
        return "\\n"
    if ch == "\t":
        return "\\t"
    if ch == "\r":
        return "\\r"
    if ch == "\ufeff":
        return "\\ufeff"
    if ch == "\u200b":
        return "\\u200b"
    codepoint = ord(ch)
    if codepoint < 32 or codepoint == 127:
        return f"\\x{codepoint:02x}"
    return ch


def _make_clean_kwargs(args: argparse.Namespace) -> dict:
    end_inline_sep = resolve_end_inline_sep(args.end_inline_sep)
    return dict(
        lowercase=not args.no_lowercase,
        end_inline_sep=end_inline_sep,
        min_line_chars=args.min_line_chars,
        min_alpha_ratio=args.min_alpha_ratio,
        normalize_numbers=not args.keep_numbers,
        drop_uppercase_metadata=not args.keep_upper_metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerar estatísticas de vocabulário após limpeza do BrWaC")
    parser.add_argument("--max_textos", type=int, default=10000, help="Número máximo de textos analisados")
    parser.add_argument("--min_len", type=int, default=50, help="Comprimento mínimo do documento após limpeza")
    parser.add_argument("--min_line_chars", type=int, default=40, help="Comprimento mínimo por linha após limpeza")
    parser.add_argument("--min_alpha_ratio", type=float, default=0.4, help="Proporção mínima de caracteres alfabéticos por linha")
    parser.add_argument("--keep_numbers", action="store_true", help="Não normalizar números (mantém dígitos originais)")
    parser.add_argument("--keep_upper_metadata", action="store_true", help="Manter linhas em caixa alta (metadados)")
    parser.add_argument("--seed", type=int, default=42, help="Seed de reprodução")
    parser.add_argument("--no_lowercase", action="store_true", help="Não converter para minúsculas")
    parser.add_argument("--end_inline_sep", type=str, choices=["space", "newline"], default="newline", help="Separador usado para substituir <END>")
    parser.add_argument("--top_n", type=int, default=50, help="Número de caracteres mais frequentes mostrados no console")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Arquivo CSV de saída (por padrão salva em versions/v3-brwac/logs/vocab_stats_<timestamp>.csv)",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    print("Carregando dataset BrWaC (split 'train')...")
    dataset = load_dataset("nlpufg/brwac")
    ds = dataset["train"].shuffle(seed=args.seed)
    print(f"Dataset carregado com {len(ds)} exemplos")

    clean_kwargs = _make_clean_kwargs(args)
    counter: Counter[str] = Counter()

    processados = 0
    documentos = 0
    for exemplo in ds:
        if documentos >= args.max_textos:
            break
        texto = preparar_texto(exemplo["text"], **clean_kwargs)
        if len(texto) < args.min_len:
            continue
        counter.update(texto)
        documentos += 1
        processados += len(texto)
        if documentos % 2000 == 0:
            print(f"Documentos válidos: {documentos} | caracteres acumulados: {processados}")

    if documentos == 0:
        print("Nenhum documento atendeu aos critérios informados.")
        return

    total_chars = sum(counter.values())
    unique_chars = len(counter)
    ascii_chars = sum(count for ch, count in counter.items() if ord(ch) < 128)

    print(f"\nDocumentos analisados: {documentos}")
    print(f"Total de caracteres: {total_chars}")
    print(f"Vocabulário único: {unique_chars}")
    print(f"Fração ASCII: {ascii_chars / total_chars:.4f}")

    print(f"\nTop {args.top_n} caracteres:")
    for ch, count in counter.most_common(args.top_n):
        frac = count / total_chars
        print(f"{_repr_char(ch):>8} | count={count:8d} | frac={frac:.4%} | cat={unicodedata.category(ch)}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    logs_dir = _HERE.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else logs_dir / f"vocab_stats_{ts}.csv"

    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["symbol", "codepoint", "category", "count", "fraction"])
        for ch, count in counter.most_common():
            frac = count / total_chars
            writer.writerow(
                [
                    _repr_char(ch),
                    f"U+{ord(ch):04X}",
                    unicodedata.category(ch),
                    count,
                    f"{frac:.8f}",
                ]
            )

    print(f"\nCSV salvo em: {output_path}")


if __name__ == "__main__":
    main()
