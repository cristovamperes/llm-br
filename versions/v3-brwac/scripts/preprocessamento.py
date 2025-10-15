# -*- coding: utf-8 -*-
"""Utilidades de limpeza e normalização para o corpus BrWaC."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

ALLOWED_EXTRA_CHARS: set[str] = {
    ",",
    ".",
    ";",
    ":",
    "!",
    "?",
    '"',
    "'",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "-",
    "–",
    "—",
    "_",
    "/",
    "\\",
    "%",
    "&",
    "*",
    "+",
    "=",
    "@",
    "#",
    "$",
    "°",
    "º",
    "ª",
    "|",
    "«",
    "»",
    "`",
    "´",
    "’",
    "‘",
    "“",
    "”",
    "~",
    "^",
    "…",
    "·",
    "•",
    "<",
    ">",
    "€",
}


def _clean_char(ch: str, allowed_extra: set[str]) -> str:
    if ch == "\n":
        return "\n"
    if ch in {"\t", "\r"}:
        return " "
    cat = unicodedata.category(ch)
    if cat.startswith(("L", "N")):
        return ch
    if ch in allowed_extra:
        return ch
    return " "


def resolve_end_inline_sep(option: str) -> str:
    if option == "space":
        return " "
    if option == "newline":
        return "\n"
    raise ValueError(f"Opção inválida para end_inline_sep: {option}")


def preparar_texto(
    texto: str,
    *,
    lowercase: bool = True,
    end_marker: str = "<END>",
    end_inline_sep: str = "\n",
    min_line_chars: int = 40,
    min_alpha_ratio: float = 0.4,
    normalize_numbers: bool = True,
    drop_uppercase_metadata: bool = True,
) -> str:
    allowed_extra = set(ALLOWED_EXTRA_CHARS)

    texto = unicodedata.normalize("NFKC", texto)
    texto = texto.replace("\ufeff", "")
    texto = texto.replace("\u200b", "")

    if end_marker:
        texto = re.sub(r"\s*" + re.escape(end_marker) + r"\s*", end_inline_sep, texto)

    texto = re.sub(r"http[s]?://\S+", " ", texto)
    texto = re.sub(r"\S+@\S+", " ", texto)
    texto = texto.replace("\r", "")
    texto = texto.replace("\t", " ")

    filtered = (_clean_char(ch, allowed_extra) for ch in texto)
    texto = "".join(filtered)
    texto = re.sub(r"[ ]{2,}", " ", texto)
    texto = re.sub(r"\n{2,}", "\n", texto)

    if normalize_numbers:
        texto = re.sub(r"\d{2,}", "0", texto)

    linhas_processadas: list[str] = []
    for raw_line in texto.split("\n"):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        linha_norm = " ".join(raw_line.split())
        linha_proc = linha_norm.lower() if lowercase else linha_norm
        if len(linha_proc) < min_line_chars:
            continue
        alpha = sum(ch.isalpha() for ch in linha_proc)
        if len(linha_proc) == 0 or alpha / len(linha_proc) < min_alpha_ratio:
            continue
        if drop_uppercase_metadata and linha_norm.isupper() and len(linha_norm) < 80:
            continue
        linhas_processadas.append(linha_proc)

    return "\n".join(linhas_processadas)
