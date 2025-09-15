import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


@dataclass
class ModelSpec:
    name: str
    model_path: str
    mapping_path: str


def load_mappings(mapping_path: str) -> Tuple[Dict[str, int], Dict[int, str], int | None]:
    with open(mapping_path, "rb") as f:
        mp = pickle.load(f)
    # Normalize keys across versions
    if "char_para_int" in mp and "int_para_char" in mp:
        c2i = mp["char_para_int"]
        i2c = mp["int_para_char"]
    elif "char_to_id" in mp and "id_to_char" in mp:
        c2i = mp["char_to_id"]
        i2c = mp["id_to_char"]
    else:
        c2i = mp.get("char_to_idx") or mp.get("char_to_int") or mp.get("char2idx")
        i2c = mp.get("idx_to_char") or mp.get("int_to_char") or mp.get("idx2char")
        if c2i is None or i2c is None:
            raise ValueError(f"Formato de mapeamentos desconhecido em {mapping_path}")
    # Convert keys to consistent types if needed
    try:
        i2c = {int(k): v for k, v in i2c.items()}
    except Exception:
        pass
    # Optional: sequence length saved by some versions
    seq_len = mp.get("tamanho_sequencia")
    return c2i, i2c, seq_len


def sample_with_temperature(probs: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 1e-9)
    if temperature <= 0:
        # Greedy
        return int(np.argmax(probs))
    logits = np.log(probs) / temperature
    logits -= logits.max()
    exp = np.exp(logits)
    p = exp / exp.sum()
    return int(rng.choice(len(p), p=p))


def infer_shapes(model: tf.keras.Model) -> Tuple[str, int | None, int | None]:
    """Return (input_type, seq_len, vocab_size_for_onehot)"""
    ishape = model.input_shape
    # Embedding-based models typically have input shape (None, seq_len)
    if len(ishape) == 2:
        return "embedding", ishape[1], None
    # One-hot models: (None, seq_len, vocab_size)
    if len(ishape) == 3:
        return "onehot", ishape[1], ishape[2]
    raise ValueError(f"Forma de entrada inesperada: {ishape}")


def prepare_seed_indices(seed: str, c2i: Dict[str, int], seq_len: int) -> List[int]:
    # Replace unknown chars with space or first vocab char
    rep = " " if " " in c2i else next(iter(c2i.keys()))
    filtered = [ch if ch in c2i else rep for ch in seed]
    idxs = [c2i[ch] for ch in filtered]
    if len(idxs) < seq_len:
        pad_idx = c2i.get(" ", idxs[0] if idxs else 0)
        idxs = [pad_idx] * (seq_len - len(idxs)) + idxs
    else:
        idxs = idxs[-seq_len:]
    return idxs


def generate_text(model: tf.keras.Model, c2i: Dict[str, int], i2c: Dict[int, str], seed: str, length: int, temperature: float, rng: np.random.Generator) -> str:
    input_type, seq_len, onehot_vocab = infer_shapes(model)
    if seq_len is None:
        # Fallback window if model accepts variável length
        seq_len = 100
    vocab_size = len(i2c)
    seq = prepare_seed_indices(seed, c2i, seq_len)
    out_chars = []
    for _ in range(length):
        if input_type == "embedding":
            x = np.array(seq, dtype=np.int32)[None, :]
        else:
            x = np.zeros((1, seq_len, vocab_size if onehot_vocab is None else onehot_vocab), dtype=np.float32)
            for t, idx in enumerate(seq):
                if idx < x.shape[2]:
                    x[0, t, idx] = 1.0
        preds = model.predict(x, verbose=0)[0]
        # If output is a sequence, take last timestep
        if preds.ndim > 1:
            preds = preds[-1]
        next_idx = sample_with_temperature(preds, temperature, rng)
        out_chars.append(i2c.get(next_idx, "?"))
        seq = seq[1:] + [next_idx]
    return "".join(out_chars)


def run_compare(prompt: str, length: int, temperature: float, specs: List[ModelSpec], seed: int) -> None:
    rng = np.random.default_rng(seed)
    for spec in specs:
        if not os.path.exists(spec.model_path):
            print(f"[WARN] {spec.name}: modelo ausente: {spec.model_path}")
            continue
        if not os.path.exists(spec.mapping_path):
            print(f"[WARN] {spec.name}: mapeamentos ausentes: {spec.mapping_path}")
            continue
        print(f"\n=== {spec.name} ===")
        c2i, i2c, _ = load_mappings(spec.mapping_path)
        model = tf.keras.models.load_model(spec.model_path)
        gen = generate_text(model, c2i, i2c, prompt, length, temperature, rng)
        print(f"Prompt: {prompt}")
        print(f"Output: {gen}")


def main():
    parser = argparse.ArgumentParser(description="Comparar geração de texto entre v1, v2, v3 com o mesmo prompt")
    parser.add_argument("--prompt", type=str, default="O Brasil ")
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", type=str, default="", help="Comma: v1,v2,v3 para filtrar")
    args = parser.parse_args()

    all_specs = {
        "v1": ModelSpec("v1-char-rnn", "versions/v1-char-rnn/models/modelo_char_rnn.keras", "versions/v1-char-rnn/mappings/mapeamentos.pkl"),
        "v2": ModelSpec("v2-char-lm", "versions/v2-char-lm/models/modelo_char_lm_v2.keras", "versions/v2-char-lm/mappings/mapeamentos_v2.pkl"),
        "v3": ModelSpec("v3-brwac", "versions/v3-brwac/models/modelo_brwac.keras", "versions/v3-brwac/mappings/mapeamentos_brwac.pkl"),
    }
    keys = [k.strip() for k in args.only.split(",") if k.strip()] if args.only else ["v1", "v2", "v3"]
    specs = [all_specs[k] for k in keys if k in all_specs]
    run_compare(args.prompt, args.length, args.temperature, specs, args.seed)


if __name__ == "__main__":
    main()
