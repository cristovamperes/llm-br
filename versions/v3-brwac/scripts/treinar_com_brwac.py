import argparse
import os
import tempfile

import tensorflow as tf
from datasets import load_dataset
from llm_generico import LLMSimplesGenerico


def preparar_texto(texto: str) -> str:
    """Limpeza simples: normaliza espaços e remove quebras de linha extras."""
    texto = texto.replace("\n", " ").replace("\r", " ")
    texto = texto.lower()
    while "  " in texto:
        texto = texto.replace("  ", " ")
    return texto.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinar LLM simples por caracteres com BrWaC")
    parser.add_argument("--epocas", type=int, default=10, help="Número de épocas de treinamento")
    parser.add_argument("--tamanho_sequencia", type=int, default=100, help="Tamanho da sequência de entrada")
    parser.add_argument("--batch_size", type=int, default=128, help="Tamanho do batch")
    parser.add_argument("--tamanho_lstm", type=int, default=256, help="Tamanho da camada LSTM")
    parser.add_argument(
        "--modelo_saida",
        type=str,
        default="versions/v3-brwac/models/modelo_brwac.keras",
        help="Caminho do arquivo de modelo (.keras)",
    )
    parser.add_argument(
        "--mapeamentos_saida",
        type=str,
        default="versions/v3-brwac/mappings/mapeamentos_brwac.pkl",
        help="Caminho do arquivo de mapeamentos (.pkl)",
    )
    parser.add_argument("--max_textos", type=int, default=50000, help="Número máximo de textos para treinamento")
    parser.add_argument("--validacao_split", type=float, default=0.1, help="Proporção para validação")

    args = parser.parse_args()

    print("Carregando dataset BrWaC (split 'train')...")
    dataset = load_dataset("nlpufg/brwac")
    print(f"Dataset carregado com {len(dataset['train'])} exemplos")

    print(f"Preparando amostra de até {args.max_textos} textos...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        caminho_texto = f.name
        count = 0
        for exemplo in dataset["train"]:
            if count >= args.max_textos:
                break
            texto = exemplo["text"]
            texto_p = preparar_texto(texto)
            if len(texto_p) > 50:
                f.write(texto_p + " ")
                count += 1
                if count % 5000 == 0:
                    print(f"Processados {count} textos...")
        print(f"Processados {count} textos no total")

    try:
        llm = LLMSimplesGenerico(
            tamanho_sequencia=args.tamanho_sequencia,
            tamanho_lstm=args.tamanho_lstm,
            epocas_treino=args.epocas,
            batch_size=args.batch_size,
            validacao_split=args.validacao_split,
        )

        print("Iniciando treinamento...")
        # Callbacks alinhados com notebooks: monitor em val_loss se houver validação
        monitor = "val_loss" if args.validacao_split and args.validacao_split > 0 else "loss"
        cb = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=args.modelo_saida,
                save_best_only=True,
                monitor=monitor,
                mode="min",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, min_lr=5e-5),
            tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=5, restore_best_weights=True),
        ]

        llm.treinar(
            caminho_texto=caminho_texto,
            nome_arquivo_modelo=args.modelo_saida,
            nome_arquivo_maps=args.mapeamentos_saida,
            callbacks=cb,
        )
        print(f"Treinamento concluído! Modelo salvo em '{args.modelo_saida}'")
    finally:
        # Limpar arquivo temporário
        try:
            os.unlink(caminho_texto)
        except OSError:
            pass


if __name__ == "__main__":
    main()
