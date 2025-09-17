import tensorflow as tf
from tokenizers import Tokenizer

# --- 1. Carregar Tokenizador e Hiperparâmetros ---
print("Carregando tokenizador...")
tokenizer = Tokenizer.from_file("brwac_wordpiece.json")

# Hiperparâmetros para um modelo PEQUENO
VOCAB_SIZE = tokenizer.get_vocab_size()
SEQ_LENGTH = 128      # Comprimento da sequência
EMBED_DIM = 256       # Dimensão da embedding
NUM_HEADS = 4         # Número de cabeças de atenção
FF_DIM = 512          # Dimensão da camada feed-forward
NUM_BLOCKS = 4        # Número de blocos Transformer empilhados
BATCH_SIZE = 32
CORPUS_FILE = "brwac_corpus.txt"

# --- 2. Preparar o Dataset com tf.data ---
print("Preparando dataset...")
# Carregar e tokenizar todo o corpus (para corpus grandes, usar .from_generator)
raw_text = open(CORPUS_FILE, "r", encoding="utf-8").read()
token_ids = tokenizer.encode(raw_text).ids

# Criar um dataset a partir dos IDs
dataset = tf.data.Dataset.from_tensor_slices(token_ids)

# Criar janelas de sequência
sequences = dataset.window(SEQ_LENGTH + 1, shift=1, drop_remainder=True)
sequences = sequences.flat_map(lambda window: window.batch(SEQ_LENGTH + 1))

# Criar pares de (entrada, alvo)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Embaralhar e criar lotes (batches)
BUFFER_SIZE = 10000
dataset = (
    dataset.shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

print(f"Dataset pronto com {len(list(dataset))} batches.")

# --- 3. Construir o Modelo ---
print("Construindo o modelo Transformer...")

# (Cole as classes PositionalEmbedding e TransformerBlock aqui)
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, max_len=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(max_len, embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs, use_causal_mask=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Modelo final
inputs = tf.keras.layers.Input(shape=(SEQ_LENGTH,))
x = PositionalEmbedding(VOCAB_SIZE, EMBED_DIM)(inputs)
for _ in range(NUM_BLOCKS):
    x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)(x)
outputs = tf.keras.layers.Dense(VOCAB_SIZE)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# --- 4. Compilar e Treinar ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

print("\nIniciando treinamento...")
history = model.fit(dataset, epochs=5)

# Salve o modelo final
model.save("transformer_brwac.keras")
print("Modelo treinado e salvo em transformer_brwac.keras")