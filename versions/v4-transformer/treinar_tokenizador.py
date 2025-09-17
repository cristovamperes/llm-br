from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# Crie um tokenizador vazio com o modelo WordPiece (usado pelo BERT)
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Configure o trainer
# vocab_size: O tamanho do seu vocabulário. Para um modelo pequeno, 10k-16k é um bom começo.
# special_tokens: Tokens especiais que o modelo precisa.
trainer = WordPieceTrainer(
    vocab_size=16000,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

# Defina o arquivo do seu corpus
files = ["brwac_corpus.txt"]

# Treine o tokenizador
tokenizer.train(files, trainer)

# Salve o tokenizador treinado em um arquivo. Este arquivo é crucial!
tokenizer.save("brwac_wordpiece.json")

print("Tokenizador treinado e salvo em brwac_wordpiece.json")