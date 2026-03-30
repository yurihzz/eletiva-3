
# Laboratório 6 - P2: Construindo um Tokenizador BPE e
                  # Explorando o WordPiece 




# TAREFA 1: O Motor de Frequências


# Corpus de treinamento inicializado estritamente conforme especificado
vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}


def get_stats(vocab):
    """
    Recebe o dicionário de vocabulário (palavra segmentada -> frequência)
    e retorna um dicionário com a frequência acumulada de todos os pares
    adjacentes de caracteres/símbolos encontrados no corpus.

    Parâmetros:
        vocab (dict): Dicionário {str: int} onde a chave é a palavra
                      com símbolos separados por espaço e o valor é
                      a frequência dela no corpus.

    Retorna:
        pairs (dict): Dicionário {(str, str): int} com a contagem de
                      cada par adjacente.
    """
    pairs = {}
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs


# ------ Validação da Tarefa 1 --------------------------------
print("=" * 60)
print("TAREFA 1: Validação do Motor de Frequências")
print("=" * 60)

stats = get_stats(vocab)
print(f"Frequência do par ('e', 's'): {stats[('e', 's')]}")

# Validação obrigatória: o par ('e','s') deve ter contagem máxima = 9
assert stats[('e', 's')] == 9, "ERRO: A contagem do par ('e','s') deveria ser 9!"
print("✓ Validação OK — par ('e', 's') retornou contagem máxima de 9")
print()



# TAREFA 2: O Loop de Fusão


import re


def merge_vocab(pair, v_in):
    """
    Recebe o par mais frequente e o dicionário de vocabulário atual.
    Substitui todas as ocorrências isoladas desse par pela versão
    unificada (sem espaço entre os dois símbolos) e retorna o
    dicionário atualizado.

    Parâmetros:
        pair  (tuple): Par de símbolos a ser fundido, ex: ('e', 's').
        v_in  (dict):  Dicionário de vocabulário atual.

    Retorna:
        v_out (dict): Dicionário de vocabulário após a fusão.

    Nota sobre uso de IA: a lógica da expressão regular utilizada para
    garantir que apenas o par de tokens isolados seja substituído (e não
    sub-sequências acidentais) foi elaborada com auxílio de IA generativa
    (Claude/Anthropic) e revisada pelo autor — conforme declarado no
    README.md.
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    # (?<!\S) garante que o primeiro símbolo começa num limite de token
    # (?!\S)  garante que o segundo símbolo termina num limite de token
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    replacement = ''.join(pair)
    for word in v_in:
        new_word = pattern.sub(replacement, word)
        v_out[new_word] = v_in[word]
    return v_out


# ------ Loop Principal de Treinamento (K = 5 iterações) ------
print("=" * 60)
print("TAREFA 2: Loop Principal de Treinamento do Tokenizador (K=5)")
print("=" * 60)

num_merges = 5

for i in range(1, num_merges + 1):
    stats = get_stats(vocab)
    best_pair = max(stats, key=lambda p: stats[p])
    vocab = merge_vocab(best_pair, vocab)

    print(f"Iteração {i}:")
    print(f"  Par mais frequente fundido : {best_pair}  (freq={stats[best_pair]})")
    print(f"  Vocabulário após a fusão   :")
    for word, freq in vocab.items():
        print(f"    '{word}': {freq}")
    print()

print("✓ Após 5 iterações observamos tokens morfológicos como 'est</w>'.")
print()



# TAREFA 3: Integração Industrial e WordPiece


print("=" * 60)
print("TAREFA 3: WordPiece com BERT Multilíngue (Hugging Face)")
print("=" * 60)

from transformers import AutoTokenizer

# Instancia o tokenizador multilíngue do BERT (WordPiece)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Frase de teste elaborada para forçar o particionamento morfológico
frase = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."

tokens = tokenizer.tokenize(frase)

print(f"Frase original:")
print(f"  {frase}")
print(f"\nTokens gerados pelo WordPiece (bert-base-multilingual-cased):")
print(f"  {tokens}")
print(f"\nTotal de tokens: {len(tokens)}")
