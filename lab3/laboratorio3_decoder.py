import numpy as np

# ==============================================================
# LABORATÓRIO 3 - IMPLEMENTANDO O DECODER
# ==============================================================

print("=" * 60)
print("LABORATÓRIO 3 - IMPLEMENTANDO O DECODER")
print("=" * 60)


# ==============================================================
# TAREFA 1: MÁSCARA CAUSAL (LOOK-AHEAD MASK)
# ==============================================================
print("\n" + "=" * 60)
print("TAREFA 1: MÁSCARA CAUSAL (LOOK-AHEAD MASK)")
print("=" * 60)

def create_causal_mask(seq_len):
    """
    Cria uma máscara causal (look-ahead mask) de dimensão [seq_len, seq_len].
    - Triangular inferior (incluindo diagonal): 0
    - Triangular superior: -infinito
    """
    # Começa com uma matriz de zeros
    mask = np.zeros((seq_len, seq_len))
    # Preenche a parte triangular superior (acima da diagonal) com -inf
    mask = np.where(np.triu(np.ones((seq_len, seq_len)), k=1) == 1, -np.inf, mask)
    return mask

def softmax(x, axis=-1):
    """Softmax numericamente estável."""
    # Subtrai o máximo para estabilidade numérica (ignora -inf corretamente)
    x_max = np.where(np.isneginf(x), -np.inf, x)
    shifted = x - np.nanmax(np.where(np.isneginf(x_max), -1e9, x_max), axis=axis, keepdims=True)
    exp_x = np.exp(np.clip(shifted, -500, 500))
    # Zera posições que eram -inf
    exp_x = np.where(np.isneginf(x), 0.0, exp_x)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-9)

# Parâmetros
seq_len = 5
d_k = 64

# Máscara causal
M = create_causal_mask(seq_len)
print(f"\nMáscara Causal M ({seq_len}x{seq_len}):")
print(M)

# Matrizes fictícias Q e K
np.random.seed(42)
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)

# Cálculo: scores = (Q @ K^T) / sqrt(d_k) + M
scores = (Q @ K.T) / np.sqrt(d_k) + M

# Aplicar Softmax
attention_weights = softmax(scores, axis=-1)

print(f"\nMatriz de Atenção após Softmax (com máscara causal):")
print(np.round(attention_weights, 4))

print("\n✅ PROVA REAL - Verificando que posições futuras têm probabilidade 0.0:")
for i in range(seq_len):
    for j in range(seq_len):
        if j > i:
            assert abs(attention_weights[i, j]) < 1e-6, f"ERRO: posição [{i},{j}] = {attention_weights[i,j]}"
print("   Todas as posições futuras (triangular superior) são estritamente 0.0 ✓")


# ==============================================================
# TAREFA 2: A PONTE ENCODER-DECODER (CROSS-ATTENTION)
# ==============================================================
print("\n" + "=" * 60)
print("TAREFA 2: CROSS-ATTENTION (PONTE ENCODER-DECODER)")
print("=" * 60)

# Dimensões
batch_size  = 1
seq_len_frances = 10   # comprimento da frase do Encoder (francês)
seq_len_ingles  = 4    # comprimento do que o Decoder já gerou (inglês)
d_model = 512
d_k2    = 64           # dimensão de Q e K na atenção

# Tensores fictícios
np.random.seed(0)
encoder_output = np.random.randn(batch_size, seq_len_frances, d_model)  # [1, 10, 512]
decoder_state  = np.random.randn(batch_size, seq_len_ingles,  d_model)  # [1,  4, 512]

print(f"\nEncoder output shape : {encoder_output.shape}  (batch, seq_frances, d_model)")
print(f"Decoder state shape  : {decoder_state.shape}   (batch, seq_ingles,  d_model)")

# Matrizes de projeção arbitrárias (pesos)
W_Q = np.random.randn(d_model, d_k2)   # projeta decoder_state  -> Q
W_K = np.random.randn(d_model, d_k2)   # projeta encoder_output -> K
W_V = np.random.randn(d_model, d_k2)   # projeta encoder_output -> V

def cross_attention(encoder_out, decoder_state):
    """
    Calcula o Cross-Attention entre o Decoder (Query) e o Encoder (Key, Value).
    
    - Q vem do decoder_state
    - K e V vêm do encoder_out
    - SEM máscara causal (o Decoder pode olhar toda a frase do Encoder)
    
    Retorna:
        output: [batch, seq_ingles, d_k]
        weights: [batch, seq_ingles, seq_frances]
    """
    # Projeções
    Q = decoder_state @ W_Q    # [batch, seq_ingles,  d_k]
    K = encoder_out   @ W_K    # [batch, seq_frances, d_k]
    V = encoder_out   @ W_V    # [batch, seq_frances, d_k]

    # Scaled Dot-Product Attention (SEM máscara causal)
    # scores: [batch, seq_ingles, seq_frances]
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k2)

    # Softmax sobre a dimensão das Keys (francês)
    weights = softmax(scores, axis=-1)

    # Contexto: [batch, seq_ingles, d_k]
    output = weights @ V

    return output, weights

context, attn_weights = cross_attention(encoder_output, decoder_state)

print(f"\nSaída do Cross-Attention (contexto) shape : {context.shape}")
print(f"Pesos de atenção shape                    : {attn_weights.shape}")
print(f"\nPesos de atenção (batch 0) - cada linha do Decoder atendendo ao Encoder:")
print(np.round(attn_weights[0], 4))
print(f"\nSoma das linhas (deve ser ~1.0 para validar Softmax):")
print(np.round(attn_weights[0].sum(axis=-1), 6))
print("✅ Cross-Attention calculada com sucesso ✓")


# ==============================================================
# TAREFA 3: LOOP DE INFERÊNCIA AUTO-REGRESSIVO
# ==============================================================
print("\n" + "=" * 60)
print("TAREFA 3: LOOP DE INFERÊNCIA AUTO-REGRESSIVO")
print("=" * 60)

# Vocabulário fictício
VOCAB_SIZE = 10_000

# Índices especiais
TOKEN_START = "<START>"
TOKEN_EOS   = "<EOS>"

# Vocabulário fictício: índice -> token
# Índice 0 reservado para <EOS>
np.random.seed(7)
vocab = {i: f"palavra_{i}" for i in range(1, VOCAB_SIZE)}
vocab[0] = TOKEN_EOS

def generate_next_token(current_sequence, encoder_out):
    """
    Mock do Decoder: simula a passagem pelo Decoder e retorna
    um vetor de probabilidades sobre o vocabulário.

    Parâmetros:
        current_sequence (list[str]): tokens gerados até agora.
        encoder_out (np.ndarray)    : saída do Encoder [batch, seq, d_model].

    Retorna:
        probs (np.ndarray): vetor de probabilidades [VOCAB_SIZE].
    """
    # Simula logits aleatórios (representando a projeção linear final do Decoder)
    logits = np.random.randn(VOCAB_SIZE)

    # A cada 5 passos força <EOS> para o exemplo não rodar indefinidamente
    # (em um modelo real, isso emerge do treinamento)
    if len(current_sequence) >= 5:
        logits[0] = 1e6   # garante argmax == 0 == <EOS>

    # Softmax -> distribuição de probabilidades
    probs = softmax(logits, axis=0)
    return probs

# Estado inicial
current_sequence = [TOKEN_START]
max_steps = 20  # segurança contra loop infinito

print(f"\nSequência inicial : {current_sequence}")
print(f"Vocabulário fictício: {VOCAB_SIZE} tokens  |  <EOS> = índice 0\n")

step = 0
while step < max_steps:
    step += 1

    # Gera distribuição de probabilidades para o próximo token
    probs = generate_next_token(current_sequence, encoder_output)

    # Seleciona token com maior probabilidade (argmax)
    next_token_idx = int(np.argmax(probs))
    next_token     = vocab[next_token_idx]

    print(f"Passo {step:02d} | argmax={next_token_idx:5d} | prob={probs[next_token_idx]:.6f} | token gerado: '{next_token}'")

    # Adiciona o novo token à sequência
    current_sequence.append(next_token)

    # Verifica condição de parada
    if next_token == TOKEN_EOS:
        print(f"\n🛑 Token <EOS> detectado! Geração encerrada.")
        break

print(f"\n✅ Frase final gerada:")
print("   " + " ".join(current_sequence))
print("\n" + "=" * 60)
print("FIM DO LABORATÓRIO 3")
print("=" * 60)
