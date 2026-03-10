"""
Laboratório P1 - Construindo o Transformer Encoder "From Scratch"
Disciplina: Tópicos em Inteligência Artificial – 2026.1
Professor: Prof. Dimmy Magalhães
Instituição: iCEV - Instituto de Ensino Superior
"""

import numpy as np
import pandas as pd

# Semente para reprodutibilidade
np.random.seed(42)

# ============================================================
# PASSO 1: PREPARAÇÃO DOS DADOS
# ============================================================

# 1. Vocabulário como DataFrame
vocabulario = {
    "palavra": ["o", "banco", "bloqueou", "cartao", "meu", "de", "credito", "a", "conta", "foi"],
    "id":      [  0,      1,         2,        3,     4,    5,         6,    7,       8,     9 ]
}
df_vocab = pd.DataFrame(vocabulario)
print("=== Vocabulário ===")
print(df_vocab.to_string(index=False))

# Mapeamento palavra -> id
word2id = dict(zip(df_vocab["palavra"], df_vocab["id"]))

# 2. Frase de entrada e conversão para IDs
frase = ["o", "banco", "bloqueou", "o", "cartao"]
token_ids = [word2id[w] for w in frase]
print(f"\nFrase de entrada : {frase}")
print(f"IDs dos tokens   : {token_ids}")

# 3. Parâmetros do modelo
vocab_size     = len(df_vocab)
d_model        = 64        # Simplificado (paper usa 512)
d_ff           = d_model * 4   # 256 — paper usa d_ff = 2048
d_k            = d_model       # dimensão de Q, K, V por cabeça (single-head aqui)
N_LAYERS       = 6
EPSILON        = 1e-6

# Tabela de embeddings: shape (vocab_size, d_model)
embedding_table = np.random.randn(vocab_size, d_model)

# 4. Tensor de entrada X: shape (BatchSize=1, SeqLen, d_model)
X_input = embedding_table[token_ids]           # (SeqLen, d_model)
X_input = X_input[np.newaxis, :, :]           # (1, SeqLen, d_model)

print(f"\nFormato do tensor de entrada X: {X_input.shape}  → (Batch, Tokens, d_model)")


# ============================================================
# PASSO 2: MOTOR MATEMÁTICO
# ============================================================

# ----------------------------------------------------------
# 2.1  Softmax própria (estável numericamente)
# ----------------------------------------------------------
def softmax(x):
    """Softmax ao longo do último eixo."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # subtrai max p/ estabilidade
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# ----------------------------------------------------------
# 2.2  Scaled Dot-Product Attention
# ----------------------------------------------------------
class ScaledDotProductAttention:
    def __init__(self, d_model, d_k):
        # Pesos projetores Q, K, V — inicializados aleatoriamente
        self.WQ = np.random.randn(d_model, d_k) * np.sqrt(2.0 / d_model)
        self.WK = np.random.randn(d_model, d_k) * np.sqrt(2.0 / d_model)
        self.WV = np.random.randn(d_model, d_k) * np.sqrt(2.0 / d_model)
        self.d_k = d_k

    def forward(self, X):
        """
        X: (Batch, SeqLen, d_model)
        Retorna: (Batch, SeqLen, d_k)
        """
        Q = X @ self.WQ   # (B, S, d_k)
        K = X @ self.WK   # (B, S, d_k)
        V = X @ self.WV   # (B, S, d_k)

        # Produto escalar QK^T / sqrt(d_k)
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(self.d_k)  # (B, S, S)

        # Softmax
        attn_weights = softmax(scores)  # (B, S, S)

        # Soma ponderada dos valores
        output = attn_weights @ V       # (B, S, d_k)
        return output


# ----------------------------------------------------------
# 2.3  Layer Normalization
# ----------------------------------------------------------
def layer_norm(X, epsilon=EPSILON):
    """
    Normaliza ao longo do último eixo (features).
    X: (..., d_model)
    """
    mean = np.mean(X, axis=-1, keepdims=True)
    var  = np.var(X,  axis=-1, keepdims=True)
    return (X - mean) / np.sqrt(var + epsilon)


# ----------------------------------------------------------
# 2.4  Feed-Forward Network
# ----------------------------------------------------------
class FeedForwardNetwork:
    def __init__(self, d_model, d_ff):
        # Pesos e biases das duas camadas densas
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros((1, 1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros((1, 1, d_model))

    def forward(self, X):
        """
        FFN(x) = max(0, xW1 + b1)W2 + b2
        X: (Batch, SeqLen, d_model)
        """
        hidden = np.maximum(0, X @ self.W1 + self.b1)   # ReLU  (B, S, d_ff)
        output = hidden @ self.W2 + self.b2              # (B, S, d_model)
        return output


# ----------------------------------------------------------
# 2.5  Bloco Encoder completo (uma camada)
# ----------------------------------------------------------
class EncoderLayer:
    def __init__(self, d_model, d_k, d_ff):
        self.attention = ScaledDotProductAttention(d_model, d_k)
        self.ffn       = FeedForwardNetwork(d_model, d_ff)

    def forward(self, X):
        """
        Fluxo exato conforme o enunciado:
        1. X_att   = SelfAttention(X)
        2. X_norm1 = LayerNorm(X + X_att)
        3. X_ffn   = FFN(X_norm1)
        4. X_out   = LayerNorm(X_norm1 + X_ffn)
        """
        X_att   = self.attention.forward(X)          # (B, S, d_k==d_model)
        X_norm1 = layer_norm(X + X_att)              # conexão residual + LN
        X_ffn   = self.ffn.forward(X_norm1)          # FFN
        X_out   = layer_norm(X_norm1 + X_ffn)        # conexão residual + LN
        return X_out


# ============================================================
# PASSO 3: EMPILHANDO N=6 CAMADAS
# ============================================================

# Cria as 6 camadas independentes (cada uma com seus próprios pesos)
encoder_layers = [EncoderLayer(d_model, d_k, d_ff) for _ in range(N_LAYERS)]

X = X_input.copy()

print(f"\n=== Passagem pelas {N_LAYERS} camadas do Encoder ===")
print(f"Formato de entrada na Camada 1: {X.shape}")

for i, layer in enumerate(encoder_layers):
    X = layer.forward(X)
    print(f"  → Saída da Camada {i+1}: {X.shape}")

# ----------------------------------------------------------
# VALIDAÇÃO DE SANIDADE
# ----------------------------------------------------------
print("\n=== Validação de Sanidade ===")
assert X.shape == X_input.shape, (
    f"ERRO: formato esperado {X_input.shape}, obtido {X.shape}"
)
print(f"✓ Dimensões preservadas: {X.shape}  (Batch={X.shape[0]}, Tokens={X.shape[1]}, d_model={X.shape[2]})")
print("✓ Tensor Z (representação contextualizada) gerado com sucesso!")

print("\n=== Primeiros valores do Vetor Z (token 'o') ===")
print(X[0, 0, :8].round(6))  # primeiros 8 valores do primeiro token
