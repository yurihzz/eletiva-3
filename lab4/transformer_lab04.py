"""
Laboratório Técnico 04: O Transformer Completo "From Scratch"
Instituto de Ensino Superior ICEV

Implementação completa da arquitetura Encoder-Decoder Transformer
com inferência auto-regressiva para tradução de toy sequence.

Partes geradas/complementadas com IA, revisadas por [Seu Nome]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# TAREFA 1: BLOCOS DE CONSTRUÇÃO (Labs anteriores refatorados)
# =============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention (Lab 01).

    Fórmula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        Q: tensor de queries  (..., seq_len_q, d_k)
        K: tensor de keys     (..., seq_len_k, d_k)
        V: tensor de values   (..., seq_len_k, d_v)
        mask: máscara opcional (..., seq_len_q, seq_len_k)
              - valores True/1 = posições a MASCARAR (recebem -inf)

    Returns:
        output: tensor de saída (..., seq_len_q, d_v)
        attn_weights: pesos de atenção (..., seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)

    # Produto escalar entre Q e K transposto, escalado por sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Aplica máscara causal (zera o futuro com -infinito)
    if mask is not None:
        scores = scores.masked_fill(mask == 1, float('-inf'))

    # Softmax sobre a última dimensão (seq_len_k)
    attn_weights = F.softmax(scores, dim=-1)

    # Produto com V
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


def make_causal_mask(seq_len):
    """
    Cria máscara causal (Lab 02): impede que posição i atenda posição j > i.
    Retorna matriz upper-triangular com 1 nas posições futuras (a mascarar).

    Args:
        seq_len: comprimento da sequência

    Returns:
        mask: tensor (seq_len, seq_len) com 1 no triângulo superior (futuro)
    """
    # torch.triu com diagonal=1 zera a diagonal principal e abaixo dela,
    # deixando 1 apenas no triângulo superior → posições FUTURAS
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask


class PositionwiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network (Lab 03).

    Fórmula: FFN(x) = max(0, xW1 + b1)W2 + b2
    Expansão de d_model → d_ff → d_model com ReLU no meio.

    Args:
        d_model: dimensão do modelo (ex: 512)
        d_ff:    dimensão interna expandida (ex: 2048)
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu    = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.linear2(self.relu(self.linear1(x)))


def add_and_norm(x, sublayer_output, norm):
    """
    Add & Norm (Lab 03).

    Fórmula: Output = LayerNorm(x + Sublayer(x))
    Conexão residual + normalização de camada.

    Args:
        x:               tensor de entrada original (identidade)
        sublayer_output: saída da sub-camada (attention ou FFN)
        norm:            instância de nn.LayerNorm

    Returns:
        tensor normalizado após conexão residual
    """
    return norm(x + sublayer_output)


# =============================================================================
# MULTI-HEAD ATTENTION
# Necessário para projetar Q, K, V em múltiplas "cabeças" de atenção.
# O paper original usa h=8 cabeças para d_model=512 → d_k = d_v = 64.
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.

    MultiHead(Q,K,V) = Concat(head_1,...,head_h) * W_O
    onde head_i = Attention(Q*W_Q_i, K*W_K_i, V*W_V_i)

    Args:
        d_model: dimensão do modelo
        num_heads: número de cabeças de atenção
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads  # dimensão por cabeça

        # Projeções lineares para Q, K, V e saída
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """
        Divide o tensor em múltiplas cabeças.
        (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)

    def combine_heads(self, x):
        """
        Reconecta as cabeças.
        (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: tensores de entrada (batch, seq_len, d_model)
            mask:    máscara opcional (batch, 1, seq_len_q, seq_len_k) ou (seq_len, seq_len)
        """
        # Projeta e divide em cabeças
        Q = self.split_heads(self.W_Q(Q))  # (batch, heads, seq_q, d_k)
        K = self.split_heads(self.W_K(K))  # (batch, heads, seq_k, d_k)
        V = self.split_heads(self.W_V(V))  # (batch, heads, seq_k, d_k)

        # Aplica scaled dot-product attention em cada cabeça
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Reconecta cabeças e projeta saída
        output = self.combine_heads(attn_output)  # (batch, seq_q, d_model)
        return self.W_O(output)


# =============================================================================
# TAREFA 2: ENCODER BLOCK
# =============================================================================

class EncoderBlock(nn.Module):
    """
    Bloco do Encoder.

    Fluxo:
        X → [Self-Attention] → Add & Norm → [FFN] → Add & Norm → Z

    O Encoder contextualiza bidirecionalmente (sem máscara causal),
    produzindo a matriz de memória Z que alimenta o Cross-Attention do Decoder.

    Args:
        d_model:   dimensão do modelo
        num_heads: número de cabeças de atenção
        d_ff:      dimensão interna da FFN
    """
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn            = PositionwiseFFN(d_model, d_ff)
        self.norm1          = nn.LayerNorm(d_model)
        self.norm2          = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        """
        Args:
            x:        tensor de entrada (batch, src_seq_len, d_model)
                      já somado com Positional Encoding
            src_mask: máscara de padding do encoder (opcional)

        Returns:
            z: matriz de memória contextualizada (batch, src_seq_len, d_model)
        """
        # --- Sub-camada 1: Self-Attention (Q=K=V=x, sem máscara causal) ---
        attn_out = self.self_attention(x, x, x, mask=src_mask)
        x = add_and_norm(x, attn_out, self.norm1)   # Add & Norm

        # --- Sub-camada 2: FFN ---
        ffn_out = self.ffn(x)
        x = add_and_norm(x, ffn_out, self.norm2)    # Add & Norm

        return x  # Z: memória rica contextualizada


# =============================================================================
# TAREFA 3: DECODER BLOCK
# =============================================================================

class DecoderBlock(nn.Module):
    """
    Bloco do Decoder.

    Fluxo:
        Y → [Masked Self-Attention] → Add & Norm
          → [Cross-Attention(Q=Y', K=Z, V=Z)] → Add & Norm
          → [FFN] → Add & Norm
          → Linear → Softmax → probabilidades

    O Decoder usa:
      - Masked Self-Attention: impede ver tokens futuros (máscara causal)
      - Cross-Attention: Q vem do decoder, K e V vêm da memória Z do encoder

    Args:
        d_model:   dimensão do modelo
        num_heads: número de cabeças de atenção
        d_ff:      dimensão interna da FFN
    """
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention       = MultiHeadAttention(d_model, num_heads)
        self.ffn                   = PositionwiseFFN(d_model, d_ff)
        self.norm1                 = nn.LayerNorm(d_model)
        self.norm2                 = nn.LayerNorm(d_model)
        self.norm3                 = nn.LayerNorm(d_model)

    def forward(self, y, Z, tgt_mask=None, src_mask=None):
        """
        Args:
            y:        tensor alvo (batch, tgt_seq_len, d_model)
            Z:        memória do encoder (batch, src_seq_len, d_model)
            tgt_mask: máscara causal para o self-attention do decoder
            src_mask: máscara de padding do encoder (opcional)

        Returns:
            y: tensor processado (batch, tgt_seq_len, d_model)
        """
        # --- Sub-camada 1: Masked Self-Attention (Q=K=V=y, com máscara causal) ---
        # A máscara causal impede que a posição i veja posições j > i
        masked_attn_out = self.masked_self_attention(y, y, y, mask=tgt_mask)
        y = add_and_norm(y, masked_attn_out, self.norm1)  # Add & Norm

        # --- Sub-camada 2: Cross-Attention (Q vem do decoder, K e V vêm de Z) ---
        cross_attn_out = self.cross_attention(y, Z, Z, mask=src_mask)
        y = add_and_norm(y, cross_attn_out, self.norm2)   # Add & Norm

        # --- Sub-camada 3: FFN ---
        ffn_out = self.ffn(y)
        y = add_and_norm(y, ffn_out, self.norm3)          # Add & Norm

        return y


# =============================================================================
# POSITIONAL ENCODING
# Injeta informação de posição na sequência de embeddings.
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional Encoding sinusoidal (Vaswani et al., 2017).

    Args:
        d_model:  dimensão do modelo
        max_len:  comprimento máximo de sequência suportado
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # índices pares
        pe[:, 1::2] = torch.cos(position * div_term)  # índices ímpares
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Soma o encoding posicional ao embedding
        return x + self.pe[:, :x.size(1), :]


# =============================================================================
# TRANSFORMER COMPLETO (Encoder-Decoder)
# =============================================================================

class Transformer(nn.Module):
    """
    Transformer Encoder-Decoder completo.

    Arquitetura:
        Encoder: Embedding + PE → N × EncoderBlock → Z (memória)
        Decoder: Embedding + PE → N × DecoderBlock → Linear → Softmax

    Args:
        src_vocab_size: tamanho do vocabulário de entrada
        tgt_vocab_size: tamanho do vocabulário de saída
        d_model:        dimensão do modelo (padrão: 512)
        num_heads:      número de cabeças de atenção (padrão: 8)
        d_ff:           dimensão interna da FFN (padrão: 2048)
        num_layers:     número de blocos encoder/decoder empilhados (padrão: 6)
        max_len:        comprimento máximo de sequência
    """
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        max_len=100
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings de entrada e saída
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding  = PositionalEncoding(d_model, max_len)

        # Pilha do Encoder: N blocos empilháveis
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # Pilha do Decoder: N blocos empilháveis
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # Projeção final: d_model → vocab_size
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask=None):
        """
        Codifica a sequência de entrada, produzindo a memória Z.

        Args:
            src:      índices de tokens de entrada (batch, src_seq_len)
            src_mask: máscara de padding (opcional)

        Returns:
            Z: memória contextualizada (batch, src_seq_len, d_model)
        """
        # Embedding + escala + Positional Encoding
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Passa por cada EncoderBlock (empilhados)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x  # Z

    def decode(self, tgt, Z, tgt_mask=None, src_mask=None):
        """
        Decodifica usando a memória Z do encoder.

        Args:
            tgt:      índices de tokens alvo (batch, tgt_seq_len)
            Z:        memória do encoder (batch, src_seq_len, d_model)
            tgt_mask: máscara causal do decoder
            src_mask: máscara de padding do encoder (opcional)

        Returns:
            logits: (batch, tgt_seq_len, tgt_vocab_size)
        """
        # Embedding + escala + Positional Encoding
        y = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        y = self.pos_encoding(y)

        # Passa por cada DecoderBlock (empilhados)
        for layer in self.decoder_layers:
            y = layer(y, Z, tgt_mask, src_mask)

        # Projeção linear → vocabulário + Softmax
        logits = self.output_linear(y)         # (batch, tgt_seq_len, vocab_size)
        probs  = F.softmax(logits, dim=-1)     # probabilidades por token

        return probs

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass completo (usado durante treino).

        Args:
            src:      sequência de entrada (batch, src_seq_len)
            tgt:      sequência alvo (batch, tgt_seq_len)
            src_mask: máscara encoder (opcional)
            tgt_mask: máscara causal decoder

        Returns:
            probs: (batch, tgt_seq_len, tgt_vocab_size)
        """
        Z     = self.encode(src, src_mask)
        probs = self.decode(tgt, Z, tgt_mask, src_mask)
        return probs


# =============================================================================
# TAREFA 4: INFERÊNCIA AUTO-REGRESSIVA
# =============================================================================

def autoregressive_inference(model, encoder_input, vocab, max_new_tokens=20):
    """
    Loop auto-regressivo de inferência (geração token a token).

    Algoritmo:
        1. Codifica encoder_input → memória Z
        2. Inicia sequência do decoder com <START>
        3. A cada iteração:
           a. Cria máscara causal para o comprimento atual
           b. Decodifica → obtém distribuição de probabilidade
           c. Seleciona o token de maior probabilidade (greedy)
           d. Concatena novo token à sequência
           e. Para se gerar <EOS> ou atingir max_new_tokens

    Args:
        model:          instância do Transformer
        encoder_input:  tensor (1, src_seq_len) com índices do encoder
        vocab:          dicionário {token_str: índice}
        max_new_tokens: limite de segurança para evitar loop infinito

    Returns:
        generated_tokens: lista de strings com os tokens gerados
    """
    model.eval()

    # Vocabulário reverso: índice → string
    idx_to_token = {v: k for k, v in vocab.items()}

    START_IDX = vocab['<START>']
    EOS_IDX   = vocab['<EOS>']

    with torch.no_grad():
        # Passo 1: Encoder — processa a entrada e gera memória Z
        Z = model.encode(encoder_input)
        print(f"\n[ENCODER] Processou '{' '.join([idx_to_token[i.item()] for i in encoder_input[0]])}'"
              f" → Memória Z: {Z.shape}")

        # Passo 2: Decoder inicia com token <START>
        # decoder_input: (1, 1) com o índice do <START>
        decoder_input = torch.tensor([[START_IDX]])
        generated_tokens = ['<START>']

        print(f"\n[DECODER] Iniciando geração auto-regressiva...\n")
        print(f"  Iteração 0 → Token: <START>")

        # Passo 3: Loop auto-regressivo
        for step in range(max_new_tokens):
            tgt_seq_len = decoder_input.size(1)

            # Cria máscara causal para o comprimento atual da sequência
            # Shape: (tgt_seq_len, tgt_seq_len) — impede ver tokens futuros
            causal_mask = make_causal_mask(tgt_seq_len)

            # Adiciona dimensões de batch e head: (1, 1, tgt_seq_len, tgt_seq_len)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            # Decodifica: obtém probabilidades para todos os tokens da seq atual
            probs = model.decode(decoder_input, Z, tgt_mask=causal_mask)

            # Pega apenas a distribuição do ÚLTIMO token gerado
            # probs: (1, tgt_seq_len, vocab_size) → last_probs: (vocab_size,)
            last_probs = probs[0, -1, :]

            # Greedy decoding: seleciona o token com maior probabilidade
            next_token_idx = torch.argmax(last_probs).item()
            next_token_str = idx_to_token[next_token_idx]

            print(f"  Iteração {step + 1} → Token previsto: '{next_token_str}' "
                  f"(prob: {last_probs[next_token_idx].item():.4f})")

            # Concatena o novo token à sequência do decoder
            next_token_tensor = torch.tensor([[next_token_idx]])
            decoder_input     = torch.cat([decoder_input, next_token_tensor], dim=1)
            generated_tokens.append(next_token_str)

            # Para ao gerar <EOS>
            if next_token_idx == EOS_IDX:
                print(f"\n[DECODER] Token <EOS> gerado. Inferência concluída.")
                break
        else:
            print(f"\n[DECODER] Limite de {max_new_tokens} tokens atingido.")

    return generated_tokens


# =============================================================================
# DEMONSTRAÇÃO COMPLETA
# =============================================================================

def main():
    print("=" * 65)
    print("  Laboratório 04 — Transformer Completo 'From Scratch'")
    print("  ICEV — Instituto de Ensino Superior")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Vocabulários fictícios (toy sequence)
    # ------------------------------------------------------------------
    src_vocab = {
        '<PAD>': 0, '<START>': 1, '<EOS>': 2,
        'Thinking': 3, 'Machines': 4
    }

    tgt_vocab = {
        '<PAD>': 0, '<START>': 1, '<EOS>': 2,
        'Máquinas': 3, 'Pensantes': 4,
        'machine': 5, 'learning': 6,
        'palavra_A': 7, 'palavra_B': 8
    }

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # ------------------------------------------------------------------
    # Hiperparâmetros
    # Usamos valores menores para demonstração (originais: 512, 8, 2048, 6)
    # ------------------------------------------------------------------
    d_model   = 64     # dimensão do modelo (paper: 512)
    num_heads = 8      # número de cabeças  (paper: 8)
    d_ff      = 256    # dimensão FFN       (paper: 2048)
    num_layers= 2      # camadas empilhadas (paper: 6)

    assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"

    # ------------------------------------------------------------------
    # Instancia o modelo completo
    # ------------------------------------------------------------------
    print("\n[MODELO] Instanciando Transformer Encoder-Decoder...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=100
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parâmetros totais: {total_params:,}")
    print(f"  d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}, "
          f"num_layers={num_layers}")

    # ------------------------------------------------------------------
    # Tensor de entrada do encoder simulando "Thinking Machines"
    # Shape: (batch=1, src_seq_len=2)
    # ------------------------------------------------------------------
    encoder_input = torch.tensor([[
        src_vocab['Thinking'],
        src_vocab['Machines']
    ]])
    print(f"\n[INPUT] encoder_input (frase 'Thinking Machines'): {encoder_input}")
    print(f"        shape: {encoder_input.shape}")

    # ------------------------------------------------------------------
    # Teste do forward pass completo (modo treino com teacher forcing)
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("[TESTE] Forward pass completo (modo treino)...")

    # Sequência alvo de exemplo: <START> Máquinas Pensantes <EOS>
    decoder_target = torch.tensor([[
        tgt_vocab['<START>'],
        tgt_vocab['Máquinas'],
        tgt_vocab['Pensantes'],
        tgt_vocab['<EOS>']
    ]])

    tgt_seq_len = decoder_target.size(1)
    causal_mask = make_causal_mask(tgt_seq_len).unsqueeze(0).unsqueeze(0)

    output_probs = model(encoder_input, decoder_target, tgt_mask=causal_mask)
    print(f"  Saída do forward: shape = {output_probs.shape}")
    print(f"  (batch=1, tgt_seq_len={tgt_seq_len}, vocab_size={tgt_vocab_size})")
    print(f"  Soma das probabilidades no passo 0: {output_probs[0, 0, :].sum().item():.6f} (deve ≈ 1.0)")

    # ------------------------------------------------------------------
    # Tarefa 4: Inferência auto-regressiva
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("[INFERÊNCIA] Loop auto-regressivo iniciado...")

    generated = autoregressive_inference(
        model=model,
        encoder_input=encoder_input,
        vocab=tgt_vocab,
        max_new_tokens=10
    )

    print(f"\n[RESULTADO] Sequência gerada: {' → '.join(generated)}")

    # ------------------------------------------------------------------
    # Verificações de sanidade dos módulos individuais
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("[SANITY CHECK] Verificando módulos individualmente...\n")

    batch, seq, d = 1, 3, d_model

    # Atenção
    Q = torch.randn(batch, seq, d)
    K = torch.randn(batch, seq, d)
    V = torch.randn(batch, seq, d)
    attn_out, attn_w = scaled_dot_product_attention(Q, K, V)
    print(f"  scaled_dot_product_attention: output={attn_out.shape}, "
          f"weights={attn_w.shape} ✓")

    # Máscara causal
    mask = make_causal_mask(seq)
    print(f"  make_causal_mask({seq}):\n{mask.int()}")
    assert mask[0, 1].item() == True  and mask[1, 0].item() == False, \
        "Máscara causal incorreta!"
    print(f"  Máscara causal: OK ✓")

    # FFN
    ffn = PositionwiseFFN(d_model=d, d_ff=d * 4)
    x   = torch.randn(batch, seq, d)
    print(f"  FFN output: {ffn(x).shape} ✓")

    # Add & Norm
    norm   = nn.LayerNorm(d)
    sublyr = torch.randn(batch, seq, d)
    an_out = add_and_norm(x, sublyr, norm)
    print(f"  Add & Norm output: {an_out.shape} ✓")

    # EncoderBlock
    enc = EncoderBlock(d_model=d, num_heads=num_heads, d_ff=d * 4)
    z   = enc(x)
    print(f"  EncoderBlock output (Z): {z.shape} ✓")

    # DecoderBlock
    dec     = DecoderBlock(d_model=d, num_heads=num_heads, d_ff=d * 4)
    y       = torch.randn(batch, seq, d)
    dec_out = dec(y, z)
    print(f"  DecoderBlock output: {dec_out.shape} ✓")

    print("\n" + "=" * 65)
    print("  Todos os módulos verificados com sucesso!")
    print("  Laboratório 04 concluído.")
    print("=" * 65)


if __name__ == "__main__":
    main()
