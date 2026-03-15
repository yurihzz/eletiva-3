# Laboratório Técnico 04 — O Transformer Completo "From Scratch"

**Instituto de Ensino Superior ICEV**

> **Partes geradas/complementadas com IA, revisadas por [Seu Nome]**

---

## Descrição

Implementação completa da arquitetura **Transformer Encoder-Decoder** em PyTorch,
integrando todos os módulos construídos nos laboratórios anteriores (Labs 01–03)
em uma única topologia coerente, com inferência auto-regressiva fim-a-fim.

---

## Estrutura do Projeto

```
.
├── transformer_lab04.py   # Código principal com toda a implementação
└── README.md              # Este arquivo
```

---

## Requisitos

```bash
pip install torch
```

---

## Como Executar

```bash
python transformer_lab04.py
```

---

## Organização do Código

### Tarefa 1 — Blocos de Construção (Refatoração)

| Função / Classe | Descrição |
|---|---|
| `scaled_dot_product_attention(Q, K, V, mask)` | Atenção escalonada com produto escalar (Lab 01) |
| `make_causal_mask(seq_len)` | Máscara causal upper-triangular (Lab 02) |
| `PositionwiseFFN(d_model, d_ff)` | FFN com expansão de dimensão e ReLU (Lab 03) |
| `add_and_norm(x, sublayer_output, norm)` | Conexão residual + LayerNorm (Lab 03) |
| `MultiHeadAttention(d_model, num_heads)` | Atenção multi-cabeça com projeções W_Q, W_K, W_V, W_O |

### Tarefa 2 — Encoder Block

**Fluxo exato:**
```
X (+ Positional Encoding)
  → Self-Attention (Q=K=V=X, sem máscara causal)
  → Add & Norm
  → FFN
  → Add & Norm
  → Z (memória contextualizada bidirecional)
```

- Classe: `EncoderBlock(d_model, num_heads, d_ff)`
- Blocos são empilháveis via `nn.ModuleList` dentro do `Transformer`

### Tarefa 3 — Decoder Block

**Fluxo exato:**
```
Y (+ Positional Encoding)
  → Masked Self-Attention (máscara causal, -∞ no futuro)
  → Add & Norm
  → Cross-Attention (Q=saída anterior; K, V = memória Z do Encoder)
  → Add & Norm
  → FFN
  → Add & Norm
  → Linear (d_model → vocab_size)
  → Softmax → probabilidades
```

- Classe: `DecoderBlock(d_model, num_heads, d_ff)`
- A máscara causal usa `torch.triu(diagonal=1)` → posições futuras recebem `-inf` antes do Softmax

### Tarefa 4 — Inferência Auto-Regressiva

**Função:** `autoregressive_inference(model, encoder_input, vocab)`

**Algoritmo:**
1. Codifica `encoder_input` ("Thinking Machines") → memória **Z**
2. Inicia `decoder_input = [<START>]`
3. Loop `while`:
   - Gera máscara causal para o comprimento atual
   - Decodifica → distribuição de probabilidade sobre o vocabulário
   - Seleciona token com maior probabilidade (**greedy decoding**)
   - Concatena novo token à sequência do decoder
   - Para ao gerar `<EOS>` ou atingir limite de segurança

---

## Decisões de Implementação e Lógica Matemática

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

A divisão por $\sqrt{d_k}$ evita gradientes muito pequenos quando $d_k$ é grande
(os produtos escalares crescem em magnitude com a dimensão).

### Máscara Causal

Usamos `torch.triu(ones, diagonal=1)` para criar uma matriz triangular superior
onde `1` representa posições **futuras** (proibidas). Antes do Softmax, essas
posições recebem `-∞`, fazendo com que o Softmax as torne `0`.

### Add & Norm

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

A **conexão residual** soma a entrada original à saída da sub-camada,
preservando o gradiente no backward pass e facilitando o treinamento profundo.

### Cross-Attention no Decoder

No Cross-Attention, **Q vem do decoder** (o que queremos gerar) e
**K, V vêm da memória Z do encoder** (o que foi codificado da entrada).
Isso permite que cada posição do decoder "consulte" toda a sequência de entrada.

---

## Hiperparâmetros Usados (Demonstração)

| Parâmetro | Valor (demo) | Valor original (paper) |
|---|---|---|
| `d_model` | 64 | 512 |
| `num_heads` | 8 | 8 |
| `d_ff` | 256 | 2048 |
| `num_layers` | 2 | 6 |

> Valores reduzidos para demonstração sem GPU. A lógica é idêntica ao paper original.

---

## Referência

Vaswani, A. et al. **"Attention Is All You Need"**. NeurIPS 2017.
