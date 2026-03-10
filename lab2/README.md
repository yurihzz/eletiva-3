# Transformer Encoder "From Scratch"

**Disciplina:** Tópicos em Inteligência Artificial – 2026.1  
**Professor:** Prof. Dimmy Magalhães  
**Instituição:** iCEV - Instituto de Ensino Superior

---

## Descrição

Implementação do Forward Pass de um bloco **Transformer Encoder** completo, seguindo a arquitetura do artigo *"Attention Is All You Need"* (Vaswani et al., 2017), utilizando apenas `Python 3.x`, `numpy` e `pandas`.

---

## Como Rodar

### Pré-requisitos

```bash
pip install numpy pandas
```

### Execução

```bash
python transformer_encoder.py
```

A saída no terminal mostrará:

1. O vocabulário e os IDs dos tokens
2. O formato do tensor de entrada `(Batch, Tokens, d_model)`
3. A passagem pelas 6 camadas do Encoder com confirmação das dimensões
4. A validação de sanidade (entrada e saída com mesmo shape)
5. Uma amostra dos primeiros valores do Vetor Z contextualizado

---

## Estrutura do Código

| Componente | Descrição |
|---|---|
| `softmax()` | Softmax estável numericamente implementada com `np.exp` |
| `ScaledDotProductAttention` | Attention(Q,K,V) = softmax(QK^T / √dk) · V |
| `layer_norm()` | Normalização por média e variância no último eixo |
| `FeedForwardNetwork` | FFN(x) = max(0, xW1 + b1)W2 + b2 |
| `EncoderLayer` | Um bloco completo com residual connections + LN |
| Loop principal | Empilha N=6 camadas sequencialmente |

---

## Parâmetros Utilizados

| Parâmetro | Valor | Referência (paper) |
|---|---|---|
| `d_model` | 64 | 512 |
| `d_ff` | 256 | 2048 |
| `N` (camadas) | 6 | 6 |
| `epsilon` (LayerNorm) | 1e-6 | — |

> O `d_model = 64` foi utilizado conforme permitido pelo enunciado para processamento em CPU.

---

## Nota de Crédito — Uso de IA Generativa

Este projeto foi **desenvolvido com auxílio do Claude (Anthropic)**, ferramenta de IA Generativa.

A IA foi utilizada para:
- Geração da estrutura base do código e implementação das equações matemáticas
- Revisão de dimensionalidade dos tensores (broadcasting NumPy)
- Verificação da correção do fluxo completo do Encoder

Conforme as diretrizes do laboratório, o uso de IA Generativa foi empregado além do brainstorming de sintaxe. Este crédito é declarado em conformidade com a política de integridade acadêmica do enunciado.

---

## Referências

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
