# Laboratório 3 — Implementando o Decoder

Implementação dos blocos matemáticos centrais do Decoder de um Transformer, conforme especificado no enunciado do Laboratório 3 (Instituto de Ensino Superior ICEV).

---

## Requisitos

- Python 3.8+
- NumPy

Instale a dependência com:

```bash
pip install numpy
```

---

## Como executar

```bash
python laboratorio3_decoder.py
```

---

## Estrutura do código

### Tarefa 1 — Máscara Causal (`create_causal_mask`)

Cria uma matriz quadrada `[seq_len, seq_len]` onde:
- Triangular inferior + diagonal principal → `0`
- Triangular superior → `-∞`

A máscara é somada ao resultado de `QKᵀ / √d_k` antes do Softmax, zerando as probabilidades de todas as posições futuras e impedindo que o modelo "olhe para frente" durante o treinamento.

**Prova real:** após o Softmax, todas as posições acima da diagonal são estritamente `0.0`, verificado por assertion automática no código.

---

### Tarefa 2 — Cross-Attention (`cross_attention`)

Simula a ponte Encoder-Decoder:

- `encoder_output` → shape `[1, 10, 512]` (frase em francês)
- `decoder_state`  → shape `[1, 4, 512]`  (tokens já gerados em inglês)

A Query (Q) é projetada a partir do `decoder_state`; as Keys (K) e Values (V) são projetadas a partir do `encoder_output`. O Scaled Dot-Product Attention é calculado **sem máscara causal**, pois o Decoder deve ter acesso completo à frase original do Encoder.

---

### Tarefa 3 — Loop de Inferência Auto-Regressivo

- `generate_next_token` simula o Decoder devolvendo um vetor de probabilidades de tamanho `V = 10.000`.
- O loop `while` aplica `argmax` a cada passo, adiciona o token à sequência e interrompe imediatamente ao gerar `<EOS>`, imprimindo a frase final.

---

## Exemplo de saída

```
Máscara Causal M (5x5):
[[  0. -inf -inf -inf -inf]
 [  0.   0. -inf -inf -inf]
 [  0.   0.   0. -inf -inf]
 [  0.   0.   0.   0. -inf]
 [  0.   0.   0.   0.   0.]]

✅ Todas as posições futuras são estritamente 0.0

Saída do Cross-Attention shape : (1, 4, 64)
Soma das linhas de atenção     : [1. 1. 1. 1.]

Passo 01 | token: 'palavra_1584'
Passo 02 | token: 'palavra_6279'
...
🛑 Token <EOS> detectado! Geração encerrada.

✅ Frase final: <START> palavra_1584 palavra_6279 ... <EOS>
```
