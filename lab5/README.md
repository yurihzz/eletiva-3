# Laboratório Técnico 05 — Treinamento Fim-a-Fim do Transformer

> Unidade I — Projeto Final


## Visão Geral

Este projeto implementa o treinamento completo de um modelo Transformer para tradução automática (EN → DE), conectado a um dataset real do Hugging Face. O objetivo não é criar um tradutor perfeito, mas provar que a arquitetura consegue aprender, forçando a função de perda (Loss) a cair significativamente ao longo das épocas.



## Estrutura do Projeto

```
transformer_lab05/
│
├── main.py                  # Ponto de entrada principal
│
├── src/
│   ├── __init__.py
│   ├── transformer.py       # Arquitetura completa (Lab 04)
│   ├── dataset.py           # Tarefa 1 + 2: Dataset e Tokenização
│   ├── train.py             # Tarefa 3: Training Loop
│   └── overfit_test.py      # Tarefa 4: Overfitting Test
│
├── outputs/                 # Checkpoints salvos (gerado em runtime)
│
├── requirements.txt
└── README.md
```

## Pré-requisitos

```bash
pip install -r requirements.txt
```

Requer Python 3.9+ e conexão à internet para baixar o dataset e o tokenizador na primeira execução.

## Como Executar

### Execução padrão (todas as tarefas)

```bash
python main.py
```

### Opções disponíveis

| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `--epochs` | 15 | Número de épocas de treinamento |
| `--batch-size` | 32 | Tamanho do batch |
| `--lr` | 1e-3 | Taxa de aprendizado |
| `--max-samples` | 1000 | Subconjunto do dataset |
| `--skip-overfit` | False | Pular a Tarefa 4 |
| `--save-model` | False | Salvar checkpoint do modelo |
| `--device` | auto | `cpu`, `cuda` ou `auto` |

### Exemplos

```bash
# Treinamento rápido com menos épocas
python main.py --epochs 10 --batch-size 16

# Treinar e salvar o modelo
python main.py --epochs 20 --save-model

# Apenas training loop, sem overfitting test
python main.py --skip-overfit
```

## Tarefas Implementadas

### Tarefa 1 — Preparando o Dataset Real (Hugging Face)

Arquivo: `src/dataset.py`

- Dataset utilizado: [`bentrevett/multi30k`](https://huggingface.co/datasets/bentrevett/multi30k) (pares EN→DE)
- Subconjunto selecionado: primeiras 1.000 frases do split `train`
- Carregamento via `datasets.load_dataset()`

```python
# Exemplo de uso isolado
from src.dataset import load_raw_pairs
pairs = load_raw_pairs(split="train", max_samples=1000)
# → [("A man in a green hat...", "Ein Mann mit grünem Hut..."), ...]
```

### Tarefa 2 — Tokenização Básica

Arquivo: `src/dataset.py`

- Tokenizador: `bert-base-multilingual-cased` via `AutoTokenizer`
- Tokens especiais:
  - `<BOS>` → `[CLS]` (ID = 101)
  - `<EOS>` → `[SEP]` (ID = 102)
  - `<PAD>` → `[PAD]` (ID = 0)
- Frases destino (Decoder) recebem `<BOS>` no início e `<EOS>` no fim
- Padding aplicado no batch via `pad_sequence` para comprimento uniforme

```
Frase original : "A man in a green hat is standing."
IDs fonte      : [1037, 2158, 1999, 1037, 2665, 6045, 2003, 3055, 1012]

Frase destino  : "Ein Mann mit grünem Hut steht."
IDs destino    : [101, 11890, 4907, 2007, ...., 102]
                  ↑BOS                          ↑EOS
```

---

### Tarefa 3 — O Motor de Otimização (Training Loop)

Arquivo: `src/train.py`

Hiperparâmetros do laboratório:

| Parâmetro | Valor |
|-----------|-------|
| `d_model` | 128 |
| `num_heads` | 4 |
| `num_layers` | 2 |
| `d_ff` | 512 |
| `dropout` | 0.1 |
| Otimizador | Adam (β₁=0.9, β₂=0.98) |
| Loss | CrossEntropyLoss (`ignore_index=PAD`) |

Fluxo de cada iteração:

```
src  ──────────────────→  Encoder  ──────────────────────────────┐
                                                                   │
tgt_input (shift right)  →  Decoder  ←──── enc_output ←──────────┘
                               │
                            logits  →  CrossEntropyLoss(logits, tgt_output)
                                                │
                                          loss.backward()     ← Gradientes fluem por
                                                │                WQ, WK, WV, WO, W1, W2
                                          optimizer.step()    ← Pesos atualizados
```

Teacher Forcing:
```
tgt          : [<BOS>, t1, t2, t3, <EOS>]
tgt_input    : [<BOS>, t1, t2, t3]         → alimenta o Decoder
tgt_output   : [t1,    t2, t3, <EOS>]      → rótulo da CrossEntropyLoss
```

O parâmetro `ignore_index=pad_idx` garante que a rede não seja penalizada por errar tokens de padding artificiais.

---

### Tarefa 4 — A Prova de Fogo (Overfitting Test)

Arquivo: `src/overfit_test.py`

Técnica clássica de debugging: forçar o modelo a memorizar um conjunto ínfimo de dados para provar que os gradientes estão fluindo corretamente.

- 8 frases selecionadas do conjunto de treino
- 300 épocas de treinamento exclusivo nessas frases
- Loop auto-regressivo (greedy decode) gera a tradução dessas frases
- O modelo deve reproduzir (ou se aproximar muito de) a tradução original

**Exemplo de saída esperada:**

```
Amostra [1]
  Fonte (EN)   : A man in a green hat is standing.
  Esperado (DE): Ein Mann mit grünem Hut steht.
  Gerado   (DE): Ein Mann mit grünem Hut steht.
  Sobreposição : 100%  ✓ MEMORIZADO
```

Se a Loss chega próximo de 0 e as traduções coincidem, os gradientes estão fluindo corretamente e a arquitetura está funcionando. ✓

---

## Arquitetura do Transformer

**Arquivo:** `src/transformer.py`

Implementação manual completa, baseada no paper *"Attention Is All You Need"* (Vaswani et al., 2017):

```
Transformer
├── Encoder (N=2 camadas)
│   ├── Embedding + PositionalEncoding
│   └── EncoderLayer × N
│       ├── MultiHeadAttention (h=4 cabeças, WQ, WK, WV, WO)
│       │   └── ScaledDotProductAttention
│       ├── Add & LayerNorm
│       ├── PositionwiseFeedForward (W1, W2)
│       └── Add & LayerNorm
│
└── Decoder (N=2 camadas)
    ├── Embedding + PositionalEncoding
    └── DecoderLayer × N
        ├── Masked MultiHeadAttention (self-attention)
        ├── Add & LayerNorm
        ├── Cross-Attention (Encoder-Decoder)
        ├── Add & LayerNorm
        ├── PositionwiseFeedForward
        └── Add & LayerNorm
```

Módulos implementados do zero (sem uso de `nn.Transformer`):
- `MultiHeadAttention`
- `PositionwiseFeedForward`
- `PositionalEncoding`
- `EncoderLayer` / `Encoder`
- `DecoderLayer` / `Decoder`
- `Transformer`

---

## Uso de IA Generativa

Conforme permitido pelo enunciado do laboratório:

> *"Podem utilizar IA para facilitar a manipulação dos Datasets e a tokenização (Tarefa 1 e 2), mas o fluxo de Forward/Backward da Tarefa 3 deve interagir estritamente com as classes construídas por vocês nos laboratórios anteriores."*

### Ferramentas de IA utilizadas

| Ferramenta | Uso | Tarefas |
|------------|-----|---------|
| Claude (Anthropic) | Auxílio na estrutura de código para carregamento do dataset Hugging Face (`datasets.load_dataset`), pipeline de tokenização com `AutoTokenizer`, função de collate com `pad_sequence`, e organização modular do projeto | Tarefa 1 e Tarefa 2 |

### O que **NÃO** foi gerado por IA

O fluxo de Forward/Backward (Tarefa 3) foi escrito para interagir exclusivamente com as classes do laboratório:

- A chamada `model(src, tgt_input)` aciona o `Transformer.forward()` construído manualmente
- O `loss.backward()` propaga gradientes pelas matrizes `WQ`, `WK`, `WV`, `WO`, `W1`, `W2` das classes `MultiHeadAttention` e `PositionwiseFeedForward`
- O loop auto-regressivo (`greedy_decode`) em `src/train.py` usa diretamente `model.encoder`, `model.decoder` e `model.output_projection`

---

## Observação sobre Poder Computacional

Conforme o enunciado, não é esperado que o modelo traduza textos inéditos com fluência. O modelo do Google (2017) treinou por 3,5 dias em 8 GPUs dedicadas. A avaliação é feita pela integridade estrutural do fluxo dos tensores e pela capacidade de fazer a curva do Loss cair (convergência do modelo).

---

## Versionamento

```bash
git tag v1.0
git push origin v1.0
```

---


