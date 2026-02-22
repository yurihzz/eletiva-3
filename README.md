# Scaled Dot-Product Attention

Implementacao do mecanismo de Scaled Dot-Product Attention conforme descrito
no paper "Attention Is All You Need" (Vaswani et al., 2017).

---

## Estrutura do Repositorio

    .
    attention.py        -> Implementacao principal
    test_attention.py   -> Testes unitarios
    README.md           -> Esta documentacao

---

## Como Rodar

### Pre-requisitos

- Python 3.10+
- NumPy

    pip install numpy

### Executar os testes

    python test_attention.py

Saida esperada:

    =======================================================
      Testes: Scaled Dot-Product Attention
    =======================================================
    ...
      Resultado: 12/12 testes passaram.
      Todos os testes PASSARAM com sucesso!
    =======================================================

### Usar a funcao diretamente

    import numpy as np
    from attention import scaled_dot_product_attention

    Q = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    K = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    V = np.array([[1.0, 0.0],
                  [0.0, 1.0]])

    output, weights = scaled_dot_product_attention(Q, K, V)
    print("Output:\n", output)
    print("Attention Weights:\n", weights)

---

## Equacao de Referencia

    Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) * V

---

## Como a Normalizacao sqrt(d_k) Foi Aplicada

O produto escalar QK^T cresce em magnitude a medida que a dimensao d_k aumenta.
Isso empurra o Softmax para regioes de gradiente muito pequeno (gradiente saturado).

Para compensar, cada elemento da matriz de scores e dividido por sqrt(d_k)
antes de aplicar o Softmax:

    scaling_factor = np.sqrt(d_k)       # ex.: d_k=64 -> scaling=8.0
    scores = (Q @ K.T) / scaling_factor # shape: (n_queries, n_keys)

Essa divisao mantem a variancia dos scores aproximadamente constante (~1),
independente de d_k, garantindo treinamento mais estavel.

O Softmax e entao aplicado linha a linha (cada query normaliza sua distribuicao
de atencao sobre todas as keys de forma independente):

    attention_weights = softmax(scores)  # softmax ao longo de axis=-1

Para estabilidade numerica, o Softmax subtrai o valor maximo de cada linha
antes da exponenciacao (tecnica padrao, sem alteracao matematica no resultado).

---

## Exemplo de Input e Output Esperado

    Input:

    Q = [[1.0, 0.0],    K = [[1.0, 0.0],    V = [[1.0, 0.0],
         [0.0, 1.0]]         [0.0, 1.0]]         [0.0, 1.0]]

    Attention Weights (saida):

    [[0.6742, 0.3258],
     [0.3258, 0.6742]]

    A primeira query presta mais atencao a primeira key (0.67 vs 0.33),
    e a segunda query faz o oposto -- comportamento esperado pois
    Q e K sao matrizes identidade.

    Output:

    [[0.6742, 0.0000],
     [0.0000, 0.6742]]

---

## Referencia

Vaswani, A. et al. Attention Is All You Need. NeurIPS 2017.
https://arxiv.org/abs/1706.03762
=======
# eletiva-3
>>>>>>> b6df23fe5fdfd349c6d63b9fc92d457f2e50642e
