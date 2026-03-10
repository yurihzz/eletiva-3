

import numpy as np
from attention import scaled_dot_product_attention, softmax

_passed = 0
_failed = 0


def assert_allclose(actual, expected, atol=1e-6, label=""):
    global _passed, _failed
    if np.allclose(actual, expected, atol=atol):
        print(f"  [PASSOU] {label}")
        _passed += 1
    else:
        print(f"  [FALHOU] {label}")
        print(f"           Esperado:\n{expected}")
        print(f"           Obtido:\n{actual}")
        _failed += 1


def assert_true(condition, label=""):
    global _passed, _failed
    if condition:
        print(f"  [PASSOU] {label}")
        _passed += 1
    else:
        print(f"  [FALHOU] {label}")
        _failed += 1

def test_softmax_rows():
    print("\n=== Teste 1: Softmax linha a linha ===")

    x = np.array([[1.0, 2.0, 3.0],
                   [1.0, 1.0, 1.0]])

    result = softmax(x)

    
    row_sums = result.sum(axis=1)
    assert_allclose(row_sums, np.ones(2), label="Linhas somam 1")

    
    assert_true((result >= 0).all() and (result <= 1).all(),
                label="Valores entre 0 e 1")

    
    expected_uniform = np.array([1/3, 1/3, 1/3])
    assert_allclose(result[1], expected_uniform, label="Entrada uniforme -> prob. iguais")


def test_attention_simple():
   
    print("\n=== Teste 2: Exemplo numerico simples ===")

    Q = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    K = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    V = np.array([[1.0, 0.0],
                  [0.0, 1.0]])

    output, weights = scaled_dot_product_attention(Q, K, V)

    d_k = 2
    scaling = np.sqrt(d_k)
    scores = (Q @ K.T) / scaling
    expected_weights = softmax(scores)
    expected_output = expected_weights @ V

    assert_allclose(weights, expected_weights, label="Pesos de atencao corretos")
    assert_allclose(output, expected_output,   label="Output de atencao correto")

    
    assert_allclose(weights.sum(axis=1), np.ones(Q.shape[0]),
                    label="Linhas dos pesos somam 1")


def test_scaling_factor():
   
    print("\n=== Teste 3: Fator de escala sqrt(d_k) ===")

    np.random.seed(42)
    Q = np.random.randn(3, 4)
    K = np.random.randn(3, 4)
    V = np.random.randn(3, 4)

    output_scaled, weights_scaled = scaled_dot_product_attention(Q, K, V)

   
    scores_unscaled = Q @ K.T
    weights_unscaled = softmax(scores_unscaled)
    output_unscaled = weights_unscaled @ V

    
    different = not np.allclose(weights_scaled, weights_unscaled)
    assert_true(different, label="Saida com escala difere da sem escala")

   
    scores_expected = (Q @ K.T) / np.sqrt(4)
    expected_weights = softmax(scores_expected)
    assert_allclose(weights_scaled, expected_weights,
                    label="Escala sqrt(d_k=4) = 2.0 aplicada corretamente")

def test_output_shapes():
    print("\n=== Teste 4: Shapes de entrada e saida ===")

    
    Q = np.random.randn(3, 8)
    K = np.random.randn(5, 8)
    V = np.random.randn(5, 6)

    output, weights = scaled_dot_product_attention(Q, K, V)

    assert_true(output.shape == (3, 6),
                label=f"Output shape (3, 6) -- obtido: {output.shape}")
    assert_true(weights.shape == (3, 5),
                label=f"Weights shape (3, 5) -- obtido: {weights.shape}")

def test_invalid_inputs():
    print("\n=== Teste 5: Entradas invalidas ===")

    
    try:
        scaled_dot_product_attention(
            np.ones((2, 3)), np.ones((2, 4)), np.ones((2, 4))
        )
        assert_true(False, label="Deveria levantar ValueError (d_k incompativel)")
    except ValueError:
        assert_true(True, label="ValueError levantado para d_k incompativel")


    try:
        scaled_dot_product_attention(
            np.ones((2, 4)), np.ones((3, 4)), np.ones((5, 4))
        )
        assert_true(False, label="Deveria levantar ValueError (n_keys incompativel)")
    except ValueError:
        assert_true(True, label="ValueError levantado para n_keys incompativel")



if __name__ == "__main__":
    print("=" * 55)
    print("  Testes: Scaled Dot-Product Attention")
    print("=" * 55)

    test_softmax_rows()
    test_attention_simple()
    test_scaling_factor()
    test_output_shapes()
    test_invalid_inputs()

    print("\n" + "=" * 55)
    total = _passed + _failed
    print(f"  Resultado: {_passed}/{total} testes passaram.")
    if _failed == 0:
        print("  Todos os testes PASSARAM com sucesso!")
    else:
        print(f"  {_failed} teste(s) FALHARAM.")
    print("=" * 55)
