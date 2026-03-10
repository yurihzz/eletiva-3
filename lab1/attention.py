
import numpy as np


def softmax(x):

    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
   
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K e V devem ser matrizes 2D (n, d).")

    n_queries, d_k = Q.shape
    n_keys_k, d_k_k = K.shape
    n_keys_v, d_v = V.shape

    if d_k != d_k_k:
        raise ValueError(
            f"A dimensao d_k de Q ({d_k}) e K ({d_k_k}) deve ser igual."
        )
    if n_keys_k != n_keys_v:
        raise ValueError(
            f"O numero de linhas de K ({n_keys_k}) e V ({n_keys_v}) deve ser igual."
        )

    scaling_factor = np.sqrt(d_k)
    scores = (Q @ K.T) / scaling_factor 

    attention_weights = softmax(scores) 

    output = attention_weights @ V  

    return output, attention_weights
