import numpy as np

def softmax(x):
    """Numerically stable softmax computation."""
    # Ensure this uses the correct axis (last one)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Computes the Scaled Dot-Product Attention.
    
    Args:
        Q (np.ndarray): Query matrix (batch, head, query_len, d_k)
        K (np.ndarray): Key matrix (batch, head, key_len, d_k)
        V (np.ndarray): Value matrix (batch, head, key_len, d_v)
        mask (np.ndarray, optional): Mask matrix (batch, head, query_len, key_len).
    
    Returns:
        tuple: (output_matrix, attention_weights)
    """
    
    
    # 2. Calculate the raw attention scores (Q * K^T)
    # The @ operator performs matrix multiplication (np.matmul)
    # The axes are automatically handled by NumPy's broadcasting rules
    # Q: (..., query_len, d_k), K: (..., key_len, d_k) -> Scores: (..., query_len, key_len)
    scores = Q @ K.swapaxes(-2, -1)
    
    # 1. Get the dimension of the keys (d_k)
    d_k = K.shape[-1]
    
    # 3. Apply Scaling: Divide by the square root of d_k
    # This scaling prevents the dot products from becoming too large, which stabilizes softmax.
    scores = scores / np.sqrt(d_k)
    
    # 4. Apply Mask (if provided)
    if mask is not None:
        # Masking prevents attention to certain elements (e.g., future tokens in decoding).
        # We fill masked positions with a very large negative number (like -1e9) 
        # so they become zero after the softmax function.
        scores = scores + (mask * -1e9)
        
    # 5. Apply Softmax to get Attention Weights (Probabilities)
    # Summing to 1 across the last axis (the key_len dimension)
    attention_weights = softmax(scores)
    
    # 6. Calculate the final output (Weights * V)
    # Weights: (..., query_len, key_len), V: (..., key_len, d_v) -> Output: (..., query_len, d_v)
    output = attention_weights @ V
    
    return output, attention_weights