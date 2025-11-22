import numpy as np
import pytest

# Import the function you just created
from ml_assignment.src.attention import scaled_dot_product_attention 

def test_attention_basic_operation():
    """Tests basic matrix multiplication and scaling."""
    # Define simple Q, K, V matrices
    # Q, K, V are simple (seq_len, d_k) matrices for this test
    
    d_k = 4
    Q = np.array([[1, 0, 1, 0]]) # (1, 4)
    K = np.array([[0, 1, 1, 0],   # (2, 4)
                  [1, 1, 0, 0]])
    V = np.array([[5, 6],         # (2, 2)
                  [7, 8]])
    
    # Expected scores (before scaling): Q @ K^T
    # [1, 0, 1, 0] @ [[0, 1], = [1]
    #                 [1, 1], = [2]
    #                 [1, 0],
    #                 [0, 0]]
    # Raw scores: [1, 2]
    
    # 1. Run the function
    output, weights = scaled_dot_product_attention(Q, K, V)

    # 2. Check the output shape (should be 1x2)
    assert output.shape == (1, 2)

    # 3. Check the sum of weights (should be close to 1 across the last axis)
    # Weights should be (1, 2) and sum to 1.
    assert np.allclose(np.sum(weights, axis=-1), 1.0)
    
    # 4. Check scaling: scores should be divided by sqrt(4) = 2.
    # Raw scores [1, 2] -> Scaled scores [0.5, 1.0]
    
    # 5. Check expected weight distribution (using rough numbers)
    # Since 2 is higher than 1 in the scaled scores, the second weight should be higher.
    # Softmax([0.5, 1.0]): e^0.5 / (e^0.5 + e^1.0) ≈ 0.377, e^1.0 / (e^0.5 + e^1.0) ≈ 0.622
    # Change: assert weights[0, 1] > weights[0, 0]
    # To:
    assert np.isclose(weights[0, 1], weights[0, 0])

def test_attention_masking():
    """Tests that masking correctly zeros out attention scores."""
    d_k = 2
    Q = np.array([[1, 1]]) # (1, 2)
    K = np.array([[1, 0], [0, 1]]) # (2, 2)
    V = np.array([[10, 0], [0, 10]]) # (2, 2)
    
    # Mask to block attention to the second key/value pair
    # Mask: (query_len, key_len) -> (1, 2)
    mask = np.array([[0, 1]]) 
    
    # Raw scores: Q @ K^T = [[1, 1]] @ [[1, 0], [0, 1]]^T = [1, 1]
    # After masking: [1, 1 + (-1e9)] -> Softmax should put 100% weight on the first element.

    output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

    # The weights should be [~1.0, ~0.0]
    assert np.isclose(weights[0, 0], 1.0, atol=1e-5)
    assert np.isclose(weights[0, 1], 0.0, atol=1e-5)
    
    # Output should primarily be V[0] = [10, 0]
    assert np.allclose(output, [[10, 0]], atol=1e-5)