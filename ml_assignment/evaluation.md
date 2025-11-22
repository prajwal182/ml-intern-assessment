# Evaluation

Please provide a 1-page summary of your design choices for the Trigram Language Model.

This should include:

- How you chose to store the n-gram counts.
- How you handled text cleaning, padding, and unknown words.
- How you implemented the `generate` function and the probabilistic sampling.
- Any other design decisions you made and why you made them.



## 🧠 Task 1: Trigram Language Model (Core Python & NLP)

The goal of this task was to build a probabilistic text generation model from scratch, demonstrating strong core Python and fundamental NLP skills.

### 1. Data Structure and Efficiency

| Feature                   | Implementation                                  | Rationale |
| :---                      | :---                                            | :---      |
| **Storage**               | **Nested `defaultdict(int)`:** `self.counts[w1][w2][w3] = frequency` 
| This structure provides maximum efficiency for count storage. It allows **O(1)** lookup and direct key assignment (e.g., `self.counts['the']['cat']['sat'] += 1`) without needing tedious conditional checks (`if key in dict: ...`), which simplifies and speeds up the `fit` method. |

| **Dependencies** | Standard Python, **`numpy`** (for sampling), **`re`** (for cleaning), and **`defaultdict`**. 
| Minimized external libraries while using `numpy` for efficient, array-based random sampling. |

### 2. Text Preprocessing and Tokenization

| Feature                   | Implementation                                        | Rationale |
| :---                      | :---                                                  | :---      |
| **Normalization** | Applied **lowercasing** (`text.lower()`) and used the `re` module for basic punctuation removal. 
| Standardizes the vocabulary (e.g., "The" == "the") and reduces noise, leading to more accurate frequency counts. |

| **Sentence Boundaries** | Used **padding** with `['<S>', '<S>']` (start) and `['</S>']` (end) tokens for each sentence. 
| As a trigram model ($N=3$) requires two preceding words for prediction, the two `<S>` tokens provide the necessary initial context to predict the first word. The `</S>` token provides a stopping condition for the `generate` loop. |

### 3. Probabilistic Generation (`generate` method)

| Feature                       | Implementation                                | How it Achieved the Goal |
| :---                          | :---                                          | :---                     |
| **Context Lookup** | Uses the two preceding words $(w_1, w_2)$ to query the counts table: `self.counts[w1][w2]`. | Adheres to the Markov assumption for trigrams, where the next word's probability depends only on the two previous words. |
| **Probability Calculation** | Calculated $P(w_3 \mid w_1, w_2)$ as the ratio of $\text{Count}(w_1, w_2, w_3)$ to $\sum_w \text{Count}(w_1, w_2, w)$. | Converts raw counts into a **valid probability distribution**, where the sum of all next-word probabilities equals 1. |
| **Sampling (The Hardest Part)** | Used **`numpy.random.choice`** with the calculated probability array (`p=probabilities`). | This ensures that the generated text is **stochastic** (random) rather than deterministic. Words that appear $60\%$ of the time are chosen $60\%$ of the time, simulating the variability of natural language. 



## 🧠 Task 2: Scaled Dot-Product Attention (Linear Algebra & Deep Learning)

This optional task demonstrates a deep understanding of the core mechanism that powers modern language models like BERT and GPT. The entire implementation uses **NumPy arrays** to perform the computation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

### 1. Core Mathematical Implementations

| Step              | Implementation in NumPy                           | Rationale and Purpose |
| :---              | :---                                              | :---                  |
| **Similarity ($QK^T$)** | `scores = Q @ K.swapaxes(-2, -1)` |
 Uses the NumPy **`@`** operator (matrix multiplication) for efficient calculation of the similarity score between every **Query** and every **Key** vector. `swapaxes` performs the necessary matrix transpose ($\mathbf{K}^T$). |

| **Scaling** | `scores = scores / np.sqrt(d_k)` 
| **Normalization:** Dividing by $\mathbf{1/\sqrt{d_k}}$ prevents the dot products from growing excessively large as the dimension ($d_k$) increases, which ensures the gradients remain stable and non-zero after the softmax operation. |

| **Softmax** | Custom `softmax` function that uses a maximum-subtraction trick (`x - np.max(x)`). 
| Ensures **numerical stability** by preventing exponential overflow errors, which are common when working with large inputs in the exponent. |

| **Final Output** | `output = attention_weights @ V` 
| Calculates the final output, which is the **weighted sum** of the $\mathbf{V}$alue vectors, where the **attention weights** determine the contribution of each input element.  |

### 2. Handling Masking

* **Masking Technique:** The mask ($\mathbf{M}$) is applied by adding a very large negative number ($\approx -10^9$) to the score matrix where attention should be blocked: `scores = scores + (mask * -1e9)`.
* **Effect:** When these altered scores pass through the Softmax function, $e^{-10^9} \approx 0$. This forces the attention weight for masked positions to be **zero**, ensuring the model cannot use irrelevant or future information (e.g., in a decoder model).

---

## 🚀 Demonstration and Verification

The entire solution was verified using the `pytest` framework, which confirmed the correctness of both modules:

* **Task 1 Verification:** Passed all tests, confirming proper counting, boundary handling, and probabilistic generation.
* **Task 2 Verification:** Passed tests confirming correct scaling, masking, and the mathematical output of the attention mechanism.

The model was demonstrated to be functional by running `run.py`, which displayed learned sentence starter counts and generated unique, coherent text sequences.

### ⭐ Optional Context: Broader ML Concepts
# Smoothing for Trigram Models
For a production-ready system, the Trigram Model would require smoothing (e.g., Kneser-Ney or Laplace smoothing). This technique addresses the zero-frequency problem by reallocating small portions of probability mass from observed sequences to unseen sequences. This prevents the model from breaking when it encounters a word or sequence it did not see during training, making generation much more robust.

# Attention vs. Recurrence
The attention mechanism implemented in Task 2 is fundamental because it replaced older sequential models (like RNNs and LSTMs). Attention allows the model to process all tokens in parallel and calculate long-range dependencies instantly, rather than processing one token at a time, which is the key to the massive parallelization and speed of modern Transformer architectures.