import random
import numpy as np
from collections import defaultdict
import re

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        # Nested defaultdict for counts: counts[w1][w2][w3] = frequency
        # This structure handles the mapping efficiently.
        self.counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Define special tokens for padding
        self.START_TOKEN = '<S>'
        self.END_TOKEN = '</S>'
        # UNK_TOKEN is included for completeness but not fully implemented in this basic model
        self.UNK_TOKEN = '<UNK>'

    def _clean_text(self, text):
        """
        Preprocesses the raw text: converts to lowercase, simplifies punctuation, and splits into sentences.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace non-alphabetic characters (except spaces and period) with a space
        text = re.sub(r'[^a-z\s\.]', ' ', text)
        
        # Split into sentences using a period. Filter out any empty strings.
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        return sentences

    def fit(self, text: str):
        """
        Trains the trigram model on the given text by counting trigrams (w1, w2, w3).
        """
        sentences = self._clean_text(text)
        
        for sentence in sentences:
            # Tokenize the sentence into words
            words = sentence.split()
            
            # 1. Padding (Trigram, N=3, requires two start tokens)
            padded_words = [self.START_TOKEN, self.START_TOKEN] + words + [self.END_TOKEN]
            
            # 2. Counting the Trigrams
            # Iterate through the list, stopping 2 words before the end (to capture w1, w2, w3)
            for i in range(len(padded_words) - 2):
                w1 = padded_words[i]
                w2 = padded_words[i+1]
                w3 = padded_words[i+2]
                
                # Increment the count
                self.counts[w1][w2][w3] += 1

    def generate(self, max_length=50):
        """
        Generates new text using the trained trigram model via probabilistic sampling.
        """
        # 1. Starting Context
        # Start with the two required start tokens
        generated_words = [self.START_TOKEN, self.START_TOKEN]
        
        for _ in range(max_length):
            # 2. Determine the current context (w1, w2)
            w1 = generated_words[-2]
            w2 = generated_words[-1]
            
            # Handle unknown context (if w1, w2 pair was never seen)
            if w1 not in self.counts or w2 not in self.counts[w1]:
                break 
                
            # Get the next word counts for the current context (w1, w2)
            next_word_counts = self.counts[w1][w2]
            
            # 3. Calculate Probabilities
            candidates = list(next_word_counts.keys())
            counts = np.array(list(next_word_counts.values()))
            
            total_count = counts.sum()
            probabilities = counts / total_count
            
            # 4. Probabilistic Sampling
            # numpy.random.choice selects a word based on the probabilities (weights)
            # 

            #[Image of Probability Sampling in Language Modeling]

            next_word = np.random.choice(candidates, p=probabilities)
            
            # 5. Check for End Token
            if next_word == self.END_TOKEN:
                break
            
            generated_words.append(next_word)
            
        # Clean up the output: remove start tokens and join the remaining words
        output = generated_words[2:] # Remove the initial '<S>', '<S>'
        
        # This RETURN statement is critical for passing the tests!
        return ' '.join(output)