import os
import sys

# FIX: Add the directory *containing* ml-assignment (the project root) to the system path.
# This ensures Python can find the package named 'ml_assignment'.
try:
    # This should be your project root folder path
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
except Exception as e:
    print(f"Error setting path: {e}")

# The import must use underscores, as Python packages cannot have hyphens.
from ml_assignment.src.ngram_model import TrigramModel 

# --- Rest of your original code ---

def load_text(file_path):
    """Loads text from a given file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Check the exact location of your corpus file
        print(f"Error: Text file not found at {file_path}")
        return None

def main():
    # --- Configuration ---
    # Ensure this path is correct relative to the script's execution from the root
    CORPUS_PATH = os.path.join('ml_assignment', 'data', 'example_corpus.txt')
    
    raw_text = load_text(CORPUS_PATH)
    
    if raw_text is None:
        return

    # --- 1. Train the Model ---
    model = TrigramModel()
    print("--- 1. Training Model ---")
    model.fit(raw_text)
    print("Training Complete.\n")

    # --- 2. Examine Learned Counts (The Solution/Training) ---
    print("--- 2. Examining Learned Counts (Solution) ---")
    
    # Example 1: Check how many words can follow the start sequence '<S> <S>'
    start_context = model.counts['<S>']['<S>']
    print(f"Count of words following '<S> <S>': {len(start_context)} unique words.")
    
    # Example 2: Check predictions for a common phrase, e.g., "i am"
    w1, w2 = 'i', 'am'
    if w1 in model.counts and w2 in model.counts[w1]:
        print(f"Words following '{w1} {w2}': {model.counts[w1][w2]}")
    else:
        print(f"Context '{w1} {w2}' not found in training data.")
        
    print("-" * 30 + "\n")

    # --- 3. Generate New Text (The Result) ---
    print("--- 3. Generating New Text (Result) ---")
    for i in range(5):
        generated = model.generate(max_length=30)
        print(f"Generated Sentence {i+1}: {generated}")

if __name__ == "__main__":
    main()