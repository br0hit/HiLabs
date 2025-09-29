import os
import json
import re
from difflib import SequenceMatcher
from word2number import w2n
from spellchecker import SpellChecker
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# --- GLOBAL MODEL LOADING ---
try:
    print("Loading the semantic analysis model (this may take a moment)...")
    TOKENIZER = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    MODEL = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
    MODEL.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading Legal-BERT model: {e}")
    print("Please ensure you have run 'pip install transformers torch'")
    MODEL = None
    TOKENIZER = None


# --- Configuration ---
TARGET_ATTRS = ["Attr1", "Attr2", "Attr3"]
OUTPUT_ROOT_DIR = "Output"  # The parent folder containing TN, WA, etc.
MAX_WORD_DISTANCE = 3 

# --- Utility Functions ---

def load_words_from_file(filename):
    """Loads a list of words or phrases from a text file, one per line."""
    try:
        # Use utf-8 encoding for safety
        with open(filename, 'r', encoding='utf-8') as f:
            # Strip leading/trailing whitespace and filter out empty lines
            return [line.strip().lower() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"🛑 Error: {filename} not found. Please create this file in the script's directory.")
        return []

# Load the required word lists
KEYWORDS = load_words_from_file("keywords.txt")
CONDITION_WORDS = load_words_from_file("condition_words.txt")

# Check if essential files are loaded
if not KEYWORDS or not CONDITION_WORDS:
    print("❌ Cannot proceed without 'keywords.txt' and 'condition_words.txt'.")
    exit()

# --- Core Logic: Text Extraction and Difference ---

def extract_and_combine_text(file_path, attr_name):
    """Loads a JSON file and extracts/combines text for a single attribute."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    combined_text = ""
    if attr_name in data:
        # data[attr_name] is a dict where keys are clause titles
        for clause_entries in data[attr_name].values():
            # clause_entries is a list of objects for that clause
            for entry in clause_entries:
                if 'text' in entry:
                    # Normalize whitespace: replace multiple spaces/newlines with a single space
                    text = ' '.join(entry['text'].split())
                    combined_text += text + " "
    
    return combined_text.strip()

def get_diff_blocks(standard_text, contract_text):
    """
    Compares two texts and returns a single string of the difference blocks (lines 
    present in only one of the documents) in lower case.
    """
    s = SequenceMatcher(None, standard_text, contract_text)
    
    diff_blocks = []
    
    # Iterate through the operations (replace, delete, insert, equal)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag in ('replace', 'delete', 'insert'):
            # Text changed, get the differing parts from standard and contract
            standard_diff = standard_text[i1:i2].strip()
            contract_diff = contract_text[j1:j2].strip()
            
            if tag == 'replace':
                diff_blocks.extend([standard_diff, contract_diff])
            elif tag == 'delete':
                diff_blocks.append(standard_diff)
            elif tag == 'insert':
                diff_blocks.append(contract_diff)
            
    # Combine all unique difference blocks into one string
    return " ".join(diff_blocks).lower()

# --- Comparison Rules Implementation ---

def check_conditional_change(standard_text, contract_text):
    """
    Implements comparison rule #1: If a key phrase difference is an extra 
    conditional word near a keyword, mark as different.
    
    This is a proxy for the proximity rule, checking if the difference 
    block itself introduces a conditional modification to a keyword.
    """
    
    # 1. Get the raw difference blocks
    diff_blocks = get_diff_blocks(standard_text, contract_text)
    if not diff_blocks:
        return False # No difference found

    # 2. Tokenize the texts for proximity analysis (Rule 1's true intent)
    std_words = re.findall(r'\b\w+\b', standard_text.lower())
    con_words = re.findall(r'\b\w+\b', contract_text.lower())

    for keyword in KEYWORDS:
        kw_words = keyword.split()
        if not kw_words:
            continue
            
        # Check if the *entire* keyword is present in one text but not the other
        if keyword in standard_text.lower() and keyword not in contract_text.lower():
             # The keyword was removed or modified in the contract
             target_words = std_words
             reference_words = con_words
             
        elif keyword not in standard_text.lower() and keyword in contract_text.lower():
             # The keyword was added in the contract
             target_words = con_words
             reference_words = std_words
        else:
            continue # Keyword is present in both, move to next keyword

        # Now check the target text for conditional words near the keyword
        keyword_regex = r'\b' + r'\s+'.join(re.escape(w) for w in kw_words) + r'\b'
        
        # Find all start indices of the keyword phrase
        for match in re.finditer(keyword_regex, " ".join(target_words)):
            kw_index = target_words.index(kw_words[0], match.start())
            
            # Define the proximity window (2 words before, 3 words after the keyword phrase)
            start_index = max(0, kw_index - MAX_WORD_DISTANCE)
            end_index = min(len(target_words), kw_index + len(kw_words) + MAX_WORD_DISTANCE)
            
            context_words = target_words[start_index:end_index]
            
            # Check if any conditional word is in the context
            if any(c_word in context_words for c_word in CONDITION_WORDS):
                # If a keyword is contextually modified by a conditional word, 
                # and this context is part of the difference (implied by the keyword check above),
                # we mark them as different.
                return True
                
    return False

def compare_attr3_texts(standard_text, contract_text):
    """
    Applies the two comparison rules to determine if texts are similar ("1") or not ("0").
    """
    if not standard_text or not contract_text:
        return "0", "Missing content for comparison."
        
    if standard_text == contract_text:
        return "1", "Identical" 

    # --- Rule 1 Check: Conditional Word Proximity ---
    if check_conditional_change(standard_text, contract_text):
        return "0", "Conditional word (e.g., 'no', 'unless') or critical phrase difference detected near a keyword."
    
    # --- Rule 2 Check: Keyword Presence in Difference Block ---
    diff_blocks = get_diff_blocks(standard_text, contract_text)
    
    # Check if the difference block contains any of the important keywords
    for keyword in KEYWORDS:
        # Use regex to search for the keyword as a whole word/phrase
        if re.search(r'\b' + re.escape(keyword) + r'\b', diff_blocks):
            return "0", f"Difference block contains the core keyword: '{keyword}'"
            
    # If neither rule is triggered, the difference is considered minor/non-material
    return "1", "Difference is non-material (e.g., typos, formatting, non-keyword changes)."


# --------------------- ATTRIBUTE 1, 2 --------------------------------
def clean_and_normalize_text(text: str | None) -> str:
    """Cleans and normalizes a text string for comparison. Returns empty string if input is None/empty."""
    if not text:
        return ""

    text = text.lower()
    try:
        words = text.split()
        reformed_words = []
        num_word_buffer = []
        for word in words:
            try:
                w2n.word_to_num(word)
                num_word_buffer.append(word)
            except ValueError:
                if num_word_buffer:
                    reformed_words.append(str(w2n.word_to_num(" ".join(num_word_buffer))))
                    num_word_buffer = []
                reformed_words.append(word)
        if num_word_buffer:
            reformed_words.append(str(w2n.word_to_num(" ".join(num_word_buffer))))
        text = " ".join(reformed_words)
    except Exception:
        # If number conversion fails, keep original text
        pass
    
    # Spell checking
    spell = SpellChecker()
    words = re.findall(r'\b\w+\b', text)
    if words:  # Only run if words exist
        misspelled = spell.unknown(words)
        corrections = {word: spell.correction(word) for word in misspelled if spell.correction(word)}
        corrected_words = [corrections.get(word, word) for word in words]
        text = " ".join(corrected_words)

    # Final cleaning
    text = re.sub(r'[^\w\s.,-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def mask_numbers(text: str | None) -> str:
    """Replaces all sequences of digits with a '<NUM>' token. Returns empty string if input is None/empty."""
    if not text:
        return ""
    return re.sub(r'\d+', '<NUM>', text)

def get_semantic_similarity(text1: str | None, text2: str | None) -> float:
    """Calculates cosine similarity using Legal-BERT embeddings."""
    if not text1 or not text2:
        return 0.0
    if MODEL is None or TOKENIZER is None:
        raise RuntimeError("Semantic model is not loaded. Cannot calculate similarity.")

    with torch.no_grad():
        inputs1 = TOKENIZER(text1, return_tensors="pt", truncation=True, max_length=512)
        inputs2 = TOKENIZER(text2, return_tensors="pt", truncation=True, max_length=512)

        outputs1 = MODEL(**inputs1)
        outputs2 = MODEL(**inputs2)

        # Mean pooling over token embeddings
        emb1 = outputs1.last_hidden_state.mean(dim=1)
        emb2 = outputs2.last_hidden_state.mean(dim=1)

    similarity_score = cosine_similarity(emb1.numpy(), emb2.numpy())[0][0]
    return float(similarity_score)



# --- CORE FUNCTION ---
def compare_attr1_texts(standard_text: str | None, contract_text: str | None, similarity_threshold: float = 0.99) -> int:
    """
    Compares a contract text to a standard text and returns 1 if standard, 0 otherwise.
    Safely handles None/empty inputs.
    """
    if not standard_text or not contract_text:
        return 0, 'Missing Text in Contract'  # Cannot be standard if one side is missing

    # Step 1: Clean texts
    cleaned_standard = clean_and_normalize_text(standard_text)
    cleaned_contract = clean_and_normalize_text(contract_text)

    if not cleaned_standard or not cleaned_contract:
        return 0, 'No Clean Text found in contract'

    # Step 2: Mask numbers
    masked_standard = mask_numbers(cleaned_standard)
    masked_contract = mask_numbers(cleaned_contract)

    # Step 3: Get semantic similarity
    similarity_score = get_semantic_similarity(masked_standard, masked_contract)

    # Step 4: Classify
    score =  1 if similarity_score >= similarity_threshold else 0
    reason = f"Semantic similarity ({similarity_score:.4f}) {'>= threshold' if score == 1 else '< threshold'}"
    return score, reason

# --- Main Processing Function ---

def process_all_folders(root_dir):
    """
    Iterates through all state folders in the root directory and performs comparison
    for Attr1, Attr2, and Attr3.
    """
    all_results = {}

    for item_name in os.listdir(root_dir):
        state_folder = os.path.join(root_dir, item_name)

        if os.path.isdir(state_folder):
            print(f"\n===========================================================")
            print(f"       ➡️ Processing and Comparing Folder: {item_name}")
            print(f"===========================================================")

            state_results = {attr: {} for attr in TARGET_ATTRS}

            for attr in TARGET_ATTRS:
                if attr == "Attr3":
                    break
                standard_text = None
                contract_data = {}

                # Extract texts for this attribute
                for filename in os.listdir(state_folder):
                    if filename.endswith('.json'):
                        file_path = os.path.join(state_folder, filename)
                        text = extract_and_combine_text(file_path, attr)
                        if text is None:
                            continue

                        if 'standard_template' in filename.lower():
                            standard_text = text
                        elif '_contract' in filename.lower():
                            try:
                                parts = filename.lower().split('_')
                                contract_part = next(p for p in parts if 'contract' in p)
                                contract_num = re.search(r'contract(\d+)', contract_part).group(1)
                                contract_name = f"Contract{contract_num}"
                                contract_data[contract_name] = text
                            except:
                                print(f"Warning: Could not parse contract number from {filename}. Skipping.")
                                continue

                if standard_text is None:
                    print(f"🛑 ERROR: Standard template not found for {attr} in {state_folder}. Skipping this attr.")
                    continue

                sorted_contracts = sorted(contract_data.keys())
                print(f"\n--- {attr}: Standard Template Length = {len(standard_text)} chars ---")

                for contract_name in sorted_contracts:
                    contract_text = contract_data[contract_name]
                    
                    # print(f"Standard Text for {contract_name}: {standard_text}")
                    # print(f"Contract Text for {contract_name}: {contract_text}")

                    if attr in ["Attr1", "Attr2"]:
                        score, reason = compare_attr1_texts(standard_text, contract_text)
                    else:  # Attr3
                        score, reason = compare_attr3_texts(standard_text, contract_text)

                    state_results[attr][contract_name] = {
                        "Similarity_Score": score,
                        "Reason": reason
                    }

                    print(f"| {contract_name:<12} | Score: {score:<1} | Classification: {'✅ Similar' if str(score) == '1' else '❌ Non-Standard'}")
                    print(f"| {'':<12} | Reason: {reason}")
                    print("-" * 70)

            all_results[item_name] = state_results

    return all_results


# --- Execute Script ---
if __name__ == "__main__":
    final_results = process_all_folders(OUTPUT_ROOT_DIR)
    
    print("\n\n--- FINAL SUMMARY OF COMPARISON RESULTS ---")
    print(json.dumps(final_results, indent=4))