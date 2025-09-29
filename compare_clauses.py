import os
import json
import re
from difflib import SequenceMatcher

# --- Configuration ---
TARGET_ATTR = "Attr3"
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
        print(f"🛑 Error: {filename} not found. Please create this file in the script's directory and populate it.")
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
        # print(f"Error reading {file_path}: {e}")
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
    Compares two texts and returns a single string of the minimal, 
    non-matching text segments found in either document.
    """
    s = SequenceMatcher(None, standard_text, contract_text)
    
    diff_parts = []
    
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag in ('replace', 'delete', 'insert'):
            # Text that was ONLY in the standard (deleted)
            if tag in ('delete', 'replace') and i2 > i1:
                 diff_parts.append(standard_text[i1:i2].strip())
            # Text that was ONLY in the contract (inserted)
            if tag in ('insert', 'replace') and j2 > j1:
                 diff_parts.append(contract_text[j1:j2].strip())
            
    # Combine all unique, non-empty difference blocks into one string
    return " ".join(sorted(list(set(filter(None, diff_parts))))).lower()

# --- Comparison Rules Implementation ---

def check_for_material_change(standard_text, contract_text):
    """
    Combines and refines the logic for Rule 1 (Conditional Proximity) and 
    Rule 2 (Keyword Presence in Difference Block).
    """
    diff_blocks = get_diff_blocks(standard_text, contract_text)
    
    # Ignore tiny differences (e.g., punctuation/typos) that might contain common words
    # but aren't material changes.
    MIN_DIFF_LENGTH = 5 
    if len(diff_blocks) < MIN_DIFF_LENGTH and not any(re.search(r'\b' + re.escape(kw) + r'\b', diff_blocks) for kw in KEYWORDS):
         return False, ""
    
    # Check if any keyword or conditional word is present in the difference block
    has_keyword_in_diff = any(re.search(r'\b' + re.escape(kw) + r'\b', diff_blocks) for kw in KEYWORDS)
    has_conditional_in_diff = any(re.search(r'\b' + re.escape(cw) + r'\b', diff_blocks) for cw in CONDITION_WORDS)

    # 1. Check for Rule 1 Trigger (Conditional + Keyword in Difference Block)
    if has_keyword_in_diff and has_conditional_in_diff:
        # If the change block is only a few words and contains both, it's a critical, conditional change.
        if len(diff_blocks.split()) < 5: 
             return True, f"Rule 1 triggered: Substantial Conditional modification detected in difference block: '{diff_blocks}'"

    # 2. Check for Rule 2 Trigger (Keyword Presence in Difference Block)
    if has_keyword_in_diff:
        # Safely find the specific keyword to include in the reason (Fix for StopIteration)
        found_keyword = None
        for kw in KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', diff_blocks):
                found_keyword = kw
                break
        
        if found_keyword:
            return True, f"Rule 2 triggered: Difference block contains a core keyword: '{found_keyword}'"
        else:
            # Fallback (should not happen after the safety check)
            return True, "Rule 2 triggered: Difference block contains a core keyword (unable to identify the exact keyword)."

    return False, ""

import string

def normalize_word(word: str) -> str:
    """Normalize words (lowercase + strip punctuation)."""
    return word.lower().strip(string.punctuation)

def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev_row = range(len(b) + 1)
    for i, ca in enumerate(a, 1):
        curr_row = [i]
        for j, cb in enumerate(b, 1):
            insertions = prev_row[j] + 1
            deletions = curr_row[j - 1] + 1
            substitutions = prev_row[j - 1] + (ca != cb)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def is_similar_word(w: str, target: str, tolerance: int = 1) -> bool:
    """Check if w is similar to target within edit distance tolerance."""
    return levenshtein(w, target) <= tolerance

def load_wordlist(filepath: str, normalize=True):
    """Load words/phrases from a file."""
    with open(filepath, "r") as f:
        words = [line.strip() for line in f if line.strip()]
    return [normalize_word(w) if normalize else w for w in words]

def filter_text(text: str, keywords, condition_words, tolerance=1):
    """Return filtered text keeping only keyword-matching or condition words."""
    words = text.split()
    filtered = []
    for w in words:
        nw = normalize_word(w)

        # If exact condition word, keep
        if nw in condition_words:
            filtered.append(nw)
            continue

        # If close to any keyword, keep
        if any(is_similar_word(nw, kw, tolerance) for kw in keywords):
            filtered.append(nw)
            continue

    return " ".join(filtered)

def compare_attr3_texts(standard_text, contract_text,
                        keywords_file="keywords.txt",
                        condition_file="condition_words.txt",
                        tolerance=1):
    """
    Compare by filtering texts to only relevant words (keywords + condition words).
    """
    if not standard_text or not contract_text:
        return "0", "Missing content for comparison."

    # Load word lists
    keywords = load_wordlist(keywords_file, normalize=True)
    condition_words = set(load_wordlist(condition_file, normalize=True))

    # Build reduced versions
    std_filtered = filter_text(standard_text, keywords, condition_words, tolerance)
    con_filtered = filter_text(contract_text, keywords, condition_words, tolerance)

    if std_filtered == con_filtered:
        return "1", f"Similar (filtered texts match)\nSTD: {std_filtered}\nCON: {con_filtered}"
    else:
        return "0", f"Dissimilar\nSTD: {std_filtered}\nCON: {con_filtered}"


# --- Main Processing Function ---

def process_all_folders(root_dir):
    """
    Iterates through all state folders in the root directory and performs comparison.
    """
    all_results = {}
    
    if not os.path.exists(root_dir):
        print(f"🛑 Error: Root directory '{root_dir}' not found.")
        return all_results

    # Iterate through all items in the Output root directory
    for item_name in os.listdir(root_dir):
        state_folder = os.path.join(root_dir, item_name)
        
        if os.path.isdir(state_folder):
            print(f"\n===========================================================")
            print(f"      ➡️ Processing and Comparing Folder: {item_name}")
            print(f"===========================================================")
            
            standard_text = None
            contract_data = {}
            
            # 1. First Pass: Extract all Attr3 texts
            for filename in os.listdir(state_folder):
                file_path = os.path.join(state_folder, filename)
                
                if filename.endswith('.json'):
                    text = extract_and_combine_text(file_path, TARGET_ATTR)
                    if text is None:
                        continue

                    if 'standard_template' in filename.lower():
                        standard_text = text
                    elif '_contract' in filename.lower():
                        # Extract contract number, e.g., 'Contract1'
                        try:
                            parts = filename.lower().split('_')
                            contract_part = next(p for p in parts if 'contract' in p)
                            contract_num = re.search(r'contract(\d+)', contract_part).group(1)
                            contract_name = f"Contract{contract_num}"
                            contract_data[contract_name] = text
                        except:
                            continue

            if standard_text is None:
                print(f"🛑 ERROR: Standard template not found in {state_folder} or failed to process. Skipping state.")
                continue

            # 2. Second Pass: Comparison
            state_results = {}
            sorted_contracts = sorted(contract_data.keys())
            
            print(f"Standard Template (Attr3) Text Length: {len(standard_text)} chars")
            print("-" * 70)
            
            for contract_name in sorted_contracts:
                contract_text = contract_data[contract_name]
                
                # Comparison function call
                similarity_score, reason = compare_attr3_texts(standard_text, contract_text)
                
                state_results[contract_name] = {
                    "Similarity_Score": similarity_score,
                    "Reason": reason
                }
                
                print(f"| {contract_name:<12} | Score: {similarity_score:<1} | Classification: {'✅ Similar' if similarity_score == '1' else '❌ Non-Standard'}")
                print(f"| {'':<12} | Reason: {reason}")
                print("-" * 70)
            
            all_results[item_name] = state_results

    return all_results

# --- Execute Script ---
if __name__ == "__main__":
    final_results = process_all_folders(OUTPUT_ROOT_DIR)
    
    print("\n\n--- FINAL SUMMARY OF COMPARISON RESULTS ---")
    print(json.dumps(final_results, indent=4))