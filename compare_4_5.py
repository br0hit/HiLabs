import os
import re
INPUT_DIR = "output"
MEDICARE_ADVANTAGE_KEYWORDS = ["Medicare"]
MEDICAID_KEYWORDS = ["CMS", "Professional Market Master Fee Schedule", "Medicaid"]
number_pattern = re.compile(r"\d+(\.\d+)?%?|\$\d+(\.\d+)?")

def classify_medicare_advantage(lines):
    is_standard = True  # Assume standard until proven otherwise
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if the line contains a number, percentage, or currency
        if number_pattern.search(line):
            # If it does, it must also contain a keyword to be standard
            if not any(keyword in line for keyword in MEDICARE_ADVANTAGE_KEYWORDS):
                is_standard = False
                break  # Found a non-standard line, no need to check further
    return "Standard" if is_standard else "Non-Standard"

def classify_medicaid(lines):
    is_standard = True  # Assume standard until proven otherwise
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if the line contains a number, percentage, or currency
        if number_pattern.search(line):
            # If it does, it must also contain a keyword to be standard
            if not any(keyword in line for keyword in MEDICAID_KEYWORDS):
                is_standard = False
                break  # Found a non-standard line, no need to check further
    return "Standard" if is_standard else "Non-Standard"

def main():
    """
    Main function to read files, parse sections, and classify contracts.
    """
    # Create the directory if it doesn't exist to avoid errors
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        return

    for file_name in os.listdir(INPUT_DIR):
        if file_name.endswith(".txt"):
            file_path = os.path.join(INPUT_DIR, file_name)
            contract_name = os.path.splitext(file_name)[0]

            # Containers for lines from each section
            medicare_advantage_lines = []
            medicaid_lines = []
            current_section = None

            # Read the file and separate lines into sections
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "==== Medicare Advantage ====" in line:
                            current_section = "medicare"
                        elif "==== Medicaid ====" in line:
                            current_section = "medicaid"
                        elif current_section:
                            if current_section == "medicare":
                                medicare_advantage_lines.append(line)
                            elif current_section == "medicaid":
                                medicaid_lines.append(line)
            except IOError as e:
                print(f"Could not read file {file_name}: {e}")
                continue

            # Classify each section
            medicare_status = "Standard"
            if medicare_advantage_lines:
                medicare_status = classify_medicare_advantage(medicare_advantage_lines)

            medicaid_status = "Standard"
            if medicaid_lines:
                medicaid_status = classify_medicaid(medicaid_lines)
            
            # A contract is Non-Standard if any of its sections are Non-Standard
            final_status = "Standard"
            if medicare_status == "Non-Standard" or medicaid_status == "Non-Standard":
                final_status = "Non-Standard"

            print(f"{contract_name}: {medicaid_status}")

if _name_ == "_main_":
    main()