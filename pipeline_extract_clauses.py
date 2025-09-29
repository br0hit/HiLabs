# run_pipeline.py

import re
import os
import time
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import json
import shutil
import argparse
from pathlib import Path

# --- Configuration ---

# NOTE: Update this path if Tesseract is installed elsewhere
# On Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = r"venv\Tesseract-OCR\tesseract.exe"
# On Linux/macOS (if in PATH, you might not need this line)
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# Tesseract configuration
TESSERACT_CONFIG = r'--oem 3 --psm 4'
DPI = 300
FOOTER_CUTOFF_RATIO = 0.10  # remove bottom 10%


# Regex for sections
SECTION_PATTERN = re.compile(r'^(?:Section\s*)?(\d+(?:\.\d+)*)(?:[:\.\)]?)$', re.IGNORECASE)


# --- Part 1: Section & Bounding Box Detection (from sec_bbox.py) ---

def preprocess_for_ocr(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 15)

def detect_and_clean_redactions_for_sections(cv_img, min_area=300):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 12)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img, _ = cv_img.shape
    cleaned = cv_img.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area or area > 0.9 * (w_img * h_img):
            continue
        roi = gray[y:y + h, x:x + w]
        if roi.size > 0 and max(w / (h+1e-6), h / (w+1e-6)) > 2:
            if roi.mean() < 60 and roi.std() < 30:
                cv2.rectangle(cleaned, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return cleaned

def process_page_for_sections(cv_img, page_num, carryover):
    h_img, _, _ = cv_img.shape
    cleaned_img = detect_and_clean_redactions_for_sections(cv_img)
    gray = preprocess_for_ocr(cleaned_img)
    df = pytesseract.image_to_data(gray, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna(subset=['text'])
    
    # Force the 'text' column to be a string type to prevent errors on blank pages
    df['text'] = df['text'].astype(str)

    df = df[df['text'].str.strip() != ""].copy()
    df[['left', 'top', 'width', 'height', 'conf']] = df[['left', 'top', 'width', 'height', 'conf']].astype(int)
    df = df[df['conf'] > 35]
    df = df[df['top'] < int((1 - FOOTER_CUTOFF_RATIO) * h_img)]

    lines = []
    for (block, par, line), g in df.groupby(['block_num', 'par_num', 'line_num']):
        g = g.sort_values('left')
        text = " ".join(g['text'].tolist())
        x1, y1 = g['left'].min(), g['top'].min()
        x2, y2 = (g['left'] + g['width']).max(), (g['top'] + g['height']).max()
        lines.append([block, x1, y1, x2, y2, text])

    merged = []
    for ln in sorted(lines, key=lambda r: (r[0], r[2])):
        if not merged:
            merged.append(ln)
            continue
        prev = merged[-1]
        if ln[0] == prev[0] and (ln[2] - prev[4]) < 25 and abs(ln[1] - prev[1]) < 80:
            prev[1] = min(prev[1], ln[1])
            prev[2] = min(prev[2], ln[2])
            prev[3] = max(prev[3], ln[3])
            prev[4] = max(prev[4], ln[4])
            prev[5] += " " + ln[5]
        else:
            merged.append(ln)

    current = carryover or {"id": None, "text": [], "bbox": None, "continued": False}
    if carryover and carryover.get("id"):
        current["bbox"] = None
        current["continued"] = True

    page_sections = []
    for _, x1, y1, x2, y2, text in merged:
        words = text.split()
        if not words: continue
        m = SECTION_PATTERN.match(words[0]) or SECTION_PATTERN.match(re.sub(r'[^\w\.]', '', words[0]))

        if m:
            if current.get("id"):
                page_sections.append({
                    "id": current["id"], "label": " ".join(" ".join(current["text"]).split(".")[0:1]),
                    "bbox": current["bbox"], "page": page_num,
                    "continued_from": current.get("continued", False), "continues": True
                })
            current = {"id": m.group(1), "text": [" ".join(words[1:])], "bbox": [x1, y1, x2, y2], "continued": False}
        elif current.get("id"):
            current["text"].append(text)
            if current["bbox"]:
                current["bbox"] = [min(current["bbox"][0], x1), min(current["bbox"][1], y1), max(current["bbox"][2], x2), max(current["bbox"][3], y2)]
            else:
                current["bbox"] = [x1, y1, x2, y2]

    if current.get("id"):
        page_sections.append({
            "id": current["id"], "label": " ".join(" ".join(current["text"]).split(".")[0:1]),
            "bbox": current["bbox"], "page": page_num,
            "continued_from": current.get("continued", False), "continues": True
        })
        current["continues"] = True

    return page_sections, current

def run_section_detection(pdf_path, output_bbox_dir, output_text_file):
    """Orchestrates Part 1: Detects all section bounding boxes in a PDF."""
    os.makedirs(output_bbox_dir, exist_ok=True)
    
    all_page_sections = []
    carryover = None
    page_num = 1
    total_pages = 0 # To track the total number of processed pages

    # 💡 CRUCIAL MODIFICATION: Loop to process one page at a time
    while True:
        try:
            # Load ONLY the current page (page_num) from the PDF
            # Returns a list that is either empty or contains one PIL image
            pil_pages = convert_from_path(
                pdf_path, 
                dpi=DPI, 
                first_page=page_num, 
                last_page=page_num # Only process this single page
            )
        except Exception as e:
            # Handle error during PDF conversion (e.g., corrupt file)
            print(f"  ERROR: Could not convert page {page_num} of PDF. Details: {e}")
            break

        # Check if pdf2image returned an image. If not, we have processed the last page.
        if not pil_pages:
            break
        
        pil_page = pil_pages[0] # Get the single PIL image
        print(f"  Processing page {page_num}...") 

        # --- Page Processing Logic ---
        try:
            cv_img = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)
            
            # 💡 MANUAL MEMORY RELEASE: Delete the large PIL image after conversion
            del pil_page 

            page_sections, carryover = process_page_for_sections(cv_img, page_num, carryover)
            all_page_sections.extend(page_sections)

            # Create and save visualization image
            vis = cv_img.copy()
            for sec in page_sections:
                if sec.get("bbox"):
                    x1, y1, x2, y2 = sec["bbox"]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
            out_path = os.path.join(output_bbox_dir, f"page_{page_num:03d}.png")
            cv2.imwrite(out_path, vis)
            
            # 💡 MANUAL MEMORY RELEASE: Delete the large OpenCV images
            del cv_img
            del vis

        except Exception as page_error:
            print(f"  Warning: Failed to process page {page_num} of {pdf_path.name}. Skipping page. Error: {page_error}")
            
        total_pages += 1
        page_num += 1 # Increment to load the next page

    if total_pages > 0:
        print(f"  Converted PDF to {total_pages} images.")
        
        # Write metadata text file (Logic remains the same)
        with open(output_text_file, "w", encoding="utf-8") as f:
            for sec in all_page_sections:
                # ... (writing metadata content remains the same)
                f.write(f"Section {sec['id']}\n")
                f.write(f"  Label: {sec.get('label', '')[:60]}\n")
                f.write(f"  Image: page_{sec['page']:03d}.png\n")
                
                bbox = sec.get('bbox')
                if bbox:
                    # Explicitly convert numpy types to standard integers for clean output
                    clean_bbox = [int(coord) for coord in bbox]
                    f.write(f"  BBox: {clean_bbox}\n")
                else:
                    f.write(f"  BBox: None\n")

                if sec.get("continued_from"): f.write("  (continued from previous page)\n")
                if sec.get("continues"): f.write("  (continues to next page)\n")
                f.write("\n")
                
        return True, all_page_sections
    else:
        # File was empty or conversion failed entirely
        print(f"  No pages were successfully processed for {pdf_path.name}.")
        return False, None

# --- Part 2: Clause Extraction (from extract_clauses.py) ---

def parse_metadata_file(filepath):
    sections = []
    current = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("Section"):
                if current: sections.append(current)
                current = {"section": line.split()[1]}
            elif line.startswith("Label:"):
                current["label"] = line.replace("Label:", "").strip()
            elif line.startswith("Image:"):
                current["image"] = line.replace("Image:", "").strip()
            elif line.startswith("BBox:"):
                bbox_str = line.replace("BBox:", "").strip()
                if bbox_str != 'None':
                    current["bbox"] = [int(x) for x in re.findall(r"\d+", bbox_str)]
        if current: sections.append(current)
    return sections

def find_section_by_phrase(phrase, metadata):
    for i, s in enumerate(metadata):
        if phrase.lower() in s.get("label", "").lower():
            return i, s["section"]
    return None, None

def collect_adjacent_sections(idx, section_number, metadata):
    collected = [metadata[idx]]
    i = idx + 1
    while i < len(metadata) and metadata[i]["section"] == section_number:
        collected.append(metadata[i])
        i += 1
    i = idx - 1
    while i >= 0 and metadata[i]["section"] == section_number:
        collected.insert(0, metadata[i])
        i -= 1
    return collected

def detect_redactions_for_clauses(image, min_area=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(cv2.bitwise_not(gray), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 15)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h_img, w_img = gray.shape
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area and w * h < 0.98 * h_img * w_img:
            boxes.append((x, y, x + w, y + h))
    return boxes

def extract_clause(metadata, images_folder, phrase):
    idx, section_number = find_section_by_phrase(phrase, metadata)
    if section_number is None:
        return {"error": f"No section found for phrase '{phrase}'"}
    
    section_entries = collect_adjacent_sections(idx, section_number, metadata)
    results = []
    for s in section_entries:
        bbox = s.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        
        img_path = os.path.join(images_folder, s["image"])
        if not os.path.exists(img_path):
            results.append({"error": f"Image file not found: {s['image']}"})
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape
        x1, y1, x2, y2 = bbox
        crop = img[max(0, y1 - 5):min(h, y2 + 5), max(0, x1 - 5):min(w, x2 + 5)]

        red_boxes = detect_redactions_for_clauses(crop)
        for (rx1, ry1, rx2, ry2) in red_boxes:
            cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (255, 255, 255), -1)
            cv2.putText(crop, "[REDACTED]", (rx1 + 5, ry1 + (ry2 - ry1) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        text = pytesseract.image_to_string(crop)
        results.append({"section": s.get("section"), "image": s.get("image"), "bbox": bbox, "text": text.strip()})

    return results if results else {"error": f"No valid section data for phrase '{phrase}'"}

def run_clause_extraction(sections_metadata_file, images_folder, output_json_path, attributes_to_extract):
    """Orchestrates Part 2: Extracts clauses based on metadata and images."""
    metadata = parse_metadata_file(sections_metadata_file)
    output_data = {}

    for attr, phrases in attributes_to_extract.items():
        print(f"  Processing attribute: {attr}")
        output_data[attr] = {}
        for phrase in phrases:
            result = extract_clause(metadata, images_folder, phrase)
            output_data[attr][phrase] = result

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"  JSON output written to {output_json_path}")
    return True
# --- Main Pipeline Orchestrator ---

import fitz  # PyMuPDF
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import os
import csv
from difflib import SequenceMatcher
import tempfile # Added for temporary directory management
import shutil # Added to remove the temp directory


OUTPUT_ROOT = "Output"
SEARCH_TERM = "Service Description"
SEARCH_TOKENS = SEARCH_TERM.split()

# --- HELPER: Fuzzy match ---
def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# --- HELPER: Highlight pages ---
def highlight_occurrences(pdf_path, search_tokens, temp_dir):
    doc = fitz.open(pdf_path)
    found_pages = []

    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).copy()
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        found = False
        i = 0
        while i < len(data["text"]):
            word = data["text"][i].strip()
            if not word:
                i += 1
                continue
            match = True
            for j, token in enumerate(search_tokens):
                if (i + j >= len(data["text"]) or
                        data["text"][i + j].strip().lower() != token.lower()):
                    match = False
                    break
            if match:
                found = True
                i += len(search_tokens)
            else:
                i += 1

        if found:
            found_pages.append(page_num + 1)
            # Add next 3 context pages
            for extra_page in range(1, 4):
                next_page = page_num + extra_page
                if next_page < len(doc):
                    found_pages.append(next_page + 1)
                    
    return sorted(set(found_pages))

# --- HELPER: Table detection & OCR ---
def detect_and_crop_tables(image_path, page_num, temp_dir):
    img = cv2.imread(image_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 15, -2)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                                         horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                                       vertical_kernel, iterations=2)
    table_mask = cv2.add(detect_horizontal, detect_vertical)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    cropped_table_paths = []
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 150 and h > 150:
            table_roi = img[y:y + h, x:x + w]
            base_name = os.path.basename(image_path)
            table_img_path = os.path.join(temp_dir, base_name.replace(".png", f"_table_{i+1}.png"))
            cv2.imwrite(table_img_path, table_roi)
            cropped_table_paths.append(table_img_path)
    return cropped_table_paths

def extract_table_to_matrix_accurate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return []
    original_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 15, -2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                     cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 1000]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])
    rows, table_matrix = [], []
    if bounding_boxes:
        current_row = [bounding_boxes[0]]
        for box in bounding_boxes[1:]:
            if abs(box[1] - current_row[0][1]) < 20:
                current_row.append(box)
            else:
                rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [box]
        rows.append(sorted(current_row, key=lambda b: b[0]))
    for row_boxes in rows:
        row_texts = []
        for x, y, w, h in row_boxes:
            cell_img = original_img[y:y + h, x:x + w]
            cell_img_bordered = cv2.copyMakeBorder(cell_img, 10, 10, 10, 10,
                                                   cv2.BORDER_CONSTANT, value=[255, 255, 255])
            text = pytesseract.image_to_string(cell_img_bordered, config='--oem 1 --psm 6').strip()
            row_texts.append(text.replace("\n", " ").replace("  ", " "))
        table_matrix.append(row_texts)
    return table_matrix

def save_matrix_to_csv(table_matrix, output_csv_path):
    if not table_matrix:
        return
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar='\\')
        for row in table_matrix:
            modified_row = [cell.strip('"') + "  ####" for cell in row]

            writer.writerow(modified_row)

# --- Program detection (UPDATED) ---
def detect_program_from_table(table_matrix):
    """
    Detects the program type based on keywords in the table.
    If 'medicare' is found, it's Medicare Advantage. Otherwise, it's Medicaid.
    """
    table_text = " ".join(" ".join(row) for row in table_matrix).lower()
    if "medicare" in table_text:
        return "Medicare Advantage"
    else:
        return "Medicaid"

# --- Extract Rate/Methodology ---
def extract_rate_methodology_from_csv(input_filename, output_filename, current_program):
    extracted_rates = []
    delimiter = "####,"
    target_header = "Rate/Methodology"
    similarity_threshold = 0.5
    rate_column_index, header_row_index = None, -1
    with open(input_filename, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        for i, line in enumerate(lines):
            headers = [h.strip().replace("####", "") for h in line.strip().split(delimiter)]
            for j, header in enumerate(headers):
                if similar(target_header, header) >= similarity_threshold:
                    rate_column_index = j
                    header_row_index = i
                    break
            if rate_column_index is not None:
                break
        if rate_column_index is None:
            rate_column_index, header_row_index = 2, 0
        for line in lines[header_row_index + 1:]:
            parts = [p.replace("####", "") for p in line.strip().split(delimiter)]
            if len(parts) > rate_column_index:
                rate_data = parts[rate_column_index].strip().strip('"')
                if rate_data:
                    extracted_rates.append(rate_data)
    if extracted_rates:
        with open(output_filename, 'a', encoding='utf-8') as f:
            f.write(f"\n==== {current_program} ====\n")
            for rate in extracted_rates:
                f.write(rate + '\n')

# --- MAIN: Folder pipeline (UPDATED) ---
def process_pdf(pdf_path, output_dir):
    temp_dir = tempfile.mkdtemp()
    
    try:
        found_pages = highlight_occurrences(pdf_path, SEARCH_TOKENS, temp_dir)
        if not found_pages:
            print(f"No relevant pages found in {pdf_path}")
            return

        doc = fitz.open(pdf_path)
        page_image_paths = []
        for i in found_pages:
            if i - 1 >= len(doc):
                continue
            pix = doc[i - 1].get_pixmap(dpi=300)
            img_path = os.path.join(temp_dir, f"{os.path.basename(pdf_path).replace('.pdf','')}_page_{i}.png")
            pix.save(img_path)
            page_image_paths.append((i, img_path))
        doc.close()

        txt_output_path = os.path.join(output_dir, f"{os.path.basename(pdf_path).replace('.pdf','')}_extracted_rates.txt")
        if os.path.exists(txt_output_path):
            os.remove(txt_output_path)

        for page_num, img_path in page_image_paths:
            cropped_tables = detect_and_crop_tables(img_path, page_num, temp_dir)
            for table_path in cropped_tables:
                table_data_matrix = extract_table_to_matrix_accurate(table_path)
                if not table_data_matrix:
                    continue

                # Detect the program for THIS specific table
                current_program = detect_program_from_table(table_data_matrix)

                csv_path = table_path.replace(".png", ".csv")
                save_matrix_to_csv(table_data_matrix, csv_path)
                
                if os.path.exists(csv_path):
                    # Use the program detected for the current table
                    extract_rate_methodology_from_csv(csv_path, txt_output_path, current_program)

        print(f"✅ Processed {pdf_path}. Output: {txt_output_path}")

    finally:
        shutil.rmtree(temp_dir)


# --- Main Pipeline Orchestrator ---
def main(data_folder, output_folder, cleanup):
    start_time = time.time()
    
    # --- Load configuration ---
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            attributes_to_extract = config["attributes_to_extract"]
        print("Successfully loaded configuration from config.json")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not load configuration from config.json. Details: {e}")
        return

    data_dir = Path(data_folder)
    output_dir = Path(output_folder)
    
    if not data_dir.is_dir():
        print(f"Error: Data directory not found at '{data_dir}'")
        return

    output_dir.mkdir(exist_ok=True)
    
    # 💡 MODIFIED: Collects all PDFs one level deep (e.g., Data/TN/*.pdf, Data/WA/*.pdf)
    # This assumes 'Data' is the root and TN, WA are the direct subfolders.
    pdf_files = list(data_dir.glob("*/[a-zA-Z0-9]*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in immediate subdirectories of {data_dir}.")
        return
        
    print(f"Found {len(pdf_files)} PDF(s) to process.")
    processed_count = 0
    skipped_count = 0

    for pdf_path in pdf_files:
        print(f"\n--- Processing file: {pdf_path.name} ---")
        base_name = pdf_path.stem
        
        # 💡 MODIFIED: Determine the output directory based on the immediate parent folder name (e.g., 'TN' or 'WA')
        # This creates Output/TN or Output/WA
        final_output_dir = output_dir / pdf_path.parent.name
        
        intermediate_dir = output_dir / f"{base_name}_intermediate" # Temp files still go to root Output/
        img_dir = intermediate_dir / "images"
        metadata_txt = intermediate_dir / f"{base_name}_sections.txt"
        
        # 💡 FINAL JSON Path: placed inside the correct Output/TN or Output/WA folder
        final_json = final_output_dir / f"{base_name}_clauses.json"
        
        # Ensure the final output directory (Output/TN, Output/WA) exists
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        process_pdf(pdf_path, OUTPUT_ROOT)
        
        # 💡 CHECK: Skip if the final JSON file already exists
        if final_json.exists():
            print(f"  Output already exists at '{final_json.name}'. Skipping.")
            skipped_count += 1
            continue

        try:
            print("Step 1: Detecting sections and bounding boxes...")
            success, _ = run_section_detection(pdf_path, img_dir, metadata_txt)
            if not success:
                # If section detection failed, still clean up and continue
                if intermediate_dir.exists():
                    shutil.rmtree(intermediate_dir)
                continue

            print("Step 2: Extracting clauses...")
            run_clause_extraction(metadata_txt, img_dir, final_json, attributes_to_extract)
            processed_count += 1
            
            # 💡 CLEANUP: Always delete intermediate files (Step 3)
            print(f"Step 3: Cleaning up intermediate files for {pdf_path.name}...")
            try:
                shutil.rmtree(intermediate_dir)
                print(f"  Deleted folder: {intermediate_dir}")
            except OSError as e:
                print(f"  Error deleting {intermediate_dir}: {e}")

        except Exception as e:
            print(f"  An unexpected error occurred while processing {pdf_path.name}: {e}")
            print("  Moving to the next file.")
            
            # Attempt to clean up even on error
            if intermediate_dir.exists():
                 try:
                    shutil.rmtree(intermediate_dir)
                 except OSError:
                    pass
            continue
        

    total_time = time.time() - start_time
    print(f"\n✅ Pipeline finished in {total_time:.2f} seconds.")
    print(f"  - Processed: {processed_count} new file(s)")
    print(f"  - Skipped:   {skipped_count} existing file(s)")
    print(f"All outputs saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process contract PDFs to extract specific clauses.")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="Data",
        help="Path to the main data folder containing subdirectories (e.g., 'TN', 'WA')."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="Output",
        help="Path to the folder where all output JSON files will be saved."
    )
    # NOTE: The 'cleanup' flag is ignored in the new logic as cleanup is now mandatory.
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="This flag is now ignored. Intermediate files are always deleted."
    )
    
    args = parser.parse_args()
    main(args.data_folder, args.output_folder, args.cleanup)