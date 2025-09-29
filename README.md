## HiLabs Hackathon: Smart Contract Language Tagging for Negotiation Assist

  

GitHub Link: https://github.com/br0hit/HiLabs
Video Explanation: https://drive.google.com/drive/folders/1adDufhRMvmXdm9CazGmyl69jrtAqKF7Z?usp=sharing


## Results

The resultsare present in the dashboard/data.csv section
1 - Standard
0 - Non standard 

Total Standard obtained: 28
Total Non Standard obtained: 22

## Requirements

  

1. Docker

2. Empty port 5000 to run the flask application

  

## Setup

  

First we will run the docker container to extract the text from the pdf for our attributes
Since some of the pds have images, this step is crucial for us to extract the necessary information related to the text 


  

docker compose up --build 

  
  

Once we have extracted the attribute information, we will use this to classify the docs

  

python compare_clauses.py

  
  
  
  

### Setting up flask:

  

We have developed a dashboard to show the results obtained, we can visualize it by running the following code:

  

python app.py

---

## Attributes and Descriptions

1. **Medicaid Timely Filing**  
   This provision relates to proper claims submission guidelines and timelines specific to Medicaid programs.

2. **Medicare Timely Filing**  
   This provision relates to proper claims submission guidelines and timelines specific to Medicare programs.

3. **No Steerage / SOC**  
   Provision allows Elevance the right to develop provider networks without requiring specific steerage or shared obligation clauses.

4. **Medicaid Fee Schedule**  
   Specifies the methodology for calculating the reimbursement rate for Medicaid-covered services.

5. **Medicare Fee Schedule**  
   Specifies the methodology for calculating the reimbursement rate for Medicare-covered services.

---
## Approach and Architectural Decisions

The solution is divided into two phases: **Clause Extraction** and **Clause Classification**.  
Each attribute follows a specific methodology, leveraging a mix of **OCR-based extraction** and **rule/semantic-based classification**.

---

### 1. Clause Extraction for Attributes 1,2 & 3

1. **Convert PDF to Images**  
   - Each contract PDF is split into page images using `pdf2image` (300 DPI).  

2. **Preprocessing and Redaction Handling**  
   - Pages are binarized using adaptive thresholding.  
   - Redacted areas are detected with contour analysis and replaced with white patches to simplify OCR.  

3. **Section Detection**  
   - Sections are identified by regex on phrases.  
   - Bounding boxes are stored in metadata files, linking each section to page images.  

4. **OCR and Clause Cropping**  
   - Clauses are cropped by bounding box.  
   - Tesseract OCR extracts clean text.  
   - Redacted content is marked as `[REDACTED]`.  

5. **Handling Continuations**  
   - Clauses spanning multiple pages are merged.  

6. **Output**  
   - Extracted clauses are saved as JSON per attribute, per contract, with metadata (section, bounding box, page).

---

### 3. Extraction of Fee Schedule Tables (Attributes 4 & 5)

For attributes involving **Fee Schedules** where clauses are represented as tables:

1. **Keyword Anchoring**  
   - Identify page numbers containing the keyword **"Service Description"**.  

2. **Table Continuation Handling**  
   - Detect and merge tables spanning multiple pages.  

3. **Table Detection and Cropping**  
   - Use **OpenCV contour detection** to isolate table regions from the page images.  

4. **OCR to CSV**  
   - Run Tesseract OCR on the cropped tables.  
   - Convert extracted text into structured **CSV files** for downstream analysis.  

---

### 2. Clause Classification (Attribute-Specific Rules)

#### Attribute: Medicaid Timely Filing & Medicare Timely Filing
- Clean and normalize text (spell-check, convert numbers, strip punctuation).  
- Mask all numeric values with `<NUM>`.  
- Compare extracted clauses with state standard template using **Legal-BERT embeddings** and cosine similarity.  
- Classification:  
  - **Standard** if similarity ≥ 0.99.  
  - **Non-Standard** otherwise.  

#### Attribute: No Steerage / SOC
- Extract information from **Section 2.11** of contracts.  
- Compare contract text with the standard template section.  
- Rules:  
  - If the clause introduces **extra keywords** (defined from Article I: Definitions), or  
  - If the clause omits/adds **conditional words** (e.g., "unless", "except") within a ±2–3 word window around these keywords,  
  - → Marked as **Non-Standard**.  
- Otherwise → **Standard**.  

#### Attribute: Medicaid Fee Schedule
- **Tennessee (TN)**:  
  - Clause must state: reimbursement shall be a percentage of the *Professional Provider Market Master Fee Schedule* in effect on the date of service.  
  - Acceptable variants include CMS or Medicare Fee Schedules.  
  - If clause diverges (different methodology, exceptions, or carve-outs), it is **Non-Standard**.  

- **Washington (WA)**:  
  - Standard clause text is minimal.  
  - Any mention of reimbursement based on *Medicaid Fee Schedules* (Equipment/Drugs) is considered **Standard**.  
  - Divergence → **Non-Standard**.  

#### Attribute: Medicare Fee Schedule
- Similar methodology to Medicaid Fee Schedule.  
- Clause is considered **Standard** if reimbursement references Medicare Fee Schedules (Equipment/Drugs).  
- Divergent methodologies or conditions → **Non-Standard**.  

---



---

### 4. Architectural Decisions

- **OCR-first design**: Contracts are scanned images; OCR is essential.  
- **Hybrid classification**:  
  - Semantic similarity (Legal-BERT) for flexible clauses.  
  - Rule-based detection for critical attributes like Steerage/SOC.  
- **Bounding box–driven pipeline**: Extracts only relevant sections, reducing OCR noise.  
- **Attribute modularity**: Attribute logic and keyword lists live in external configs (`config.json`, `keywords.txt`, `condition_words.txt`).  
- **Transparency and auditability**: Each classification includes both a binary label and reasoning.  
- **Scalable design**: New attributes or rules can be added with minimal pipeline changes.  
