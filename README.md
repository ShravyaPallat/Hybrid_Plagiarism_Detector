# Hybrid Multimodal Plagiarism Detection System

Implementation of the system described in the paper:
**"Hybrid Multimodal Plagiarism Detection System Integrating Lexical–Semantic Analysis with OCR-Based Image Processing"**

---
## Features
- **TF-IDF Similarity** for direct text matching
- **Sentence-BERT Similarity** for semantic/paraphrased content detection
- **OCR Support** for image files using Tesseract
- **PDF Text Extraction**
- **Hybrid Weighted Scoring**
- **Interactive Streamlit UI**
- **Downloadable Reports**
- **Live Demo: https://hybridplagiarismdetector.streamlit.app/**

## Technologies Used
- Python
- Streamlit
- Sentence-Transformers
- PyMuPDF
- Pytesseract
- Pillow

## Supported Input Formats
- `.txt`
- `.pdf`
- `.png`
- `.jpg`
- `.jpeg`

## Project Structure

```text
Hybrid-Plagiarism-Detector/
│── app.py
│── plagiarism_detector.py
│── requirements.txt
│── README.md
## Architecture

```

---

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Tesseract binary (for OCR)
#    macOS:   brew install tesseract
#    Ubuntu:  sudo apt install tesseract-ocr
#    Windows: https://github.com/tesseract-ocr/tesseract
```

---

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.


---

## Modules

| Module | Class | Description |
|---|---|---|
| Input | `InputHandler` | Routes `.txt`, `.pdf`, image files |
| OCR | `OCRModule` | Tesseract wrapper for image→text |
| Preprocessing | `Preprocessor` | Lowercase, stop-word removal, tokenise |
| Lexical | `TFIDFEngine` | TF-IDF vectors + cosine similarity |
| Semantic | `SBERTEngine` | `all-MiniLM-L6-v2` embeddings |
| Aggregation | `SimilarityAggregator` | Weighted hybrid score |
| Report | `ReportGenerator` | Structured output |

---

## Verdict Thresholds

| Score | Verdict |
|---|---|
| ≥ 65% | HIGH — likely plagiarism |
| 35–64% | MODERATE — possible paraphrasing |
| < 35% | LOW — likely original |

---

## Graceful Degradation

- If `sentence-transformers` is not installed → SBERT falls back to TF-IDF score
- If `pytesseract` is not installed → image inputs raise a clear `ImportError`
- If `pymupdf` is not installed → PDF inputs raise a clear `ImportError`
- Text inputs always work with zero extra dependencies
