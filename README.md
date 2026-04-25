# Hybrid Multimodal Plagiarism Detection System

Implementation of the system described in the paper:
**"Hybrid Multimodal Plagiarism Detection System Integrating Lexical–Semantic Analysis with OCR-Based Image Processing"**

---

## Architecture

```
Input (text / image / PDF)
        │
        ▼
┌─────────────────┐
│  InputHandler   │  Detects file type
└────────┬────────┘
         │ image?
         ▼
┌─────────────────┐
│   OCRModule     │  Tesseract → plain text
└────────┬────────┘
         ▼
┌─────────────────┐
│  Preprocessor   │  lowercase, stop-words, tokenise
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│TF-IDF  │ │ Sentence-BERT│
│Engine  │ │ Engine       │
└───┬────┘ └──────┬───────┘
    │              │
    └──────┬───────┘
           ▼
┌──────────────────────┐
│ SimilarityAggregator │  40% TF-IDF + 60% SBERT
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  ReportGenerator     │
└──────────────────────┘
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

### Command-line

```bash
# Text files
python plagiarism_detector.py submission.txt reference.txt

# Image / scanned document
python plagiarism_detector.py scan.png reference.txt

# PDF
python plagiarism_detector.py submission.pdf reference.pdf

# Raw strings
python plagiarism_detector.py "Students often copy text." "Text is frequently copied."

# Custom weights
python plagiarism_detector.py sub.txt ref.txt --lexical-weight 0.3 --semantic-weight 0.7
```

### Streamlit UI

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Python API

```python
from plagiarism_detector import PlagiarismDetector

detector = PlagiarismDetector(lexical_weight=0.4, semantic_weight=0.6)

result = detector.detect("submission.txt", "reference.txt")

print(result["tfidf_score"])   # e.g. 0.72
print(result["sbert_score"])   # e.g. 0.84
print(result["hybrid_score"])  # e.g. 0.79
print(result["verdict"])       # e.g. "HIGH — likely plagiarism"
print(result["common_phrases"])
```

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
