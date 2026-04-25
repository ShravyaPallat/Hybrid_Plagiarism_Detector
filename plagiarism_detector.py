"""
Hybrid Multimodal Plagiarism Detection System
Integrates TF-IDF (lexical), Sentence-BERT (semantic), and OCR (image input)
"""

import re
import math
import argparse
from collections import Counter
from pathlib import Path


# ── Optional heavy dependencies ──────────────────────────────────────────────

try:
    from PIL import Image
    import pytesseract
    import shutil, os
    _tess=(
        shutil.which("tesseract")
        or "/usr/bin/tesseract"
        or  r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    if _tess:
        pytesseract.pytesseract.tesseract_cmd=_tess
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ── Stop words ────────────────────────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "can", "could", "that", "this",
    "these", "those", "it", "its", "not", "as", "if", "so", "up", "out",
    "about", "into", "than", "then", "there", "their", "they", "we", "you",
    "he", "she", "i", "my", "your", "our", "his", "her", "which", "who",
}


# ── 1. Input Handler ──────────────────────────────────────────────────────────

class InputHandler:
    """Detects file type and routes to the correct extraction method."""

    def load(self, source: str) -> str:
        """
        Accept a file path (txt, pdf, png/jpg) or a raw string.
        Returns extracted plain text.
        """
        path = Path(source)

        if not path.exists():
            # Treat as raw text
            return source

        suffix = path.suffix.lower()

        if suffix == ".txt":
            return path.read_text(encoding="utf-8")

        if suffix == ".pdf":
            return self._extract_pdf(path)

        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
            return self._extract_image(path)

        raise ValueError(f"Unsupported file type: {suffix}")

    def _extract_pdf(self, path: Path) -> str:
        if not PDF_AVAILABLE:
            raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        text = "\n".join(pages).strip()
        if len(text) < 50:
            # Likely a scanned PDF — fall back to OCR per page
            print("[InputHandler] Scanned PDF detected — applying OCR per page")
            ocr = OCRModule()
            pages = [ocr.extract_from_pil(page.get_pixmap().pil_image()) for page in doc]
            text = "\n".join(pages)
        return text

    def _extract_image(self, path: Path) -> str:
        ocr = OCRModule()
        return ocr.extract_from_path(str(path))


# ── 2. OCR Module ─────────────────────────────────────────────────────────────

class OCRModule:
    #Wraps Tesseract OCR for image-to-text extraction.

    def extract_from_path(self, image_path: str) -> str:
        if not OCR_AVAILABLE:
            raise ImportError(
            )
        img = Image.open(image_path)
        return self.extract_from_pil(img)

    def extract_from_pil(self, img) -> str:
        text = pytesseract.image_to_string(img, config="--psm 6")
        return text.strip()


# ── 3. Preprocessor ───────────────────────────────────────────────────────────

class Preprocessor:
    #Normalises text: lowercase, remove punctuation, stop-word filtering, tokenise.

    def process(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
        return tokens

    def process_to_str(self, text: str) -> str:
        return " ".join(self.process(text))


# ── 4. TF-IDF Lexical Similarity Engine ──────────────────────────────────────

class TFIDFEngine:
    #Computes TF-IDF vectors for a two-document corpus and returns cosine similarity.
   
    def _tf(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = len(tokens) or 1
        return {word: count / total for word, count in counts.items()}

    def _idf(self, word: str, docs: list[list[str]]) -> float:
        n_containing = sum(1 for doc in docs if word in doc)
        return math.log((len(docs) + 1) / (n_containing + 1)) + 1

    def vectorise(self, tokens: list[str], vocab: list[str], docs: list[list[str]]) -> list[float]:
        tf = self._tf(tokens)
        return [tf.get(w, 0.0) * self._idf(w, docs) for w in vocab]

    def cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a ** 2 for a in v1))
        mag2 = math.sqrt(sum(b ** 2 for b in v2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def compute(self, tokens1: list[str], tokens2: list[str]) -> float:
        vocab = list(set(tokens1) | set(tokens2))
        docs = [tokens1, tokens2]
        v1 = self.vectorise(tokens1, vocab, docs)
        v2 = self.vectorise(tokens2, vocab, docs)
        return round(self.cosine_similarity(v1, v2), 4)

    def common_phrases(self, text1: str, text2: str, n: int = 3) -> list[str]:
        #Return n-gram phrases that appear in both texts.
        def ngrams(text, n):
            words = text.lower().split()
            return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

        matches = []
        for size in (4, 3):
            matches += list(ngrams(text1, size) & ngrams(text2, size))
        return list(dict.fromkeys(matches))[:n]  # deduplicate, keep order


# ── 5. Sentence-BERT Semantic Similarity Engine ───────────────────────────────

class SBERTEngine:
    
    #Loads a Sentence-BERT model and computes semantic cosine similarity between two text strings.

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            if not SBERT_AVAILABLE:
                raise ImportError(
                    "sentence-transformers not installed.\n"
                    "Run: pip install sentence-transformers"
                )
            print(f"[SBERT] Loading model '{self.MODEL_NAME}'…")
            self._model = SentenceTransformer(self.MODEL_NAME)

    def compute(self, text1: str, text2: str) -> float:
        self._load()
        emb1, emb2 = self._model.encode([text1, text2], convert_to_tensor=True)
        score = float(util.cos_sim(emb1, emb2)[0][0])
        return round(max(0.0, min(1.0, score)), 4)


# ── 6. Similarity Aggregator ──────────────────────────────────────────────────

class SimilarityAggregator:

    #Combines lexical and semantic scores using configurable weights.
    #Default: 40 % TF-IDF + 60 % Sentence-BERT (as proposed in the paper).


    def __init__(self, lexical_weight: float = 0.4, semantic_weight: float = 0.6):
        assert abs(lexical_weight + semantic_weight - 1.0) < 1e-6, "Weights must sum to 1"
        self.lw = lexical_weight
        self.sw = semantic_weight

    def combine(self, lexical_score: float, semantic_score: float) -> float:
        return round(self.lw * lexical_score + self.sw * semantic_score, 4)

    def verdict(self, hybrid_score: float) -> str:
        if hybrid_score >= 0.65:
            return "HIGH — likely plagiarism"
        if hybrid_score >= 0.35:
            return "MODERATE — possible paraphrasing"
        return "LOW — likely original"


# ── 7. Report Generator ───────────────────────────────────────────────────────

class ReportGenerator:
    #Assembles and prints/returns a structured plagiarism report.

    def generate(
        self,
        submission_text: str,
        reference_text: str,
        tfidf_score: float,
        sbert_score: float,
        hybrid_score: float,
        verdict: str,
        common_phrases: list[str],
        ocr_applied: bool,
    ) -> str:
        bar = "─" * 60
        lines = [
            bar,
            " HYBRID MULTIMODAL PLAGIARISM DETECTION REPORT",
            bar,
            f"  Input mode        : {'Image (OCR applied)' if ocr_applied else 'Text document'}",
            f"  Submission length : {len(submission_text.split())} words",
            f"  Reference length  : {len(reference_text.split())} words",
            bar,
            " SIMILARITY SCORES",
            f"  TF-IDF (lexical)  : {tfidf_score * 100:.1f}%",
            f"  Sentence-BERT     : {sbert_score * 100:.1f}%  (fallback: {tfidf_score*100:.1f}% if unavailable)",
            f"  Hybrid (40/60)    : {hybrid_score * 100:.1f}%",
            bar,
            f" VERDICT  →  {verdict}",
            bar,
        ]

        if common_phrases:
            lines.append(" MATCHING PHRASES (lexical)")
            for phrase in common_phrases:
                lines.append(f"  • \"{phrase}\"")
            lines.append(bar)

        return "\n".join(lines)


# ── 8. Main Pipeline ──────────────────────────────────────────────────────────

class PlagiarismDetector:
    """
    Orchestrates the full pipeline:
      Input → OCR (if needed) → Preprocess → TF-IDF → SBERT → Aggregate → Report
    """

    def __init__(self, lexical_weight: float = 0.4, semantic_weight: float = 0.6):
        self.input_handler = InputHandler()
        self.preprocessor = Preprocessor()
        self.tfidf = TFIDFEngine()
        self.sbert = SBERTEngine()
        self.aggregator = SimilarityAggregator(lexical_weight, semantic_weight)
        self.reporter = ReportGenerator()

    def detect(self, submission_source: str, reference_source: str) -> dict:
        """
        Run full detection pipeline.

        Parameters
        ----------
        submission_source : file path or raw text string (submission)
        reference_source  : file path or raw text string (reference)

        Returns
        -------
        dict with keys: tfidf, sbert, hybrid, verdict, report
        """
        # ── Step 1: Load inputs ──────────────────────────────────────────────
        print("[1/5] Loading inputs…")
        submission_raw = self.input_handler.load(submission_source)
        reference_raw = self.input_handler.load(reference_source)

        ocr_applied = (
            Path(submission_source).exists()
            and Path(submission_source).suffix.lower()
            in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
            if Path(submission_source).exists() else False
        )

        # ── Step 2: Preprocess ───────────────────────────────────────────────
        print("[2/5] Preprocessing…")
        tokens1 = self.preprocessor.process(submission_raw)
        tokens2 = self.preprocessor.process(reference_raw)
        clean1 = self.preprocessor.process_to_str(submission_raw)
        clean2 = self.preprocessor.process_to_str(reference_raw)

        # ── Step 3: TF-IDF lexical similarity ───────────────────────────────
        print("[3/5] Computing TF-IDF cosine similarity…")
        tfidf_score = self.tfidf.compute(tokens1, tokens2)
        phrases = self.tfidf.common_phrases(submission_raw, reference_raw)

        # ── Step 4: Sentence-BERT semantic similarity ────────────────────────
        print("[4/5] Computing Sentence-BERT semantic similarity…")
        try:
            sbert_score = self.sbert.compute(clean1, clean2)
        except ImportError as e:
            print(f"  [WARN] {e}")
            print("  Falling back to TF-IDF score for semantic component.")
            sbert_score = tfidf_score

        # ── Step 5: Aggregate & report ───────────────────────────────────────
        print("[5/5] Generating report…")
        hybrid = self.aggregator.combine(tfidf_score, sbert_score)
        verdict = self.aggregator.verdict(hybrid)
        report = self.reporter.generate(
            submission_raw, reference_raw,
            tfidf_score, sbert_score, hybrid,
            verdict, phrases, ocr_applied,
        )

        print("\n" + report)

        return {
            "tfidf_score": tfidf_score,
            "sbert_score": sbert_score,
            "hybrid_score": hybrid,
            "verdict": verdict,
            "common_phrases": phrases,
            "report": report,
            "ocr_applied": ocr_applied,
        }


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Multimodal Plagiarism Detector"
    )
    parser.add_argument("submission", help="Path to submission file or raw text")
    parser.add_argument("reference", help="Path to reference file or raw text")
    parser.add_argument("--lexical-weight", type=float, default=0.4,
                        help="Weight for TF-IDF score (default: 0.4)")
    parser.add_argument("--semantic-weight", type=float, default=0.6,
                        help="Weight for SBERT score (default: 0.6)")
    args = parser.parse_args()

    detector = PlagiarismDetector(args.lexical_weight, args.semantic_weight)
    detector.detect(args.submission, args.reference)


if __name__ == "__main__":
    main()
