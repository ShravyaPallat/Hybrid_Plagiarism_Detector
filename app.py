#Streamlit UI — Hybrid Multimodal Plagiarism Detection System

import streamlit as st
import tempfile
import os
from pathlib import Path
from plagiarism_detector import (
    InputHandler, Preprocessor, TFIDFEngine,
    SBERTEngine, SimilarityAggregator, ReportGenerator,
    OCR_AVAILABLE, SBERT_AVAILABLE,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
.metric-box {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    border: 1px solid #e9ecef;
}
.metric-label { font-size: 12px; color: #6c757d; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 32px; font-weight: 700; margin: 4px 0; font-family: monospace; }
.high   { color: #dc3545; }
.medium { color: #fd7e14; }
.low    { color: #198754; }
.verdict-high   { background: #f8d7da; border: 1px solid #f5c2c7; color: #842029; border-radius: 8px; padding: 12px 16px; }
.verdict-medium { background: #fff3cd; border: 1px solid #ffecb5; color: #664d03; border-radius: 8px; padding: 12px 16px; }
.verdict-low    { background: #d1e7dd; border: 1px solid #badbcc; color: #0f5132; border-radius: 8px; padding: 12px 16px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Settings")

    st.subheader("Similarity Weights")
    lex_w = st.slider("TF-IDF weight", 0.0, 1.0, 0.4, 0.05)
    sem_w = round(1.0 - lex_w, 2)
    st.write(f"Sentence-BERT weight: **{sem_w}**")

    st.divider()
    st.subheader("System Status")
    st.write(f"{'Available' if OCR_AVAILABLE else 'Not available'} OCR (pytesseract)")
    st.write(f"{'Available' if SBERT_AVAILABLE else 'Not available'} Sentence-BERT")
    if not OCR_AVAILABLE:
        st.caption("Install: `pip install pillow pytesseract`")
    if not SBERT_AVAILABLE:
        st.caption("Install: `pip install sentence-transformers`")

    st.divider()
    st.subheader("Thresholds")
    st.write("🔴 ≥ 65% — High / likely plagiarism")
    st.write("🟡 35–64% — Moderate / paraphrasing")
    st.write("🟢 < 35% — Low / likely original")


# ── Header ─────────────────────────────────────────────────────────────────────

st.title("Hybrid Multimodal Plagiarism Detector")
st.caption("TF-IDF · Sentence-BERT · OCR — as described in the paper")
st.divider()


# ── Input columns ──────────────────────────────────────────────────────────────

col_sub, col_ref = st.columns(2)

with col_sub:
    st.subheader("Submission Document")
    input_mode = st.radio("Input type", ["Text", "File upload"], horizontal=True, key="sub_mode")

    submission_text = ""
    ocr_applied = False

    if input_mode == "Text":
        submission_text = st.text_area(
            "Paste submission text",
            height=220,
            placeholder="Paste the student's submitted text here…",
            label_visibility="collapsed",
        )
    else:
        uploaded = st.file_uploader(
            "Upload file", type=["txt", "pdf", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if uploaded:
            suffix = Path(uploaded.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            handler = InputHandler()
            with st.spinner("Extracting text…"):
                submission_text = handler.load(tmp_path)
            os.unlink(tmp_path)

            if suffix in {".png", ".jpg", ".jpeg"}:
                ocr_applied = True
                st.success(f"OCR applied — {len(submission_text.split())} words extracted")
            else:
                st.success(f"Loaded — {len(submission_text.split())} words")

            with st.expander("Preview extracted text"):
                st.text(submission_text[:1000] + ("…" if len(submission_text) > 1000 else ""))


with col_ref:
    st.subheader("Reference Document")
    ref_mode = st.radio("Input type", ["Text", "File upload"], horizontal=True, key="ref_mode")

    reference_text = ""

    if ref_mode == "Text":
        reference_text = st.text_area(
            "Paste reference text",
            height=220,
            placeholder="Paste the reference / source document here…",
            label_visibility="collapsed",
        )
    else:
        ref_uploaded = st.file_uploader(
            "Upload file", type=["txt", "pdf", "png", "jpg", "jpeg"],
            label_visibility="collapsed", key="ref_upload",
        )
        if ref_uploaded:
            suffix = Path(ref_uploaded.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(ref_uploaded.read())
                tmp_path = tmp.name

            handler = InputHandler()
            with st.spinner("Extracting text…"):
                reference_text = handler.load(tmp_path)
            os.unlink(tmp_path)
            st.success(f"Loaded — {len(reference_text.split())} words")

            with st.expander("Preview extracted text"):
                st.text(reference_text[:1000] + ("…" if len(reference_text) > 1000 else ""))


# ── Demo button ────────────────────────────────────────────────────────────────

if st.button("Load demo texts"):
    st.session_state["demo"] = True
    st.rerun()

if st.session_state.get("demo"):
    submission_text = (
        "Machine learning is a subset of artificial intelligence that enables systems to "
        "automatically learn and improve from experience without being explicitly programmed. "
        "The process of learning begins with observations or data, such as examples, direct "
        "experience, or instruction. Machine learning algorithms build a model based on sample "
        "data, known as training data, in order to make predictions or decisions without being "
        "explicitly programmed to do so."
    )
    reference_text = (
        "Machine learning (ML) is a type of artificial intelligence that allows software "
        "applications to become more accurate at predicting outcomes without being explicitly "
        "programmed. Machine learning algorithms use historical data as input to predict new "
        "output values. It gives enterprises a view of trends in customer behavior and business "
        "operational patterns, and supports the development of new products."
    )
    st.info("Demo texts loaded — click **Run Analysis** below.")


# ── Run analysis ───────────────────────────────────────────────────────────────

st.divider()
run = st.button("Run Analysis", type="primary", use_container_width=True)

if run:
    if not submission_text.strip():
        st.error("Please provide a submission document.")
        st.stop()
    if not reference_text.strip():
        st.error("Please provide a reference document.")
        st.stop()

    preprocessor = Preprocessor()
    tfidf_engine = TFIDFEngine()
    sbert_engine = SBERTEngine()
    aggregator = SimilarityAggregator(lex_w, sem_w)
    reporter = ReportGenerator()

    progress = st.progress(0, text="Starting pipeline…")

    # Step 1 — Preprocess
    progress.progress(20, "Preprocessing text…")
    tokens1 = preprocessor.process(submission_text)
    tokens2 = preprocessor.process(reference_text)
    clean1 = preprocessor.process_to_str(submission_text)
    clean2 = preprocessor.process_to_str(reference_text)

    # Step 2 — TF-IDF
    progress.progress(40, "Computing TF-IDF similarity…")
    tfidf_score = tfidf_engine.compute(tokens1, tokens2)
    phrases = tfidf_engine.common_phrases(submission_text, reference_text)

    # Step 3 — Sentence-BERT
    progress.progress(65, "Running Sentence-BERT semantic analysis…")
    try:
        sbert_score = sbert_engine.compute(clean1, clean2)
    except ImportError:
        sbert_score = tfidf_score
        st.warning("Sentence-BERT unavailable — using TF-IDF score for semantic component.")

    # Step 4 — Aggregate
    progress.progress(85, "Aggregating scores…")
    hybrid = aggregator.combine(tfidf_score, sbert_score)
    verdict = aggregator.verdict(hybrid)

    # Step 5 — Report
    progress.progress(100, "Done")
    report = reporter.generate(
        submission_text, reference_text,
        tfidf_score, sbert_score, hybrid,
        verdict, phrases, ocr_applied,
    )

    st.divider()
    st.subheader("Results")

    # Score cards
    c1, c2, c3 = st.columns(3)
    def colour(v): return "high" if v >= 0.65 else ("medium" if v >= 0.35 else "low")

    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">TF-IDF Lexical</div>
            <div class="metric-value {colour(tfidf_score)}">{tfidf_score*100:.1f}%</div>
            <div class="metric-label">String overlap</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Sentence-BERT</div>
            <div class="metric-value {colour(sbert_score)}">{sbert_score*100:.1f}%</div>
            <div class="metric-label">Semantic meaning</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Hybrid Score ({int(lex_w*100)}/{int(sem_w*100)})</div>
            <div class="metric-value {colour(hybrid)}">{hybrid*100:.1f}%</div>
            <div class="metric-label">Combined</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Verdict banner
    cls = "high" if hybrid >= 0.65 else ("medium" if hybrid >= 0.35 else "low")
    icon = "⚠️" if hybrid >= 0.65 else ("ℹ️" if hybrid >= 0.35 else "✅")
    st.markdown(
        f'<div class="verdict-{cls}"><strong>{icon} {verdict}</strong></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Matching phrases
    if phrases:
        st.subheader("Common Phrases Detected")
        for p in phrases:
            st.markdown(f"- `{p}`")

    # Pipeline summary
    with st.expander("Full Report"):
        st.code(report, language="text")

    # Download
    st.download_button(
        "Download Report",
        data=report,
        file_name="plagiarism_report.txt",
        mime="text/plain",
    )
