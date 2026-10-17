import streamlit as st
import tempfile
import os
from pathlib import Path
from plagiarism_detector import (
    InputHandler, Preprocessor, TFIDFEngine,
    SBERTEngine, SimilarityAggregator, ReportGenerator,
    OCR_AVAILABLE, SBERT_AVAILABLE,
)
from corpus_builder import build_dynamic_corpus
 
# ── Page config ────────────────────────────────────────────────────────────────
 
st.set_page_config(
    page_title="Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
)
 
st.markdown("""
<style>
.metric-box {
    background: var(--background-secondary-color);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    border: 1px solid #e9ecef;
    color: var(--text-color);
}
.metric-label { font-size: 12px; color: #6c757d; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 32px; font-weight: 700; margin: 4px 0; font-family: monospace; }
.high   { color: #dc3545; }
.medium { color: #fd7e14; }
.low    { color: #198754; }
.verdict-high   { background: #f8d7da; border: 1px solid #f5c2c7; color: #842029; border-radius: 8px; padding: 12px 16px; }
.verdict-medium { background: #fff3cd; border: 1px solid #ffecb5; color: #664d03; border-radius: 8px; padding: 12px 16px; }
.verdict-low    { background: #d1e7dd; border: 1px solid #badbcc; color: #0f5132; border-radius: 8px; padding: 12px 16px; }
.match-card {
    background: var(--background-secondary-color);
    border-radius: 8px;
    padding: 14px;
    border-left: 4px solid #dee2e6;
    margin-bottom: 10px;
    color: var(--text-color);
}
.match-card.high-border  { border-left-color: #dc3545; }
.match-card.med-border   { border-left-color: #fd7e14; }
.match-card.low-border   { border-left-color: #198754; }
.keyword-tag {
    display: inline-block;
    background: #e9ecef;
    color: #495057;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)
 
 
# ── Sidebar ────────────────────────────────────────────────────────────────────
 
with st.sidebar:
    st.title("⚙️ Settings")
 
    st.subheader("Similarity Weights")
    lex_w = st.slider("TF-IDF weight", 0.0, 1.0, 0.4, 0.05)
    sem_w = round(1.0 - lex_w, 2)
    st.write(f"Sentence-BERT weight: **{sem_w}**")
 
    st.divider()
    st.subheader("Corpus Settings")
    n_terms           = st.slider("Keywords to extract", 3, 10, 5)
    articles_per_term = st.slider("Articles per keyword", 2, 6, 3)
    st.caption(f"Fetches up to **{n_terms * articles_per_term} Wikipedia articles** per check")
 
    st.divider()
    st.subheader("System Status")
    st.write(f"{'✅' if OCR_AVAILABLE else '❌'} OCR (pytesseract)")
    st.write(f"{'✅' if SBERT_AVAILABLE else '❌'} Sentence-BERT")
 
    st.divider()
    st.subheader("Thresholds")
    st.write("🔴 ≥ 65% — High / likely plagiarism")
    st.write("🟡 35–64% — Moderate / paraphrasing")
    st.write("🟢 < 35% — Low / likely original")
 
 
# ── Header ─────────────────────────────────────────────────────────────────────
 
st.title("🔍 Hybrid Multimodal Plagiarism Detector")
st.caption("Dynamically fetches relevant Wikipedia articles based on your submission · TF-IDF · Sentence-BERT · OCR")
st.divider()
 
 
# ── Input ──────────────────────────────────────────────────────────────────────
 
st.subheader("📄 Upload or paste your submission")
input_mode = st.radio(
    "Input type",
    ["Text", "File upload (txt / pdf / image)"],
    horizontal=True,
)
 
submission_text = ""
ocr_applied     = False
 
if input_mode == "Text":
    col1, col2 = st.columns([3, 1])
    with col1:
        submission_text = st.text_area(
            "Paste submission here",
            height=220,
            placeholder="Paste the student's submitted text here…",
            label_visibility="collapsed",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Load demo text"):
            st.session_state["demo_text"] = (
                "Machine learning is a subset of artificial intelligence that enables systems to "
                "automatically learn and improve from experience without being explicitly programmed. "
                "The process of learning begins with observations or data, such as examples, direct "
                "experience, or instruction. Machine learning algorithms build a model based on sample "
                "data, known as training data, in order to make predictions or decisions without being "
                "explicitly programmed to do so. These algorithms are used in a wide variety of "
                "applications, such as email filtering and computer vision."
            )
            st.rerun()
 
    if "demo_text" in st.session_state:
        submission_text = st.session_state["demo_text"]
        st.text_area(
            "Loaded demo:",
            value=submission_text,
            height=130,
            disabled=True,
        )
 
else:
    uploaded = st.file_uploader(
        "Upload file",
        type=["txt", "pdf", "png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    if uploaded:
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
 
        handler = InputHandler()
        with st.spinner("Extracting text…"):
            result = handler.load(tmp_path)
            if isinstance(result, tuple):
                submission_text, ocr_applied = result
            else:
                submission_text = result
                ocr_applied = suffix in {".png", ".jpg", ".jpeg"}
 
        os.unlink(tmp_path)
 
        if ocr_applied:
            st.success(f"✅ OCR applied — {len(submission_text.split())} words extracted")
        else:
            st.success(f"✅ Loaded — {len(submission_text.split())} words")
 
        with st.expander("Preview extracted text"):
            st.text(submission_text[:800] + ("…" if len(submission_text) > 800 else ""))
 
 
# ── Run ────────────────────────────────────────────────────────────────────────
 
st.divider()
run = st.button("🚀 Check for Plagiarism", type="primary", use_container_width=True)
 
if run:
    if not submission_text.strip():
        st.error("Please provide a submission first.")
        st.stop()
 
    preprocessor = Preprocessor()
    tfidf_engine  = TFIDFEngine()
    sbert_engine  = SBERTEngine()
    aggregator    = SimilarityAggregator(lex_w, sem_w)
    reporter      = ReportGenerator()
 
    progress    = st.progress(0, text="Starting pipeline…")
    status_text = st.empty()
 
    # ── Step 1: Extract keywords & build dynamic corpus ────────────────────
    progress.progress(5, "Extracting keywords from submission…")
 
    corpus_placeholder = st.empty()
    corpus = []
    keywords_used = []
 
    def corpus_progress(fetched, total, keyword):
        pct = 5 + int((fetched / total) * 35)
        progress.progress(pct, f"Fetching Wikipedia: '{keyword}' ({fetched}/{total})…")
 
    with st.spinner("Building corpus from Wikipedia based on your submission…"):
        corpus, keywords_used = build_dynamic_corpus(
            submission_text,
            n_terms=n_terms,
            articles_per_term=articles_per_term,
            progress_cb=corpus_progress,
        )
 
    # Show keywords and corpus size
    keywords_html = " ".join(f'<span class="keyword-tag">🔑 {k}</span>' for k in keywords_used)
    corpus_placeholder.markdown(
        f"**Corpus built** — {len(corpus)} Wikipedia articles fetched using keywords: {keywords_html}",
        unsafe_allow_html=True,
    )
 
    if not corpus:
        st.error("Could not fetch any Wikipedia articles. Check your internet connection.")
        st.stop()
 
    # ── Step 2: Preprocess ─────────────────────────────────────────────────
    progress.progress(42, "Preprocessing text…")
    sub_tokens = preprocessor.process(submission_text)
    sub_clean  = preprocessor.process_to_str(submission_text)
 
    # ── Step 3: TF-IDF pass ────────────────────────────────────────────────
    tfidf_scores = []
    for i, doc in enumerate(corpus):
        pct = 42 + int((i / len(corpus)) * 20)
        progress.progress(pct, f"TF-IDF pass: {i+1}/{len(corpus)} articles…")
        ref_tokens = preprocessor.process(doc["text"])
        score = tfidf_engine.compute(sub_tokens, ref_tokens)
        tfidf_scores.append((i, score))
 
    tfidf_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = tfidf_scores[:10]
 
    # ── Step 4: SBERT on shortlist ─────────────────────────────────────────
    progress.progress(65, "Running Sentence-BERT semantic analysis…")
    sbert_scores = {}
    try:
        for corp_idx, tfidf_s in top_candidates:
            ref_clean = preprocessor.process_to_str(corpus[corp_idx]["text"])
            score = sbert_engine.compute(sub_clean, ref_clean)
            sbert_scores[corp_idx] = score
    except Exception as e:
        for corp_idx, tfidf_s in top_candidates:
            sbert_scores[corp_idx] = tfidf_s
        st.warning(f"Sentence-BERT unavailable — using TF-IDF for semantic component. Error: {e}")
 
    # ── Step 5: Combine & rank ─────────────────────────────────────────────
    progress.progress(88, "Aggregating scores…")
    results = []
    for corp_idx, tfidf_s in tfidf_scores:
        sbert_s = sbert_scores.get(corp_idx, tfidf_s)
        hybrid  = aggregator.combine(tfidf_s, sbert_s)
        phrases = tfidf_engine.common_phrases(submission_text, corpus[corp_idx]["text"])
        results.append({
            "title":        corpus[corp_idx]["title"],
            "tfidf_score":  tfidf_s,
            "sbert_score":  sbert_s,
            "hybrid_score": hybrid,
            "phrases":      phrases,
        })
 
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top5      = results[:5]
    top_score = top5[0]["hybrid_score"] if top5 else 0.0
    verdict   = aggregator.verdict(top_score)
 
    progress.progress(100, "Done ✓")
    status_text.empty()
 
    # ── Results ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Results")
 
    def colour(v): return "high" if v >= 0.65 else ("medium" if v >= 0.35 else "low")
 
    top_tfidf = top5[0]["tfidf_score"] if top5 else 0
    top_sbert = top5[0]["sbert_score"] if top5 else 0
 
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">Top TF-IDF Match</div>
            <div class="metric-value {colour(top_tfidf)}">{top_tfidf*100:.1f}%</div>
            <div class="metric-label">Lexical overlap</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">Top Sentence-BERT Match</div>
            <div class="metric-value {colour(top_sbert)}">{top_sbert*100:.1f}%</div>
            <div class="metric-label">Semantic similarity</div></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">Hybrid Score ({int(lex_w*100)}/{int(sem_w*100)})</div>
            <div class="metric-value {colour(top_score)}">{top_score*100:.1f}%</div>
            <div class="metric-label">Combined verdict</div></div>""",
            unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # Verdict banner
    cls  = "high" if top_score >= 0.65 else ("medium" if top_score >= 0.35 else "low")
    icon = "⚠️" if top_score >= 0.65 else ("ℹ️" if top_score >= 0.35 else "✅")
    st.markdown(
        f'<div class="verdict-{cls}"><strong>{icon} {verdict}</strong></div>',
        unsafe_allow_html=True)
 
    # Top matches
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔗 Top Matching Sources")
 
    for rank, m in enumerate(top5, 1):
        border = ("high-border" if m["hybrid_score"] >= 0.65
                  else "med-border" if m["hybrid_score"] >= 0.35
                  else "low-border")
        phrases_html = "".join(
            f'<span style="background:#fff3cd;padding:2px 6px;border-radius:4px;'
            f'font-size:12px;margin:2px;display:inline-block;">"{p}"</span>'
            for p in m["phrases"]
        ) if m["phrases"] else "<em style='font-size:12px;color:#6c757d'>No exact phrase overlap</em>"
 
        st.markdown(f"""
        <div class="match-card {border}">
            <strong>#{rank} {m['title']}</strong><br>
            <span style="font-family:monospace;font-size:13px;">
                TF-IDF: <b>{m['tfidf_score']*100:.1f}%</b> &nbsp;|&nbsp;
                SBERT: <b>{m['sbert_score']*100:.1f}%</b> &nbsp;|&nbsp;
                Hybrid: <b>{m['hybrid_score']*100:.1f}%</b>
            </span><br><br>
            {phrases_html}
        </div>""", unsafe_allow_html=True)
 
    # Report & download
    report_text = reporter.generate(
        submission_text, "", top_tfidf, top_sbert,
        top_score, verdict,
        top5[0]["phrases"] if top5 else [],
        ocr_applied,
    )
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download Report",
        data=report_text,
        file_name="plagiarism_report.txt",
        mime="text/plain",
    )
    with st.expander("View full report"):
        st.code(report_text, language="text")
 

