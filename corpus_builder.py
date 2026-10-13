"""
Corpus builder — fetches Wikipedia articles as the reference corpus.
Add any topic to TOPICS to expand the corpus instantly.
"""

import wikipedia

# ── Topics to fetch ───────────────────────────────────────────────────────────
# Add or remove any Wikipedia article title here to expand/shrink the corpus.

TOPICS = [
    # Machine Learning & AI
    "Machine learning",
    "Deep learning",
    "Artificial intelligence",
    "Natural language processing",
    "Computer vision",
    "Reinforcement learning",
    "Neural network",
    "Supervised learning",
    "Unsupervised learning",
    "Transfer learning",

    # Models & Algorithms
    "BERT (language model)",
    "Transformer (deep learning architecture)",
    "Convolutional neural network",
    "Recurrent neural network",
    "Support vector machine",
    "Random forest",
    "K-nearest neighbors algorithm",
    "Decision tree",
    "Logistic regression",
    "Gradient boosting",

    # NLP Topics
    "Sentiment analysis",
    "Text summarization",
    "Machine translation",
    "Named-entity recognition",
    "Word2vec",
    "TF-IDF",
    "Cosine similarity",

    # CS Fundamentals
    "Object-oriented programming",
    "Data structure",
    "Algorithm",
    "Operating system",
    "Database",
    "Computer network",
    "Cybersecurity",
    "Cloud computing",
    "Software engineering",
    "Compiler",

    # Other Academic Topics
    "Blockchain",
    "Internet of things",
    "Quantum computing",
    "Big data",
    "Data science",
    "Plagiarism",
    "Optical character recognition",
]


def build_corpus(progress_cb=None) -> list:
    """
    Fetch Wikipedia articles for all topics.
    Returns list of dicts with id, title, text.
    Skips topics that fail to fetch (disambiguation, not found, etc.)
    """
    corpus = []
    for i, topic in enumerate(TOPICS):
        try:
            page = wikipedia.page(topic, auto_suggest=False)
            # Use first 3000 chars — enough for good comparison, not too slow
            corpus.append({
                "id":    topic.lower().replace(" ", "_"),
                "title": f"{page.title} — Wikipedia",
                "text":  page.content[:3000],
            })
        except Exception:
            # Skip disambiguation pages, missing articles, etc.
            pass

        if progress_cb:
            progress_cb(i + 1, len(TOPICS))

    return corpus