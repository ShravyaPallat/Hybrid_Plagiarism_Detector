import wikipedia
from collections import Counter
import re
 
wikipedia.set_lang("en")
 
STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","shall","should","may","might",
    "can","could","that","this","these","those","it","its","not","as","if",
    "so","up","out","about","into","than","then","there","their","they",
    "we","you","he","she","i","my","your","our","his","her","which","who",
}
 
 
def extract_keywords(text: str, n: int = 5) -> list:
    """Extract the top n keywords from the submission using term frequency."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 3]
    freq = Counter(tokens)
    return [word for word, _ in freq.most_common(n)]
 
 
def build_dynamic_corpus(
    submission_text: str,
    n_terms: int = 5,
    articles_per_term: int = 3,
    progress_cb=None,
) -> tuple:
    """
    Build a corpus dynamically from the submission text.
 
    1. Extracts top keywords from the submission
    2. Searches Wikipedia for each keyword
    3. Fetches the top articles per keyword
    4. Returns (corpus, keywords_used)
    """
    keywords = extract_keywords(submission_text, n=n_terms)
    corpus   = []
    seen_ids = set()  # avoid duplicate articles across keyword searches
    total    = len(keywords) * articles_per_term
    fetched  = 0
 
    for keyword in keywords:
        try:
            search_results = wikipedia.search(keyword, results=articles_per_term)
        except Exception:
            continue
 
        for title in search_results:
            if title in seen_ids:
                continue
            try:
                page = wikipedia.page(title, auto_suggest=True, preload=False)
                corpus.append({
                    "id":    title.lower().replace(" ", "_"),
                    "title": f"{page.title} — Wikipedia",
                    "text":  page.content[:3000],
                })
                seen_ids.add(title)
            except wikipedia.DisambiguationError as e:
                # Pick the first specific option from the disambiguation page
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=True)
                    corpus.append({
                        "id":    e.options[0].lower().replace(" ", "_"),
                        "title": f"{page.title} — Wikipedia",
                        "text":  page.content[:3000],
                    })
                    seen_ids.add(title)
                except Exception:
                    pass
            except Exception:
                pass  # skip missing / broken pages
 
            fetched += 1
            if progress_cb:
                progress_cb(fetched, total, keyword)
 
    return corpus, keywords