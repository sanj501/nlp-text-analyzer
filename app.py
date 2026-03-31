import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import nltk

# ─── NLTK SETUP (FIX FOR STREAMLIT CLOUD) ──────────────────────
@st.cache_resource
def setup_nltk():
    packages = ['punkt', 'wordnet']
    for pkg in packages:
        try:
            nltk.data.find(f'tokenizers/{pkg}')
        except:
            nltk.download(pkg)

setup_nltk()

# ─── TEXTBLOB IMPORT + ERROR HANDLING ──────────────────────────
try:
    from textblob import TextBlob
    from textblob.exceptions import MissingCorpusError
    HAS_TEXTBLOB = True
except Exception:
    HAS_TEXTBLOB = False

# ─── FALLBACK ANALYZER ─────────────────────────────────────────
class TextBlobFallback:
    def __init__(self, text):
        self.text = text
        self._evaluate()

    def _evaluate(self):
        from types import SimpleNamespace
        pos_words = {"good","great","excellent","amazing","love","happy","positive"}
        neg_words = {"bad","terrible","awful","hate","worst","poor","negative"}

        words = self.text.lower().split()
        score = sum(1 for w in words if w in pos_words) - sum(1 for w in words if w in neg_words)

        polarity = max(-1, min(1, score/5))
        subjectivity = min(1, abs(score)/5)

        self.sentiment = SimpleNamespace(polarity=polarity, subjectivity=subjectivity)

        # Sentence split
        sents = self.text.replace('!','.').replace('?','.').split('.')
        self.sentences = [SimpleNamespace(
            sentiment=SimpleNamespace(polarity=polarity),
            __str__=lambda self=s: s.strip()
        ) for s in sents if s.strip()]

# ─── SAFE TEXTBLOB FUNCTION ────────────────────────────────────
def analyze_text(text):
    try:
        blob = TextBlob(text)
        sentences = blob.sentences
    except Exception:
        blob = TextBlobFallback(text)
        sentences = blob.sentences
    return blob, sentences

# ─── UI SETUP ──────────────────────────────────────────────────
st.set_page_config(page_title="SentimentIQ", layout="wide")

st.title("🧠 Sentiment Analysis App")

text = st.text_area("Enter your text")

if st.button("Analyze") and text.strip():

    blob, sentences = analyze_text(text)

    score = blob.sentiment.polarity
    subjectivity = getattr(blob.sentiment, "subjectivity", 0)

    label = "Positive" if score > 0.1 else "Negative" if score < -0.1 else "Neutral"

    st.subheader(f"Result: {label}")
    st.write("Polarity:", round(score,3))
    st.write("Subjectivity:", round(subjectivity,3))
    st.write("Sentences:", len(sentences))

    # Chart
    fig, ax = plt.subplots()
    ax.bar(["Positive","Neutral","Negative"],
           [max(score,0), 1-abs(score), max(-score,0)])
    st.pyplot(fig)

    # Sentence analysis
    st.subheader("Sentence Analysis")
    for i, s in enumerate(sentences):
        try:
            s_text = str(s)
            s_score = s.sentiment.polarity
        except:
            s_text = str(s)
            s_score = 0

        st.write(f"{i+1}. {s_text} → {round(s_score,2)}")

else:
    st.info("Enter text and click Analyze")
