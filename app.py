import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import nltk
from textblob import download_corpora

# ✅ FIX: Download NLP Data (RUN ONLY ONCE)
@st.cache_resource
def download_nlp_data():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('brown')
    nltk.download('averaged_perceptron_tagger')
    download_corpora.download_all()

download_nlp_data()

# ✅ FIX: Proper TextBlob import
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ModuleNotFoundError:
    TextBlob = None
    HAS_TEXTBLOB = False

    class TextBlobFallback:
        def __init__(self, text):
            self.text = text
            self.sentences = []
            self._sentiment = None
            self._evaluate()

        def _evaluate(self):
            from types import SimpleNamespace
            pos_words = {"good","great","excellent","amazing","love","happy","positive","best","wonderful","fantastic","nice"}
            neg_words = {"bad","terrible","awful","hate","worst","poor","negative","disappoint","angry","frustrating"}

            words = self.text.lower().split()
            score = 0

            for w in words:
                w = w.strip(".,!?;:")
                if w in pos_words:
                    score += 1
                elif w in neg_words:
                    score -= 1

            polarity = max(-1, min(1, score / 5))
            subjectivity = min(1, abs(score) / 5)

            self._sentiment = SimpleNamespace(polarity=polarity, subjectivity=subjectivity)

            # Sentence split
            sentences = self.text.replace("!", ".").replace("?", ".").split(".")
            self.sentences = [SimpleNamespace(sentiment=self._sentiment, __str__=lambda s=s: s.strip())
                              for s in sentences if s.strip()]

        @property
        def sentiment(self):
            return self._sentiment

    def TextBlob(text):
        return TextBlobFallback(text)


# ─── PAGE CONFIG ───
st.set_page_config(page_title="SentimentIQ", page_icon="🧠", layout="wide")


# ─── CSS ───
st.markdown("""
<style>
.stApp { background-color: #0b0c10; color: #e8eaf0; }
.stButton > button { background:#22d07a; color:black; font-weight:bold; }
.verdict-positive { background:#0f2e1c; border:1px solid #22d07a; padding:15px; border-radius:10px; }
.verdict-negative { background:#2e0f18; border:1px solid #ff4d6d; padding:15px; border-radius:10px; }
.verdict-neutral { background:#2e260f; border:1px solid #f0b429; padding:15px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)


# ─── SIDEBAR ───
with st.sidebar:
    st.title("🧠 SentimentIQ")
    threshold = st.slider("Threshold", 0.0, 0.5, 0.1)


# ─── MAIN ───
st.title("Sentiment Analysis App")

text = st.text_area("Enter text")

if st.button("Analyze"):

    if not text.strip():
        st.warning("Enter text first")
    else:
        blob = TextBlob(text)
        score = blob.sentiment.polarity
        percent = (score + 1) * 50

        if score > threshold:
            label, emoji, css = "Positive", "😊", "positive"
        elif score < -threshold:
            label, emoji, css = "Negative", "😡", "negative"
        else:
            label, emoji, css = "Neutral", "😐", "neutral"

        # Verdict
        st.markdown(f"""
        <div class="verdict-{css}">
        <h2>{emoji} {label}</h2>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        c1, c2 = st.columns(2)
        c1.metric("Score", round(score, 3))
        c2.metric("Percentage", f"{percent:.1f}%")

        # Chart
        fig, ax = plt.subplots()
        ax.bar(["Score"], [score])
        st.pyplot(fig)
