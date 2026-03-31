import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from textblob import download_corpora

# ─────────────────────────────────────────────
# ✅ DOWNLOAD NLP DATA (RUN ONCE)
# ─────────────────────────────────────────────
@st.cache_resource
def download_nlp_data():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('brown')
    nltk.download('averaged_perceptron_tagger')
    download_corpora.download_all()

download_nlp_data()

# ─────────────────────────────────────────────
# ✅ TEXTBLOB IMPORT + FALLBACK
# ─────────────────────────────────────────────
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except:
    HAS_TEXTBLOB = False

    class TextBlobFallback:
        def __init__(self, text):
            self.text = text
            self.sentences = []
            self._sentiment = None
            self._analyze()

        def _analyze(self):
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

            # sentence split
            self.sentences = []
            for s in self.text.replace("!", ".").replace("?", ".").split("."):
                s = s.strip()
                if s:
                    self.sentences.append(
                        SimpleNamespace(
                            sentiment=self._sentiment,
                            __str__=lambda s=s: s
                        )
                    )

        @property
        def sentiment(self):
            return self._sentiment

    def TextBlob(text):
        return TextBlobFallback(text)

# ─────────────────────────────────────────────
# 🎨 PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="SentimentIQ", page_icon="🧠", layout="wide")

# ─────────────────────────────────────────────
# 🎨 CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0b0c10; color: #e8eaf0; }
.stButton > button { background:#22d07a; color:black; font-weight:bold; }
.verdict-positive { background:#0f2e1c; border:1px solid #22d07a; padding:15px; border-radius:10px; }
.verdict-negative { background:#2e0f18; border:1px solid #ff4d6d; padding:15px; border-radius:10px; }
.verdict-neutral { background:#2e260f; border:1px solid #f0b429; padding:15px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 🧠 SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 SentimentIQ")
    threshold = st.slider("Threshold", 0.0, 0.5, 0.1)

# ─────────────────────────────────────────────
# 🚀 MAIN APP
# ─────────────────────────────────────────────
st.title("Sentiment Analysis App")

text = st.text_area("Enter text")

if st.button("Analyze"):

    if not text.strip():
        st.warning("Enter text first")
    else:
        blob = TextBlob(text)
        score = blob.sentiment.polarity
        percent = (score + 1) * 50

        # sentiment label
        if score > threshold:
            label, emoji, css = "Positive", "😊", "positive"
        elif score < -threshold:
            label, emoji, css = "Negative", "😡", "negative"
        else:
            label, emoji, css = "Neutral", "😐", "neutral"

        # ── Verdict ──
        st.markdown(f"""
        <div class="verdict-{css}">
        <h2>{emoji} {label}</h2>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics ──
        c1, c2 = st.columns(2)
        c1.metric("Score", round(score, 3))
        c2.metric("Percentage", f"{percent:.1f}%")

        # ─────────────────────────
        # 📊 BAR GRAPH
        # ─────────────────────────
        st.subheader("📊 Sentiment Graph")
        fig, ax = plt.subplots()
        ax.bar(["Score"], [score])
        st.pyplot(fig)

        # ─────────────────────────
        # 🥧 PIE CHART
        # ─────────────────────────
        st.subheader("📊 Distribution")
        pos = max(0, score)
        neg = max(0, -score)
        neu = 1 - abs(score)

        fig2, ax2 = plt.subplots()
        ax2.pie(
            [pos, neu, neg],
            labels=["Positive", "Neutral", "Negative"],
            autopct="%1.1f%%"
        )
        st.pyplot(fig2)

        # ─────────────────────────
        # 🧠 WORD ANALYSIS
        # ─────────────────────────
        st.subheader("🔍 Word Analysis")

        pos_words = {"good","great","excellent","amazing","love","happy","positive","best","wonderful","fantastic","nice"}
        neg_words = {"bad","terrible","awful","hate","worst","poor","negative","disappoint","angry","frustrating"}

        words = text.lower().split()
        found_pos = [w for w in words if w.strip(".,!?;:") in pos_words]
        found_neg = [w for w in words if w.strip(".,!?;:") in neg_words]

        c1, c2 = st.columns(2)

        # ✅ FIXED ERROR HERE
        pos_text = ", ".join(set(found_pos)) if found_pos else "None"
        neg_text = ", ".join(set(found_neg)) if found_neg else "None"

        c1.write(f"✅ Positive Words: {pos_text}")
        c2.write(f"❌ Negative Words: {neg_text}")

        # ─────────────────────────
        # 📝 TEXT STATS
        # ─────────────────────────
        st.subheader("📝 Text Stats")
        words_list = text.split()

        s1, s2, s3 = st.columns(3)
        s1.metric("Words", len(words_list))
        s2.metric("Characters", len(text))
        s3.metric("Sentences", len(blob.sentences))
