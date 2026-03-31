import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from textblob import download_corpora

# ✅ Download NLP data (only once)
@st.cache_resource
def download_nlp_data():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('brown')
    nltk.download('averaged_perceptron_tagger')
    download_corpora.download_all()

download_nlp_data()

# ✅ TextBlob or fallback
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ModuleNotFoundError:
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

        # Classification
        if score > threshold:
            label, emoji, css = "Positive", "😊", "positive"
        elif score < -threshold:
            label, emoji, css = "Negative", "😡", "negative"
        else:
            label, emoji, css = "Neutral", "😐", "neutral"

        # ─── Verdict ───
        st.markdown(f"""
        <div class="verdict-{css}">
        <h2>{emoji} {label}</h2>
        </div>
        """, unsafe_allow_html=True)

        # ─── Metrics ───
        c1, c2 = st.columns(2)
        c1.metric("Score", round(score, 3))
        c2.metric("Percentage", f"{percent:.1f}%")

        # ─── GRAPHS ───
        st.markdown("### 📊 Visual Analysis")

        col1, col2 = st.columns(2)

        pos_val = max(0, score)
        neg_val = max(0, -score)
        neu_val = max(0, 1 - abs(score))

        # Bar Chart
        with col1:
            fig1, ax1 = plt.subplots()
            labels = ["Positive", "Neutral", "Negative"]
            values = [pos_val, neu_val, neg_val]

            ax1.bar(labels, values)
            ax1.set_title("Sentiment Breakdown")
            ax1.set_ylim(0, 1)

            for i, v in enumerate(values):
                ax1.text(i, v + 0.02, f"{round(v,2)}", ha='center')

            st.pyplot(fig1)

        # Pie Chart
        with col2:
            fig2, ax2 = plt.subplots()
            sizes = [pos_val, neu_val, neg_val]
            labels = ["Positive", "Neutral", "Negative"]

            ax2.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=140)
            ax2.set_title("Distribution")

            st.pyplot(fig2)

        # Gauge
        st.markdown("### 🎯 Polarity Gauge")

        fig3, ax3 = plt.subplots(figsize=(6, 1.5))
        gradient = np.linspace(0, 1, 256).reshape(1, -1)

        ax3.imshow(gradient, aspect="auto", cmap=plt.cm.RdYlGn,
                   extent=[-1, 1, -0.3, 0.3])

        ax3.axvline(score, linewidth=3)
        ax3.plot(score, 0, "o")

        ax3.set_yticks([])
        ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax3.set_title("Polarity Score (-1 to +1)")

        st.pyplot(fig3)

        # ─── Sentence Analysis ───
        st.markdown("---")
        st.subheader("🔍 Sentence-Level Analysis")

        for i, sentence in enumerate(blob.sentences):
            s_text = str(sentence)
            s_score = sentence.sentiment.polarity

            if s_score > threshold:
                s_label, s_emoji = "Positive", "😊"
            elif s_score < -threshold:
                s_label, s_emoji = "Negative", "😡"
            else:
                s_label, s_emoji = "Neutral", "😐"

            st.write(f"{i+1}. {s_text}")
            st.write(f"{s_emoji} {s_label} (Score: {round(s_score,3)})")
            st.markdown("---")

        # ─── Stats ───
        st.subheader("📝 Text Statistics")

        words = text.split()

        c1, c2, c3 = st.columns(3)
        c1.metric("Words", len(words))
        c2.metric("Characters", len(text))
        c3.metric("Sentences", len(blob.sentences))

        # ─── Table ───
        st.subheader("📋 Sentence Table")

        data = []
        for i, sentence in enumerate(blob.sentences):
            s_text = str(sentence)
            s_score = sentence.sentiment.polarity

            if s_score > threshold:
                s_label = "Positive"
            elif s_score < -threshold:
                s_label = "Negative"
            else:
                s_label = "Neutral"

            data.append({
                "Sentence No": i+1,
                "Sentence": s_text,
                "Score": round(s_score, 3),
                "Sentiment": s_label
            })

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
