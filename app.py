import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from textblob import download_corpora

# ✅ Download NLP data
@st.cache_resource
def download_nlp_data():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('brown')
    nltk.download('averaged_perceptron_tagger')
    download_corpora.download_all()

download_nlp_data()

# ✅ TextBlob / fallback
try:
    from textblob import TextBlob
except:
    class TextBlobFallback:
        def __init__(self, text):
            from types import SimpleNamespace
            self.text = text
            pos_words = {"good","great","excellent","amazing","love","happy","positive","best","wonderful","fantastic","nice"}
            neg_words = {"bad","terrible","awful","hate","worst","poor","negative","disappoint","angry","frustrating"}

            words = text.lower().split()
            score = 0
            for w in words:
                w = w.strip(".,!?;:")
                if w in pos_words:
                    score += 1
                elif w in neg_words:
                    score -= 1

            polarity = max(-1, min(1, score/5))
            subjectivity = min(1, abs(score)/5)

            self.sentiment = SimpleNamespace(polarity=polarity, subjectivity=subjectivity)

            self.sentences = []
            for s in text.replace("!", ".").replace("?", ".").split("."):
                s = s.strip()
                if s:
                    self.sentences.append(SimpleNamespace(
                        sentiment=self.sentiment,
                        __str__=lambda s=s: s
                    ))

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

# ✅ Example buttons
st.markdown("### ⚡ Try Examples")
col1, col2, col3 = st.columns(3)

if col1.button("😊 Positive"):
    st.session_state["text"] = "This product is amazing and works perfectly."

if col2.button("😡 Negative"):
    st.session_state["text"] = "This is the worst experience I have ever had."

if col3.button("😐 Neutral"):
    st.session_state["text"] = "The package arrived on time and contains all items."

text = st.text_area("Enter text", value=st.session_state.get("text", ""))

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

        # Progress bar
        st.markdown("### 📈 Sentiment Strength")
        st.progress((score + 1) / 2)

        # Graphs
        st.markdown("### 📊 Visual Analysis")
        col1, col2 = st.columns(2)

        pos_val = max(0, score)
        neg_val = max(0, -score)
        neu_val = max(0, 1 - abs(score))

        # Bar
        with col1:
            fig1, ax1 = plt.subplots()
            labels = ["Positive", "Neutral", "Negative"]
            values = [pos_val, neu_val, neg_val]
            ax1.bar(labels, values)
            st.pyplot(fig1)

        # Pie
        with col2:
            fig2, ax2 = plt.subplots()
            ax2.pie([pos_val, neu_val, neg_val],
                    labels=["Positive","Neutral","Negative"],
                    autopct="%1.0f%%")
            st.pyplot(fig2)

        # Gauge
        fig3, ax3 = plt.subplots(figsize=(6,1.5))
        gradient = np.linspace(0,1,256).reshape(1,-1)
        ax3.imshow(gradient, aspect="auto", cmap=plt.cm.RdYlGn,
                   extent=[-1,1,-0.3,0.3])
        ax3.axvline(score)
        ax3.plot(score,0,"o")
        ax3.set_yticks([])
        st.pyplot(fig3)

        # Sentence analysis
        st.markdown("### 🔍 Sentence-Level Analysis")

        sentence_scores = []
        for i, s in enumerate(blob.sentences):
            s_score = s.sentiment.polarity
            sentence_scores.append(s_score)

            if s_score > threshold:
                s_label, s_emoji = "Positive", "😊"
            elif s_score < -threshold:
                s_label, s_emoji = "Negative", "😡"
            else:
                s_label, s_emoji = "Neutral", "😐"

            st.write(f"{i+1}. {str(s)}")
            st.write(f"{s_emoji} {s_label} ({round(s_score,3)})")

        # Line graph
        if sentence_scores:
            fig4, ax4 = plt.subplots()
            ax4.plot(sentence_scores, marker='o')
            ax4.axhline(0, linestyle='--')
            st.pyplot(fig4)

        # Stats
        st.markdown("### 📝 Text Statistics")
        words = text.split()
        c1, c2, c3 = st.columns(3)
        c1.metric("Words", len(words))
        c2.metric("Characters", len(text))
        c3.metric("Sentences", len(blob.sentences))

        # Keywords
        st.markdown("### 🔑 Keyword Insight")
        pos_words = {"good","great","excellent","amazing","love","happy"}
        neg_words = {"bad","terrible","worst","hate","poor"}

        found_pos = [w for w in words if w.lower() in pos_words]
        found_neg = [w for w in words if w.lower() in neg_words]

        c1, c2 = st.columns(2)
        c1.write("✅ Positive Words:", list(set(found_pos)))
        c2.write("❌ Negative Words:", list(set(found_neg)))

        # Table
        data = []
        for i, s in enumerate(blob.sentences):
            data.append({
                "Sentence": str(s),
                "Score": round(s.sentiment.polarity,3)
            })

        df = pd.DataFrame(data)
        st.dataframe(df)

        # Download
        st.markdown("### 💾 Download Report")
        report = f"""
Text: {text}
Sentiment: {label}
Score: {score}
Percentage: {percent}%
"""
        st.download_button("Download", report)
