import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')

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
            text_lower = self.text.lower()
            pos_words = set(["good", "great", "excellent", "amazing", "love", "happy", "positive", "best", "wonderful", "fantastic", "nice"])
            neg_words = set(["bad", "terrible", "awful", "hate", "worst", "poor", "negative", "disappoint", "angry", "frustrating"])
            words = [w.strip(".,!?;:") for w in text_lower.split()]
            score = 0.0

            for w in words:
                if w in pos_words:
                    score += 1.0
                if w in neg_words:
                    score -= 1.0

            polarity = max(-1.0, min(1.0, score / 5.0))
            subjectivity = min(1.0, round(abs(score) / 5.0, 2))
            self._sentiment = SimpleNamespace(polarity=polarity, subjectivity=subjectivity)

            class FallbackSentence:
                def __init__(self, text, polarity, subjectivity):
                    from types import SimpleNamespace
                    self._text = text
                    self.sentiment = SimpleNamespace(polarity=polarity, subjectivity=subjectivity)

                def __str__(self):
                    return self._text

            sentence_texts = [s.strip() for s in self.text.replace('?', '.').replace('!', '.').split('.') if s.strip()]
            self.sentences = []

            for stx in sentence_texts:
                sent_score = 0.0
                sent_words = [w.strip(".,!?;:") for w in stx.lower().split()]
                for w in sent_words:
                    if w in pos_words:
                        sent_score += 1.0
                    if w in neg_words:
                        sent_score -= 1.0
                sent_polarity = max(-1.0, min(1.0, sent_score / 5.0))
                self.sentences.append(FallbackSentence(stx, sent_polarity, min(1.0, abs(sent_score)/5.0)))

        @property
        def sentiment(self):
            return self._sentiment

        def __str__(self):
            return self.text

    def TextBlob(text):
        return TextBlobFallback(text)


# Page Config
st.set_page_config(
    page_title="SentimentIQ",
    page_icon="🧠",
    layout="wide",
)

# Sidebar
with st.sidebar:
    threshold = st.slider("Threshold", 0.0, 0.5, 0.1)

# Main UI
st.title("Sentiment Analysis App")

examples = {
    "Positive": "This product is amazing and I love it",
    "Negative": "This is the worst product ever",
    "Neutral": "The package arrived on time"
}

col1, col2, col3 = st.columns(3)

if col1.button("Positive"):
    st.session_state["text_input"] = examples["Positive"]
if col2.button("Negative"):
    st.session_state["text_input"] = examples["Negative"]
if col3.button("Neutral"):
    st.session_state["text_input"] = examples["Neutral"]

# ✅ FIXED HERE
text = st.text_area(
    "Enter text",
    value=st.session_state.get("text_input", ""),
    key="text_input"
)

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Enter text first")
    else:
        blob = TextBlob(text)
        score = blob.sentiment.polarity

        if score > threshold:
            st.success("Positive 😊")
        elif score < -threshold:
            st.error("Negative 😡")
        else:
            st.info("Neutral 😐")

        st.write("Score:", score)
