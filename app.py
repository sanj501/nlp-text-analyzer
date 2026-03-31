import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ModuleNotFoundError:
    TextBlob = None
    HAS_TEXTBLOB = False
    st = st  # keep lint happy

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
            total = max(1, len(words))

            for w in words:
                if w in pos_words:
                    score += 1.0
                if w in neg_words:
                    score -= 1.0

            polarity = max(-1.0, min(1.0, score / 5.0))
            subjectivity = min(1.0, round(abs(score) / 5.0, 2))
            self._sentiment = SimpleNamespace(polarity=polarity, subjectivity=subjectivity)

            # sentence-level split fallback
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

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0b0c10; color: #e8eaf0; }
    [data-testid="stSidebar"] { background-color: #13151c; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1a1d27;
        border: 1px solid #252836;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.75rem; }
    [data-testid="stMetricValue"] { color: #e8eaf0 !important; }

    /* Buttons */
    .stButton > button {
        background-color: #22d07a;
        color: #000;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover { background-color: #1ab865; color: #000; }

    /* Text area */
    .stTextArea textarea {
        background-color: #13151c !important;
        color: #e8eaf0 !important;
        border: 1px solid #252836 !important;
        border-radius: 8px;
    }

    /* Expander */
    .streamlit-expanderHeader { color: #e8eaf0 !important; }

    /* Verdict box styles */
    .verdict-positive { background:#0f2e1c; border:1px solid #22d07a; border-radius:12px; padding:1rem 1.5rem; }
    .verdict-negative { background:#2e0f18; border:1px solid #ff4d6d; border-radius:12px; padding:1rem 1.5rem; }
    .verdict-neutral  { background:#2e260f; border:1px solid #f0b429; border-radius:12px; padding:1rem 1.5rem; }

    /* Sentence badge */
    .badge-pos { background:#0f2e1c; color:#22d07a; border:1px solid #22d07a; border-radius:20px; padding:2px 10px; font-size:0.72rem; }
    .badge-neg { background:#2e0f18; color:#ff4d6d; border:1px solid #ff4d6d; border-radius:20px; padding:2px 10px; font-size:0.72rem; }
    .badge-neu { background:#2e260f; color:#f0b429; border:1px solid #f0b429; border-radius:20px; padding:2px 10px; font-size:0.72rem; }

    h1, h2, h3 { color: #e8eaf0 !important; }
    p { color: #e8eaf0; }
    hr { border-color: #252836; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ───────────────────────────────────────────────────
def classify(score):
    if score > 0.1:
        return "Positive", "😊", "positive"
    elif score < -0.1:
        return "Negative", "😡", "negative"
    else:
        return "Neutral", "😐", "neutral"


def sentiment_color(label):
    return {"Positive": "#22d07a", "Negative": "#ff4d6d", "Neutral": "#f0b429"}[label]


def make_bar_chart(pos, neu, neg):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")

    labels = ["Positive", "Neutral", "Negative"]
    values = [pos, neu, neg]
    colors = ["#22d07a", "#f0b429", "#ff4d6d"]

    bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor="none")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", color="#e8eaf0", fontsize=10)

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score", color="#6b7280", fontsize=9)
    ax.tick_params(colors="#6b7280", labelsize=10)
    ax.spines[:].set_color("#252836")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.xaxis.label.set_color("#6b7280")
    ax.tick_params(axis="y", colors="#e8eaf0")
    ax.tick_params(axis="x", colors="#6b7280")
    ax.set_title("Sentiment Score Breakdown", color="#e8eaf0", fontsize=11, pad=10)
    fig.tight_layout()
    return fig


def make_pie_chart(pos, neu, neg):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")

    total = pos + neu + neg
    if total == 0:
        total = 1

    sizes  = [pos / total, neu / total, neg / total]
    colors = ["#22d07a", "#f0b429", "#ff4d6d"]
    labels = ["Positive", "Neutral", "Negative"]
    explode = [0.04, 0.04, 0.04]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.0f%%",
        colors=colors, startangle=140, explode=explode,
        wedgeprops={"linewidth": 2, "edgecolor": "#1a1d27"},
        textprops={"color": "#e8eaf0", "fontsize": 9},
    )
    for at in autotexts:
        at.set_color("#0b0c10")
        at.set_fontweight("bold")
        at.set_fontsize(9)

    ax.set_title("Distribution", color="#e8eaf0", fontsize=11, pad=8)
    fig.tight_layout()
    return fig


def make_gauge(score):
    """Polarity gauge from -1 to +1."""
    fig, ax = plt.subplots(figsize=(6, 1.2))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")

    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap=plt.cm.RdYlGn,
              extent=[-1, 1, -0.3, 0.3])

    # Needle
    ax.axvline(x=score, color="white", linewidth=3, zorder=5)
    ax.plot(score, 0, "o", color="white", markersize=9, zorder=6)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["-1.0\nNegative", "-0.5", "0\nNeutral", "0.5", "+1.0\nPositive"],
                       color="#6b7280", fontsize=8)
    for spine in ax.spines.values():
        spine.set_color("#252836")
    ax.set_title("Polarity Gauge", color="#e8eaf0", fontsize=10, pad=6)
    fig.tight_layout()
    return fig


def make_sentence_chart(sentences_data):
    if not sentences_data:
        return None
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(sentences_data) * 0.55 + 0.5)))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")

    labels = [f"S{i+1}" for i in range(len(sentences_data))]
    scores = [d["score"] for d in sentences_data]
    colors = [sentiment_color(d["label"]) for d in sentences_data]

    bars = ax.barh(labels, scores, color=colors, height=0.5, edgecolor="none")
    ax.axvline(x=0, color="#6b7280", linewidth=1, linestyle="--")
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel("Polarity", color="#6b7280", fontsize=9)
    ax.tick_params(colors="#6b7280", labelsize=9)
    ax.tick_params(axis="y", colors="#e8eaf0")
    for spine in ax.spines.values():
        spine.set_color("#252836")
    ax.set_title("Per-Sentence Polarity", color="#e8eaf0", fontsize=11, pad=10)

    patches = [
        mpatches.Patch(color="#22d07a", label="Positive"),
        mpatches.Patch(color="#f0b429", label="Neutral"),
        mpatches.Patch(color="#ff4d6d", label="Negative"),
    ]
    ax.legend(handles=patches, loc="lower right", facecolor="#252836",
              edgecolor="#252836", labelcolor="#e8eaf0", fontsize=8)
    fig.tight_layout()
    return fig


# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 SentimentIQ")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    threshold = st.slider("Positive/Negative Threshold", 0.0, 0.5, 0.1, 0.05,
                          help="Scores above +threshold = Positive, below -threshold = Negative")

    show_subjectivity = st.checkbox("Show Subjectivity Score", value=True)
    show_sentence_table = st.checkbox("Show Sentence Table", value=True)

    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown("""
- **Engine**: TextBlob NLP  
- **Language**: Python 3  
- **Charts**: Matplotlib  
- **UI**: Streamlit  
    """)
    st.markdown("---")
    st.caption("SentimentIQ v1.0 · 2024")


# ─── Main App ──────────────────────────────────────────────────
st.markdown("# Sentiment Analysis App")
if not HAS_TEXTBLOB:
    st.warning("`textblob` package not found. Using built-in fallback analyzer. For best accuracy, install `textblob` in your environment and make sure it is listed in requirements.txt.")
    st.markdown("Analyze the emotional tone of any text using ** fallback heuristic **.")
else:
    st.markdown("Analyze the emotional tone of any text using **TextBlob NLP**.")
st.markdown("---")

# Example buttons
st.markdown("**Try an example:**")
col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
examples = {
    "😊 Positive": "This product is absolutely amazing! The quality exceeded all my expectations. I love how intuitive it is, and the customer support was incredibly helpful. Highly recommend!",
    "😡 Negative": "Terrible experience from start to finish. The product broke after two days. Customer service was rude and unhelpful. I wasted my money and am deeply disappointed. Avoid at all costs.",
    "😐 Neutral":  "The package arrived on time. It contains the items listed on the website. The instructions were included in the box.",
    "🤔 Mixed":    "The laptop has a beautiful display and decent battery life. However, the keyboard feels cheap and the fan noise is quite distracting. It is okay for the price, but not exceptional.",
}

if col_ex1.button("😊 Positive"): st.session_state["text_input"] = examples["😊 Positive"]
if col_ex2.button("😡 Negative"): st.session_state["text_input"] = examples["😡 Negative"]
if col_ex3.button("😐 Neutral"):  st.session_state["text_input"] = examples["😐 Neutral"]
if col_ex4.button("🤔 Mixed"):    st.session_state["text_input"] = examples["🤔 Mixed"]

# Text input
text = st.text_area(
    "📝 Enter your text below:",
    value=st.session_state.get("text_input", ""),
    height=140,
    placeholder="Paste a review, tweet, feedback, or any text here...",
    key="main_input",
)

col_btn1, col_btn2 = st.columns([1, 5])
analyze_clicked = col_btn1.button("▶ Analyze", use_container_width=True)
if col_btn2.button("✕ Clear"):
    st.session_state["text_input"] = ""
    st.rerun()

# ─── Analysis ──────────────────────────────────────────────────
if analyze_clicked and text.strip():

    blob = TextBlob(text)
    score   = blob.sentiment.polarity
    subj    = blob.sentiment.subjectivity
    percent = round((score + 1) * 50, 1)

    # Dynamic threshold
    if score > threshold:
        label, emoji, css_key = "Positive", "😊", "positive"
    elif score < -threshold:
        label, emoji, css_key = "Negative", "😡", "negative"
    else:
        label, emoji, css_key = "Neutral", "😐", "neutral"

    # ── Verdict Banner ──
    st.markdown("---")
    st.markdown(f"""
    <div class="verdict-{css_key}">
      <span style="font-size:2.5rem">{emoji}</span>&nbsp;&nbsp;
      <span style="font-size:1.5rem;font-weight:700;color:{sentiment_color(label)}">{label}</span>
      &nbsp;<span style="color:#6b7280;font-size:0.9rem">— Overall sentiment detected</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    # ── Metrics ──
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Polarity Score",    f"{score:+.3f}")
    mc2.metric("Sentiment %",       f"{percent:.1f}%")
    mc3.metric("Sentences",         len(blob.sentences))
    if show_subjectivity:
        mc4.metric("Subjectivity",  f"{subj:.3f}")

    st.markdown("---")

    # ── Charts Row 1 ──
    st.markdown("### 📊 Visual Analysis")
    ch1, ch2 = st.columns([3, 2])

    pos_val = max(0, score)
    neg_val = max(0, -score)
    neu_val = max(0, 1 - abs(score))

    with ch1:
        st.pyplot(make_bar_chart(pos_val, neu_val, neg_val))
    with ch2:
        st.pyplot(make_pie_chart(pos_val, neu_val, neg_val))

    # ── Gauge ──
    st.pyplot(make_gauge(score))

    # ── Sentence Analysis ──
    st.markdown("---")
    st.markdown("### 🔍 Sentence-Level Analysis")

    sentences_data = []
    for i, sentence in enumerate(blob.sentences):
        s_score = sentence.sentiment.polarity
        s_label, s_emoji, s_css = (
            ("Positive", "😊", "pos") if s_score > threshold
            else ("Negative", "😡", "neg") if s_score < -threshold
            else ("Neutral", "😐", "neu")
        )
        sentences_data.append({
            "index": i + 1,
            "sentence": str(sentence).strip(),
            "score": round(s_score, 3),
            "label": s_label,
            "emoji": s_emoji,
            "css": s_css,
        })

        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:12px;padding:10px 0;'
            f'border-bottom:1px solid #252836;">'
            f'<span style="color:#6b7280;font-size:0.75rem;min-width:24px;margin-top:2px">{i+1}.</span>'
            f'<span style="flex:1;font-size:0.88rem;line-height:1.6">{str(sentence).strip()}</span>'
            f'<span class="badge-{s_css}">{s_emoji} {s_label}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Sentence bar chart
    if len(sentences_data) > 1:
        st.markdown("")
        st.pyplot(make_sentence_chart(sentences_data))

    # ── Sentence Table ──
    if show_sentence_table and sentences_data:
        st.markdown("---")
        st.markdown("### 📋 Sentence Data Table")
        df = pd.DataFrame([{
            "#":         d["index"],
            "Sentence":  d["sentence"][:80] + ("…" if len(d["sentence"]) > 80 else ""),
            "Polarity":  d["score"],
            "Sentiment": d["emoji"] + " " + d["label"],
        } for d in sentences_data])
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Word Stats ──
    st.markdown("---")
    st.markdown("### 📝 Text Statistics")
    ws1, ws2, ws3, ws4 = st.columns(4)
    words = text.split()
    ws1.metric("Word Count",      len(words))
    ws2.metric("Character Count", len(text))
    ws3.metric("Sentence Count",  len(blob.sentences))
    ws4.metric("Avg Words / Sent", f"{len(words)/max(len(blob.sentences),1):.1f}")

elif analyze_clicked and not text.strip():
    st.warning("⚠️ Please enter some text before clicking Analyze.")

else:
    # Placeholder state
    st.markdown("---")
    st.info("👆 Enter text above and click **▶ Analyze** to see results.")
