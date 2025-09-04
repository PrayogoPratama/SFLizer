# SFLizer_1_04.py
# Author: Prayogo Adi Putra Pratama
# License: Apache 2.0

import streamlit as st
import spacy
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # safe backend
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import PyPDF2

# ---------------- Config ----------------
st.set_page_config(page_title="SFLizer v1.04", page_icon="📘", layout="wide")
st.title("📘 SFLizer v1.04")
st.caption("POS + Lemmatization + KWIC + SFL Mapping (stable version)")

# ---------------- Load spaCy ----------------
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# ---------------- Extract PDF ----------------
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = " ".join([page.extract_text() or "" for page in reader.pages])
    return text.strip()

# ---------------- KWIC helper ----------------
def kwic(doc, lemma, pos_tag, window=5, max_rows=50):
    tokens = list(doc)
    hits = [i for i,t in enumerate(tokens) if t.lemma_.lower()==lemma.lower() and t.pos_==pos_tag]
    rows=[]
    for i in hits[:max_rows]:
        left = " ".join(tok.text for tok in tokens[max(0,i-window):i])
        kw = tokens[i].text
        right = " ".join(tok.text for tok in tokens[i+1:i+1+window])
        rows.append({"Left": left, "Keyword": kw, "Right": right})
    return pd.DataFrame(rows)

# ---------------- Analysis functions ----------------
def analyze_pos(doc, pos_tag, label, min_len=2, remove_stop=True):
    items=[]
    for t in doc:
        if t.pos_ == pos_tag and t.is_alpha:
            if remove_stop and t.is_stop:
                continue
            lemma = t.lemma_.lower()
            if len(lemma) >= min_len:
                items.append(lemma)
    counter = Counter(items)
    df = pd.DataFrame(counter.most_common(), columns=[label,"Frequency"])
    return df

def lemma_forms(doc, lemmas, pos_tag):
    forms = defaultdict(Counter)
    for t in doc:
        if t.pos_==pos_tag and t.lemma_.lower() in lemmas:
            forms[t.lemma_.lower()][t.text] += 1
    rows=[]
    for lm,ctr in forms.items():
        for form,freq in ctr.items():
            rows.append({"Lemma": lm,"Form": form,"Frequency": freq})
    return pd.DataFrame(rows)

# ---------------- Sidebar ----------------
menu = st.sidebar.radio("Menu", ["Home","Verb","Noun","Adjective","Adverb"])
uploaded = st.sidebar.file_uploader("Upload PDF", type="pdf")
top_n = st.sidebar.slider("Top N", 10, 100, 30, 5)
kwic_window = st.sidebar.slider("KWIC window", 2, 6, 4)

# ---------------- Main ----------------
if not uploaded:
    st.info("Upload a PDF from sidebar to start.")
else:
    text = extract_text(uploaded)
    doc = nlp(text)
    st.success(f"PDF loaded: {uploaded.name} ({len(text):,} chars)")

    if menu=="Home":
        st.subheader("Welcome to SFLizer v1.04")
        st.write("Choose Verb / Noun / Adjective / Adverb from sidebar.")

    elif menu in ["Verb","Noun","Adjective","Adverb"]:
        pos_map = {
            "Verb":"VERB",
            "Noun":"NOUN",
            "Adjective":"ADJ",
            "Adverb":"ADV"
        }
        label = menu
        pos_tag = pos_map[menu]

        st.subheader(f"{menu} Analysis")
        df = analyze_pos(doc,pos_tag,label)
        st.dataframe(df.head(top_n), use_container_width=True)

        if not df.empty:
            # chart
            fig, ax = plt.subplots()
            df.head(top_n).set_index(label)["Frequency"].plot(kind="bar", ax=ax)
            st.pyplot(fig)

            # download
            st.download_button("⬇ CSV", df.to_csv(index=False).encode(), f"{menu.lower()}.csv","text/csv")

            # lemmatization forms
            st.markdown("---")
            st.subheader("Word Forms")
            lemmas = df.head(top_n)[label].tolist()
            chosen = st.multiselect("Select lemmas", options=lemmas, default=lemmas[:3])
            if chosen:
                df_forms = lemma_forms(doc, chosen, pos_tag)
                st.dataframe(df_forms, use_container_width=True)

            # KWIC
            st.markdown("---")
            st.subheader("KWIC / Concordance")
            if lemmas:
                kw = st.selectbox("Choose lemma", ["(select)"]+lemmas)
                if kw!="(select)":
                    df_kwic = kwic(doc, kw, pos_tag, window=kwic_window)
                    st.dataframe(df_kwic, use_container_width=True)
