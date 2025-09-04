import streamlit as st
import fitz  # PyMuPDF
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import io
import base64

# ============== CONFIG ==============
st.set_page_config(
    page_title="SFLizer",
    page_icon="📘",
    layout="wide"
)

# ============== LOAD SPACY ==============
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# ============== SFL MAPPING ==============
sfl_map = {
    # Material
    "do":"Material","make":"Material","build":"Material","create":"Material","use":"Material",
    "support":"Material","develop":"Material","increase":"Material","reduce":"Material","reach":"Material",
    "grow":"Material","provide":"Material","help":"Material","deliver":"Material","implement":"Material",
    "manage":"Material","improve":"Material","enhance":"Material","construct":"Material","invest":"Material",
    # Mental
    "know":"Mental","see":"Mental","think":"Mental","expect":"Mental","believe":"Mental","feel":"Mental",
    "understand":"Mental","assume":"Mental","consider":"Mental","perceive":"Mental","realize":"Mental",
    # Relational
    "be":"Relational","have":"Relational","include":"Relational","remain":"Relational",
    "consist":"Relational","represent":"Relational","mean":"Relational","equal":"Relational",
    # Verbal
    "say":"Verbal","report":"Verbal","suggest":"Verbal","claim":"Verbal","explain":"Verbal",
    "mention":"Verbal","highlight":"Verbal","state":"Verbal","assert":"Verbal","communicate":"Verbal",
    # Behavioral
    "watch":"Behavioral","listen":"Behavioral","smile":"Behavioral","cry":"Behavioral","breathe":"Behavioral",
    "stare":"Behavioral","look":"Behavioral","laugh":"Behavioral","observe":"Behavioral","shrug":"Behavioral",
    # Existential
    "exist":"Existential","arise":"Existential","appear":"Existential","emerge":"Existential",
    "occur":"Existential","happen":"Existential","prevail":"Existential"
}

# ============== FUNCTIONS ==============
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return re.sub(r"[\u200b\ufeff]", "", text)

def analyze_text(text):
    doc = nlp(text)
    verbs = [(t.text.lower(), t.lemma_.lower()) for t in doc if t.pos_ in ["VERB","AUX"] and t.is_alpha]
    lemma_freq = Counter([lemma for _, lemma in verbs])

    rows = []
    for lemma, freq in lemma_freq.items():
        rows.append({
            "Verb": lemma,
            "Frequency": freq,
            "SFL Process": sfl_map.get(lemma, "Material")
        })
    df = pd.DataFrame(rows).sort_values(by="Frequency", ascending=False)
    return df

def download_button(object_to_download, download_filename, button_text, mime_type):
    b64 = base64.b64encode(object_to_download).decode()
    return f"""
        <a href="data:{mime_type};base64,{b64}" download="{download_filename}">
            <button style="background-color:#4CAF50;color:white;padding:6px 12px;border:none;border-radius:4px;cursor:pointer;">
                {button_text}
            </button>
        </a>
    """

# ============== SIDEBAR NAVIGATION ==============
st.sidebar.title("📘 SFLizer Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Analysis", "About"])

# ============== HOME ==============
if menu == "Home":
    st.title("📘 Welcome to SFLizer")
    st.markdown(
        """
        **SFLizer** is a Python-based tool for **Systemic Functional Linguistics (SFL) Analysis**.  
        Upload your PDF document and get:
        - Verb frequency analysis  
        - SFL process classification  
        - Downloadable results (CSV, TXT, JSON)  
        - Visualization of SFL distributions  

        👉 Navigate to **Analysis** in the sidebar to get started.
        """
    )

# ============== ANALYSIS ==============
elif menu == "Analysis":
    st.title("🔎 SFL Analysis")
    uploaded = st.file_uploader("Upload your PDF file", type="pdf")

    if uploaded:
        with st.spinner("Analyzing... please wait ⏳"):
            text = extract_text(uploaded)
            df = analyze_text(text)

        st.success("Analysis complete ✅")

        # Show results
        st.subheader("📊 Verb Frequency & SFL Classification")
        st.dataframe(df.head(30))

        # Plot chart
        st.subheader("📈 Distribution of SFL Processes")
        fig, ax = plt.subplots()
        df["SFL Process"].value_counts().plot(kind="bar", ax=ax, color="skyblue")
        ax.set_ylabel("Count")
        ax.set_xlabel("SFL Process")
        ax.set_title("Distribution of SFL Processes")
        st.pyplot(fig)

        # Downloads
        st.subheader("⬇️ Download Results")
        csv = df.to_csv(index=False).encode()
        json_data = df.to_json(orient="records").encode()
        txt_data = df.to_string(index=False).encode()

        st.download_button("Download CSV", csv, "sflizer_results.csv", "text/csv")
        st.download_button("Download JSON", json_data, "sflizer_results.json", "application/json")
        st.download_button("Download TXT", txt_data, "sflizer_results.txt", "text/plain")

        # Download chart
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("Download Chart (PNG)", buf.getvalue(), "sflizer_chart.png", "image/png")

# ============== ABOUT ==============
elif menu == "About":
    st.title("ℹ️ About SFLizer")
    st.markdown(
        """
        **SFLizer v1.0**  
        Developed for the *6th International Conference on Integrating Technology in Education (ITE 2025)*.  

        - Author: *Your Name*  
        - Built with: Python, Streamlit, spaCy, PyMuPDF, pandas, matplotlib  
        - License: Apache 2.0  

        This tool demonstrates how Natural Language Processing (NLP) can be integrated with **Systemic Functional Linguistics (SFL)**  
        to analyze verbs and classify them into different process types.  

        ✨ Contributions and feedback are welcome!
        """
    )
