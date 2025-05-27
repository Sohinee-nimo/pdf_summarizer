import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
from gtts import gTTS
import tempfile
import base64
import os

# Initialize summarizer
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

# Function to extract text from PDF
def extract_text(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to convert text to audio and return playable HTML
def text_to_audio(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        return tmp.name

def get_audio_html(audio_path):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    return f'<audio controls><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'

# Streamlit App UI
st.title("📄 PDF Summarizer with Voice Output")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        temp_pdf_path = tmp_file.name

    st.info("Extracting text...")
    text = extract_text(temp_pdf_path)

    if not text.strip():
        st.warning("No text found in the PDF.")
    else:
        st.info("Summarizing text...")
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

        st.subheader("📝 Summary:")
        st.write(summary)

        st.info("Converting summary to voice...")
        audio_path = text_to_audio(summary)

        st.markdown("### 🔊 Play Summary:")
        st.markdown(get_audio_html(audio_path), unsafe_allow_html=True)

        st.success("✅ Done! Summary audio generated.")
