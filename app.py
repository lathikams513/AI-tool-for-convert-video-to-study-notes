import streamlit as st
import os
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline

# --- Page Title ---
st.set_page_config(page_title="🎀 Video to Cute Notes Converter", layout="wide")
st.title("🎀 Video to Cute Notes Converter")

# --- Sidebar ---
st.sidebar.header("📚 Quick Access")
st.sidebar.markdown("""
- Upload video
- Transcribe audio
- Generate cute notes
- Download notes
""")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ Video uploaded successfully!")

    # --- Step 1: Extract Audio ---
    st.info("🎵 Extracting audio...")
    video = VideoFileClip("temp_video.mp4")
    video.audio.write_audiofile("temp_audio.wav")
    st.success("✅ Audio extracted!")

    # --- Step 2: Transcribe Audio ---
    st.info("📝 Converting audio to text...")
    model = whisper.load_model("base")
    result = model.transcribe("temp_audio.wav", fp16=False)
    transcript = result["text"]
    st.text_area("🗣 Transcript", transcript, height=200)

    # --- Step 3: Summarize & Make Cute Notes ---
    st.info("✨ Making Cute Notes...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(transcript, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

    # Convert summary into side headings + bullet points
    sentences = summary.split(". ")
    cute_notes = "🌸 **Cute Notes** 🌸\n\n"
    for i, s in enumerate(sentences, start=1):
        if s.strip():
            cute_notes += f"### 🏷 Point {i}\n- {s.strip()}.\n\n"

    st.markdown(cute_notes)

    # --- Download Button ---
    with open("cute_notes.txt", "w", encoding="utf-8") as f:
        f.write(cute_notes)

    with open("cute_notes.txt", "rb") as f:
        st.download_button("📥 Download Notes", f, file_name="cute_notes.txt")

# --- Extra Cute Buttons Section ---
st.markdown("---")
st.markdown("### 🥰 Explore More Notes")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🐍 Python Notes"):
        st.success("""
**Python Notes:**
- Easy to learn and versatile
- Interpreted language
- Supports OOP, Functional, Procedural styles
""")

with col2:
    if st.button("📊 Data Science Notes"):
        st.info("""
**Data Science Notes:**
- Extract insights from data
- Tools: Pandas, Numpy, Matplotlib
- Steps: Collect → Clean → Analyze → Visualize
""")

with col3:
    if st.button("🤖 AI/ML Notes"):
        st.warning("""
**AI/ML Notes:**
- AI: Machines that think
- ML: Machines that learn from data
- Algorithms: Regression, Classification, Clustering
""")
