import io
import re
import gc
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)
from huggingface_hub import hf_hub_download

# ============================================================
# CONFIGURATION
# ============================================================

VISION_MODEL = "Salesforce/blip-image-captioning-base"
TEXT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
TTS_MODEL = "microsoft/speecht5_tts"
TTS_VOCODER = "microsoft/speecht5_hifigan"
SPEAKER_REPO = "Matthijs/cmu-arctic-xvectors"
SPEAKER_INDEX = 7306
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="My Story Maker", page_icon="🌈", layout="centered")

# ============================================================
# UTILITIES
# ============================================================

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@st.cache_resource
def load_speaker_embedding():
    zip_path = hf_hub_download(repo_id=SPEAKER_REPO, filename="spkrec-xvect.zip", repo_type="dataset")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as archive:
        npy_files = sorted([f for f in archive.namelist() if f.endswith(".npy")])
        selected = npy_files[SPEAKER_INDEX]
        with archive.open(selected) as f:
            emb = np.load(f)
    return torch.tensor(emb, dtype=torch.float32).unsqueeze(0)

# ============================================================
# IMAGE → DESCRIPTION
# ============================================================

def image_to_text(image):
    processor = BlipProcessor.from_pretrained(VISION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(VISION_MODEL).to(DEVICE).eval()
    inputs = processor(images=image, text="a picture of", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=40, num_beams=3)
    return processor.decode(output[0], skip_special_tokens=True).strip()

# ============================================================
# STORY GENERATION (child‑friendly, no echoes)
# ============================================================

def generate_story(description, age_group, style):
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TEXT_MODEL).to(DEVICE).eval()

    lengths = {
        "3–5": (120, "Write 3–5 very short sentences with simple words."),
        "6–7": (160, "Write 5–7 short sentences with playful descriptions."),
        "8–10": (200, "Write 7–10 sentences with imaginative language."),
    }
    max_tokens, length_instruction = lengths.get(age_group, (150, ""))

    styles = {
        "🐉 Magical": "Make it a gentle magical adventure.",
        "🚀 Adventure": "Make it a fun and safe adventure.",
        "🐾 Animal": "Make friendly animals important characters.",
        "😂 Funny": "Include something silly and funny.",
    }
    style_instruction = styles.get(style, "Make it a warm children's story.")

    prompt = f"""
Tell a children's story based on this picture: {description}.
The child is {age_group} years old.
{length_instruction}
{style_instruction}
Use simple, warm, imaginative language. Do not repeat instructions. Begin directly with the story.
End with a happy or reassuring feeling.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                                temperature=0.7, top_p=0.9, repetition_penalty=1.05)
    story = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # Remove any leftover instruction echoes
    for marker in ["Tell a children's story", "Rules:", "The child is"]:
        if marker in story:
            story = story.split(marker)[-1].strip()

    if not story.endswith((".", "!", "?")):
        story += " And everyone was happy at the end."
    return story

# ============================================================
# TEXT → SPEECH
# ============================================================

def text_to_speech(text):
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    processor = SpeechT5Processor.from_pretrained(TTS_MODEL)
    model = SpeechT5ForTextToSpeech.from_pretrained(TTS_MODEL).to(DEVICE).eval()
    vocoder = SpeechT5HifiGan.from_pretrained(TTS_VOCODER).to(DEVICE).eval()
    speaker = load_speaker_embedding().to(DEVICE)

    audio = []
    for chunk in chunks:
        if not chunk: continue
        inputs = processor(text=chunk, return_tensors="pt")
        input_ids = inputs["input_ids"].to(DEVICE)
        with torch.no_grad():
            speech = model.generate_speech(input_ids, speaker, vocoder=vocoder)
        audio.append(speech.cpu().numpy())
    combined = np.concatenate(audio)
    buf = io.BytesIO()
    sf.write(buf, combined, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf.read()

# ============================================================
# RESET
# ============================================================

def reset_story():
    for key in ["description", "story", "audio"]:
        st.session_state.pop(key, None)

# ============================================================
# MAIN APP
# ============================================================

def main():
    st.markdown('<div class="title">🌈 My Story Maker</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Turn a picture into a magical story!</div>', unsafe_allow_html=True)

    st.subheader("✨ Choose your story")
    col1, col2 = st.columns(2)
    age_group = col1.selectbox("Age", ["3–5", "6–7", "8–10"])
    story_style = col2.selectbox("Story type", ["🐉 Magical", "🚀 Adventure", "🐾 Animal", "😂 Funny"])

    st.subheader("📸 Choose a picture")
    uploaded_file = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if not uploaded_file:
        st.info("💡 Try a picture of a toy, pet, park, castle, bicycle, or drawing.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your picture", use_container_width=True)

    if st.button("✨ Make My Story! ✨", type="primary", use_container_width=True):
        reset_story()
        with st.spinner("👀 Looking at your picture..."):
            st.session_state["description"] = image_to_text(image)
        with st.spinner("🪄 Creating your story..."):
            st.session_state["story"] = generate_story(st.session_state["description"], age_group, story_style)

    if "description" in st.session_state:
        with st.expander("👀 What I saw in the picture"):
            st.write(st.session_state["description"])

    if "story" in st.session_state:
        st.subheader("📖 Your Story")
        st.markdown('<div class="story-box">', unsafe_allow_html=True)
        st.write(st.session_state["story"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("🔊 Listen to your story")
        if st.button("🎵 Read My Story", use_container_width=True):
            with st.spinner("🎵 Making the audio..."):
                st.session_state["audio"] = text_to_speech(st.session_state["story"])
            st.success("🎉 Your story is ready!")

        if st.session_state.get("audio"):
            st.audio(st.session_state["audio"], format="audio/wav")
            st.download_button("⬇️ Download audio", st.session_state["audio"], "my_story.wav", "audio/wav")

        if st.button("🌟 Make Another Story", use_container_width=True):
            reset_story()
            st.rerun()

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
