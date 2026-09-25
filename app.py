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

VISION_MODEL = "Salesforce/blip-image-captioning-base"
TEXT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
TTS_MODEL = "microsoft/speecht5_tts"
TTS_VOCODER = "microsoft/speecht5_hifigan"
SPEAKER_REPO = "Matthijs/cmu-arctic-xvectors"
SPEAKER_INDEX = 7306
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="My Story Maker", page_icon="🌈", layout="centered")

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

def image_to_text(image):
    processor = BlipProcessor.from_pretrained(VISION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(VISION_MODEL).to(DEVICE).eval()
    inputs = processor(images=image, text="a picture of", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=40, num_beams=3)
    return processor.decode(output[0], skip_special_tokens=True).strip()

def generate_story(description, age_group, style):
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TEXT_MODEL).to(DEVICE).eval()

    lengths = {
        "3–5": (150, "tell a gentle bedtime story in 4–6 sentences."),
        "6–7": (180, "tell a playful story in 6–8 sentences."),
        "8–10": (220, "tell an imaginative story in 8–10 sentences."),
    }
    max_tokens, length_instruction = lengths.get(age_group, (180, ""))

    styles = {
        "🐉 Magical": "make it a magical adventure.",
        "🚀 Adventure": "make it a fun adventure.",
        "🐾 Animal": "include friendly animals.",
        "😂 Funny": "make it silly and funny.",
    }
    style_instruction = styles.get(style, "make it warm and cheerful.")

    prompt = f"""
once upon a time, {description}.
{length_instruction}
{style_instruction}
use simple words and warm feelings. do not repeat instructions. begin directly with the story. end with a happy or reassuring feeling.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            num_beams=4,
            min_length=80,
        )

    story = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # filter out instruction echoes
    unwanted = ["tell a", "use simple", "do not repeat", "end with"]
    for marker in unwanted:
        if marker in story.lower():
            story = story.split(marker)[0].strip()

    # ensure it starts like a story
    if not story.lower().startswith("once upon a time"):
        story = "once upon a time, " + story

    # ensure it ends nicely
    if not story.endswith((".", "!", "?")):
        story += " everyone was happy at the end."

    return story

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

def reset_story():
    for key in ["description", "story", "audio"]:
        st.session_state.pop(key, None)

def main():
    st.markdown('<div class="title">🌈 my story maker</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">turn a picture into a magical story!</div>', unsafe_allow_html=True)

    st.subheader("✨ choose your story")
    col1, col2 = st.columns(2)
    age_group = col1.selectbox("age", ["3–5", "6–7", "8–10"])
    story_style = col2.selectbox("story type", ["🐉 magical", "🚀 adventure", "🐾 animal", "😂 funny"])

    st.subheader("📸 choose a picture")
    uploaded_file = st.file_uploader("upload a picture", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if not uploaded_file:
        st.info("💡 try a picture of a toy, pet, park, castle, bicycle, or drawing.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="your picture", use_container_width=True)

    if st.button("✨ make my story! ✨", type="primary", use_container_width=True):
        reset_story()
        with st.spinner("👀 looking at your picture..."):
            st.session_state["description"] = image_to_text(image)
        with st.spinner("🪄 creating your story..."):
            st.session_state["story"] = generate_story(st.session_state["description"], age_group, story_style)

    if "description" in st.session_state:
        with st.expander("👀 what i saw in the picture"):
            st.write(st.session_state["description"])

    if "story" in st.session_state:
        st.subheader("📖 your story")
        st.write(st.session_state["story"])

        st.subheader("🔊 listen to your story")
        if st.button("🎵 read my story", use_container_width=True):
            with st.spinner("🎵 making the audio..."):
                st.session_state["audio"] = text_to_speech(st.session_state["story"])
            st.success("🎉 your story is ready!")

        if st.session_state.get("audio"):
            st.audio(st.session_state["audio"], format="audio/wav")
            st.download_button("⬇️ download audio", st.session_state["audio"], "my_story.wav", "audio/wav")

        if st.button("🌟 make another story", use_container_width=True):
            reset_story()
            st.rerun()

if __name__ == "__main__":
    main()
