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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_story(description, age_group="3–5", story_style="🐉 Magical"):
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TEXT_MODEL).to(DEVICE).eval()

    max_tokens = 250 if age_group == "3–5" else 350

    style_map = {
        "🐉 Magical": "magical adventure with gentle magic",
        "🚀 Adventure": "fun and exciting journey",
        "🐾 Animal": "heartwarming animal story",
        "😂 Funny": "silly and humorous story",
    }
    style_desc = style_map.get(story_style, "warm children's story")

    # STRICT SYSTEM PROMPT TO FORCE NARRATIVE STRUCTURE
    system_prompt = (
        "You are a children's story writer. You write real, structured stories with a clear character, "
        "a specific mini-adventure, and a satisfying conclusion. "
        "Do NOT write generic descriptions or list magical things. Tell a simple step-by-step story."
    )

    user_prompt = f"""Write a short {style_desc} for kids aged {age_group}.

Topic: {description}

Follow this strict story outline:
1. Introduce ONE main character with a name.
2. The character starts doing something related to: {description}.
3. A small problem or curious thing happens.
4. The character solves it gently and feels happy at the end.

Keep sentences simple and natural."""

    # Format using chat template if model supports it
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback for base models
        prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nStory:"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.4,          # Lower temp prevents random topic jumping
            top_p=0.85,               # Focused vocabulary selection
            repetition_penalty=1.2,   # Stops generic phrase loops
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extract generated output
    generated_tokens = output[0][prompt_len:]
    story = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Truncate at last complete sentence
    last_punct = max(story.rfind("."), story.rfind("!"), story.rfind("?"))
    if last_punct != -1:
        story = story[: last_punct + 1]

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
