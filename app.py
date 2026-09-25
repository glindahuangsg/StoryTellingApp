import streamlit as st
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from PIL import Image
import io
import soundfile as sf

# ✅ Force CPU mode to reduce memory usage
device = torch.device("cpu")

# -----------------------------------------------------------
# Function: load_models
# Purpose: Load and cache all AI models (image captioning, text generation, TTS)
# -----------------------------------------------------------
@st.cache_resource
def load_models():
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float32
    ).to(device)

    text_model_id = "google/flan-t5-small"
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    text_model = AutoModelForSeq2SeqLM.from_pretrained(
        text_model_id,
        torch_dtype=torch.float32
    ).to(device)

    try:
        tts = pipeline("text-to-speech", model="espnet/kan-bayashi_ljspeech_vits")
    except Exception:
        tts = None
    return blip_processor, blip_model, text_tokenizer, text_model, tts

blip_processor, blip_model, text_tokenizer, text_model, tts = load_models()

# -----------------------------------------------------------
# Function: img2text
# Purpose: Generate a caption from an uploaded image using BLIP
# -----------------------------------------------------------
def img2text(image_file):
    raw_image = Image.open(image_file).convert("RGB")
    inputs = blip_processor(raw_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=20)
    return blip_processor.decode(out[0], skip_special_tokens=True)

# -----------------------------------------------------------
# Function: generate_story
# Purpose: Generate a bedtime story based on the image caption
# -----------------------------------------------------------
def generate_story(caption, text_tokenizer=text_tokenizer, text_model=text_model, device=device):
    def run_prompt(prompt):
        inputs = text_tokenizer(prompt, return_tensors="pt").to(device)
        output = text_model.generate(
            **inputs,
            max_new_tokens=180,
            min_length=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )
        return text_tokenizer.decode(output[0], skip_special_tokens=True).strip()

    prompt = (
        f"Once upon a time, my dear, let me tell you a gentle bedtime story. "
        f"This story is about {caption}. "
        f"It should sound like a parent speaking softly to their child, "
        f"with a clear beginning, middle, and a happy ending. "
        f"End with a comforting line such as 'and now you can rest peacefully, knowing everything is safe and happy.'"
    )
    story = run_prompt(prompt)

    bad_phrases = ["series", "post", "collection", "book", "illustration"]
    if any(bp in story.lower() for bp in bad_phrases):
        retry_prompt = (
            f"Once upon a time, my dear, there was {caption}. "
            f"Tell it as a short bedtime story in a parent's gentle voice, "
            f"ending with comfort and happiness."
        )
        story = run_prompt(retry_prompt)

    return story.strip()

# -----------------------------------------------------------
# Function: story_to_audio
# Purpose: Convert the generated story into audio using TTS
# -----------------------------------------------------------
def story_to_audio(story_text):
    if tts is None:
        return None
    try:
        sentences = story_text.split(". ")
        audio_buffers = []
        for chunk in sentences:
            if not chunk.strip():
                continue
            audio_out = tts(chunk.strip())
            samples = audio_out["audio"]
            rate = audio_out["sampling_rate"]
            buf = io.BytesIO()
            sf.write(buf, samples, rate, format="WAV")
            buf.seek(0)
            audio_buffers.append(buf.read())
        return b"".join(audio_buffers)
    except Exception:
        return None

# -----------------------------------------------------------
# Function: main
# Purpose: Build the Streamlit UI for the app
# -----------------------------------------------------------
def main():
    st.set_page_config(page_title="Kids Story Generator", page_icon="📖", layout="centered")

    # 🎨 Kid-friendly header
    st.markdown(
        "<h1 style='text-align:center; color:#FF69B4;'>🌟 Magical Storytime 🌟</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#228B22; font-size:22px;'>Upload a picture and let’s create a bedtime adventure together!</p>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("📷 Choose a fun picture", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="✨ Your Picture ✨", use_container_width=True)

        if st.button("🎉 Generate Story"):
            caption = img2text(uploaded_file)
            st.success(f"📝 Magic Caption: {caption}")

            # 🕒 Show friendly waiting message while story loads
            with st.spinner("✨ Hold on tight! Your magical bedtime story is being written... ✨"):
                story = generate_story(caption)

            # Display story in a playful box
            st.markdown(
                f"<div style='background-color:#FFFACD; padding:20px; border-radius:15px; font-size:18px;'>"
                f"<b>📖 Your Story:</b><br>{story}</div>",
                unsafe_allow_html=True
            )

            audio_bytes = story_to_audio(story)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                st.info("🔊 Sit back, relax, and listen to your magical story!")
            else:
                st.warning("🔊 Audio unavailable right now, but you can enjoy reading the story!")

# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    main()
