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
    # BLIP model for image captioning
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float32
    ).to(device)

    # Instruction-tuned lightweight text model (Flan-T5-small)
    text_model_id = "google/flan-t5-small"
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    text_model = AutoModelForSeq2SeqLM.from_pretrained(
        text_model_id,
        torch_dtype=torch.float32
    ).to(device)

    # Smaller TTS model for audio generation
    try:
        tts = pipeline("text-to-speech", model="espnet/kan-bayashi_ljspeech_vits")
    except Exception:
        tts = None  # fallback mode if TTS fails

    return blip_processor, blip_model, text_tokenizer, text_model, tts

# Load models once and reuse
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
# Fix: Use natural storytelling prompt so output feels like a parent telling a story
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

    # ✅ Warm, narrative-style prompt
    prompt = (
        f"Tell a gentle bedtime story for children aged 3–10. "
        f"Begin with 'Once upon a time' and make it sound like a parent speaking softly. "
        f"The story should be about {caption}, with a beginning, middle, and happy ending."
    )
    story = run_prompt(prompt)

    # ✅ Retry with simpler narrative if output is meta-text
    bad_phrases = ["series", "post", "collection", "book"]
    if any(bp in story.lower() for bp in bad_phrases):
        retry_prompt = (
            f"Once upon a time, there was {caption}. "
            f"Tell it as a short bedtime story with a happy ending, like a parent speaking to a child."
        )
        story = run_prompt(retry_prompt)

    return story.strip()

# -----------------------------------------------------------
# Function: story_to_audio
# Purpose: Convert the generated story into audio using TTS
# Splits long stories into chunks to avoid memory issues
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
# Handles image upload, caption generation, story creation, and audio playback
# -----------------------------------------------------------
def main():
    st.set_page_config(page_title="Kids Story Generator", page_icon="📖", layout="centered")

    # Title and description
    st.markdown(
        "<h1 style='text-align:center; color:#FF69B4;'>✨ Magical Story Generator ✨</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#228B22; font-size:20px;'>Upload a picture and hear your bedtime story!</p>",
        unsafe_allow_html=True
    )

    # File uploader
    uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Picture", use_container_width=True)

        # Generate story button
        if st.button("Generate Story"):
            caption = img2text(uploaded_file)
            st.success(f"📝 Caption: {caption}")

            story = generate_story(caption)
            st.markdown(
                f"<div style='background-color:#FFFACD; padding:15px; border-radius:10px;'>"
                f"<b>📖 Story:</b><br>{story}</div>",
                unsafe_allow_html=True
            )

            # Generate audio
            audio_bytes = story_to_audio(story)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
            else:
                st.warning("🔊 Audio unavailable due to memory limits. Please enjoy reading the story!")

# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    main()
