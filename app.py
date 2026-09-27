# ============================================================
# Kids Story Generator - Image -> Story -> Audio
# Optimized for Streamlit Cloud
# ============================================================
import io
import traceback
import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)

# Force CPU to reduce memory usage on Streamlit Cloud
device = torch.device("cpu")


# -----------------------------------------------------------
# Function: load_models
# Purpose: Load and cache all AI models (captioning, story, TTS)
# -----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all models. Cached so they are loaded only once per session."""
    # 1. Image captioning model (BLIP)
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float32,
    ).to(device)

    # 2. Story generation model (FLAN-T5)
    text_model_id = "google/flan-t5-small"
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    text_model = AutoModelForSeq2SeqLM.from_pretrained(
        text_model_id,
        torch_dtype=torch.float32,
    ).to(device)

    # 3. Text-to-speech model (MMS-TTS, native transformers support)
    tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")

    return blip_processor, blip_model, text_tokenizer, text_model, tts


# -----------------------------------------------------------
# Function: img2text
# Purpose: Generate a caption from an uploaded image using BLIP
# -----------------------------------------------------------
def img2text(image_file, blip_processor, blip_model):
    """Generate a short caption from the uploaded image."""
    raw_image = Image.open(image_file).convert("RGB")
    inputs = blip_processor(raw_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=20)
    return blip_processor.decode(out[0], skip_special_tokens=True)


# -----------------------------------------------------------
# Function: generate_story
# Purpose: Generate a bedtime story based on the image caption
# -----------------------------------------------------------
def generate_story(caption, text_tokenizer, text_model):
    """Expand the caption into a gentle bedtime story."""

    def run_prompt(prompt):
        inputs = text_tokenizer(prompt, return_tensors="pt").to(device)
        output = text_model.generate(
            **inputs,
            max_new_tokens=180,
            min_length=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
        return text_tokenizer.decode(output[0], skip_special_tokens=True).strip()

    prompt = (
        f"Once upon a time, my dear, let me tell you a gentle bedtime story. "
        f"This story is about {caption}. "
        f"It should sound like a parent speaking softly to their child, "
        f"with a clear beginning, middle, and a happy ending. "
        f"End with a comforting line such as "
        f"'and now you can rest peacefully, knowing everything is safe and happy.'"
    )
    story = run_prompt(prompt)

    # Retry if the story contains unwanted meta-words
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
# Purpose: Convert the generated story into a WAV audio buffer
# -----------------------------------------------------------
def story_to_audio(story_text, tts):
    """Convert story text into a single WAV byte buffer."""
    if tts is None:
        return None
    try:
        # Split story into sentence chunks
        sentences = story_text.replace("\n", " ").split(". ")
        audio_chunks = []
        sample_rate = None

        for chunk in sentences:
            chunk = chunk.strip()
            if not chunk:
                continue
            audio_out = tts(chunk)
            samples = np.array(audio_out["audio"]).squeeze()
            sample_rate = audio_out["sampling_rate"]
            audio_chunks.append(samples)

        if not audio_chunks:
            return None

        # Concatenate all audio chunks
        full_audio = np.concatenate(audio_chunks)

        # Write to an in-memory WAV buffer
        buf = io.BytesIO()
        wavfile.write(buf, sample_rate, full_audio)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        # Show full traceback for debugging
        st.error(f"TTS error: {str(e)}")
        st.code(traceback.format_exc())
        return None


# -----------------------------------------------------------
# Function: main
# Purpose: Build the Streamlit UI
# -----------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Kids Story Generator",
        page_icon="📖",
        layout="centered",
    )

    # Kid-friendly header
    st.markdown(
        "<h1 style='text-align:center; color:#FF69B4;'>🌟 Magical Storytime 🌟</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#228B22; font-size:22px;'>"
        "Upload a picture and let's create a bedtime adventure together!"
        "</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "📷 Choose a fun picture", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="✨ Your Picture ✨", use_container_width=True)

        if st.button("🎉 Generate Story"):
            # Lazy load models only when needed
            with st.spinner("Loading models, this may take a few minutes..."):
                try:
                    (
                        blip_processor,
                        blip_model,
                        text_tokenizer,
                        text_model,
                        tts,
                    ) = load_models()
                except Exception as e:
                    st.error(f"Model loading failed: {str(e)}")
                    st.code(traceback.format_exc())
                    st.stop()

            # Step 1: Image -> Caption
            with st.spinner("Looking at your picture..."):
                try:
                    caption = img2text(uploaded_file, blip_processor, blip_model)
                    st.success(f"📝 Magic Caption: {caption}")
                except Exception as e:
                    st.error(f"Caption generation failed: {str(e)}")
                    st.code(traceback.format_exc())
                    st.stop()

            # Step 2: Caption -> Story
            with st.spinner(
                "✨ Hold on tight! Your magical bedtime story is being written... ✨"
            ):
                try:
                    story = generate_story(caption, text_tokenizer, text_model)
                except Exception as e:
                    st.error(f"Story generation failed: {str(e)}")
                    st.code(traceback.format_exc())
                    st.stop()

            st.markdown(
                f"<div style='background-color:#FFFACD; padding:20px; "
                f"border-radius:15px; font-size:18px;'>"
                f"<b>📖 Your Story:</b><br>{story}</div>",
                unsafe_allow_html=True,
            )

            # Step 3: Story -> Audio
            with st.spinner("Generating audio..."):
                audio_bytes = story_to_audio(story, tts)

            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                st.info("🔊 Sit back, relax, and listen to your magical story!")
            else:
                st.warning(
                    "🔊 Audio unavailable right now, but you can enjoy reading the story!"
                )


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    main()
