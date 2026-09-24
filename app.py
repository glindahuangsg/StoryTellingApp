"""
Story Time - Simple AI Storytelling App
----------------------------------------
Turns a photo into a short story you can listen to.

How it works:
  1. Upload a picture.
  2. An AI model looks at the picture and writes a short caption.
  3. Another AI model turns that caption into a short story.
  4. A third AI model reads the story out loud.
"""

import io

import numpy as np
import soundfile as sf
import streamlit as st
from PIL import Image
from transformers import pipeline


# ---- Load the three AI models -----------------------------------------------
# @st.cache_resource means each model is only actually loaded ONE time, then
# reused, instead of being re-downloaded every time a button is clicked.

@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


@st.cache_resource
def load_story_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")


@st.cache_resource
def load_audio_model():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")


# ---- The three steps of the pipeline ----------------------------------------

def image_to_caption(image):
    """Look at the image and describe it in words."""
    caption_model = load_caption_model()
    result = caption_model(image)
    return result[0]["generated_text"]


def caption_to_story(caption):
    """Turn the caption into a short children's story."""
    story_model = load_story_model()
    prompt = (
        "Write a short, gentle story for young children based on this scene: "
        f"'{caption}'. The story should be about 50 to 100 words long."
    )
    result = story_model(
        prompt,
        max_new_tokens=180,
        min_new_tokens=60,
        # These two settings stop the model from getting "stuck" and looping
        # the same phrase over and over when pushed past what it would
        # naturally write (which is what min_new_tokens above does).
        no_repeat_ngram_size=3,   # never repeat the same 3-word phrase twice
        repetition_penalty=1.3,   # discourage reusing recent words in general
    )
    return result[0]["generated_text"]


def story_to_audio(story):
    """Turn the story text into playable audio (WAV bytes)."""
    audio_model = load_audio_model()
    result = audio_model(story)

    # The model gives back numbers representing sound, plus a sample rate.
    # np.squeeze removes an extra "wrapper" layer around those numbers.
    audio_array = np.squeeze(result["audio"])
    sample_rate = result["sampling_rate"]

    # Write those numbers into a real WAV file, kept in memory (not on disk).
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_array, sample_rate, format="WAV")
    return audio_buffer.getvalue()


# ---- The app itself ----------------------------------------------------------

def main():
    """Draw the page and run the pipeline when the user clicks the button."""
    st.set_page_config(page_title="Story Time", page_icon="📖")
    st.title("📖 Story Time")
    st.write("Upload a picture and turn it into a story you can listen to!")

    uploaded_file = st.file_uploader("Upload a picture", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your picture", width=300)

        if st.button("Make My Story!"):
            with st.spinner("Looking at your picture..."):
                caption = image_to_caption(image)
            st.write("**Caption:**", caption)

            with st.spinner("Writing your story..."):
                story = caption_to_story(caption)
            st.write("**Story:**")
            st.write(story)

            with st.spinner("Recording the story..."):
                audio_bytes = story_to_audio(story)
            st.write("**Listen:**")
            st.audio(audio_bytes, format="audio/wav")


# ---- Run the app ---------------------------------------------------------------
# This "if" line is a common Python pattern: it means "only run main() when this
# file is executed directly (like `streamlit run app.py`)", not when it's imported.

if __name__ == "__main__":
    main()
