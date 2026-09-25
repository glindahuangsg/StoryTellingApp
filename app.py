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

# Load models once
@st.cache_resource
def load_models():
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float32
    ).to(device)

    # ✅ Instruction-tuned lightweight text model
    text_model_id = "google/flan-t5-small"
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    text_model = AutoModelForSeq2SeqLM.from_pretrained(
        text_model_id,
        torch_dtype=torch.float32
    ).to(device)

    # ✅ TTS pipeline (may fail on low memory)
    try:
        tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    except Exception:
        tts = None  # fallback mode
    return blip_processor, blip_model, text_tokenizer, text_model, tts

blip_processor, blip_model, text_tokenizer, text_model, tts = load_models()

# Functions
def img2text(image_file):
    raw_image = Image.open(image_file).convert("RGB")
    inputs = blip_processor(raw_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=20)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def generate_story(caption, text_tokenizer=text_tokenizer, text_model=text_model, device=device):
    prompt = (
        f"Create a bedtime story for children aged 3–10. "
        f"Make it warm, simple, and magical. "
        f"Include a beginning, middle, and end. "
        f"Base it on this idea: {caption}. "
        f"End with a happy feeling."
    )
    inputs = text_tokenizer(prompt, return_tensors="pt").to(device)
    output = text_model.generate(
        **inputs,
        max_new_tokens=150,
        min_length=60,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )
    story = text_tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # ✅ Fallback if model outputs irrelevant or repetitive text
    bad_phrases = ["series", "post", "collection", "book"]
    if any(bp in story.lower() for bp in bad_phrases):
        story = (
            "Once upon a time, children played happily in the park. "
            "They laughed together, shared their toys, and discovered magical adventures among the trees. "
            "As the sun set, they went home with smiles, ready for sweet dreams."
        )

    return story

def story_to_audio(story_text):
    if tts is None:
        return None
    try:
        audio_out = tts(story_text)
        samples = audio_out["audio"]
        rate = audio_out["sampling_rate"]
        buf = io.BytesIO()
        sf.write(buf, samples, rate, format="WAV")
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

# Streamlit UI
def main():
    st.set_page_config(page_title="Kids Story Generator", page_icon="📖", layout="centered")

    st.markdown(
        "<h1 style='text-align:center; color:#FF69B4;'>✨ Magical Story Generator ✨</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#228B22; font-size:20px;'>Upload a picture and hear your bedtime story!</p>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # ✅ FIX: use_container_width instead of use_column_width
        st.image(image, caption="Your Picture", use_container_width=True)

        if st.button("Generate Story"):
            caption = img2text(uploaded_file)
            st.success(f"📝 Caption: {caption}")

            story = generate_story(caption)
            st.markdown(
                f"<div style='background-color:#FFFACD; padding:15px; border-radius:10px;'>"
                f"<b>📖 Story:</b><br>{story}</div>",
                unsafe_allow_html=True
            )

            audio_bytes = story_to_audio(story)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
            else:
                st.warning("🔊 Audio unavailable due to memory limits. Please enjoy reading the story!")

if __name__ == "__main__":
    main()
