import streamlit as st
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from PIL import Image
import io

# ----------------------------
# Device selector
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load models once (cache)
# ----------------------------
@st.cache_resource
def load_models():
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)

    text_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    if text_tokenizer.pad_token_id is None:
        text_tokenizer.pad_token_id = text_tokenizer.eos_token_id
    text_model = AutoModelForCausalLM.from_pretrained(
        text_model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)

    # Hugging Face pipeline for TTS (simpler for Streamlit deployment)
    tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")

    return blip_processor, blip_model, text_tokenizer, text_model, tts

blip_processor, blip_model, text_tokenizer, text_model, tts = load_models()

# ----------------------------
# Helper functions
# ----------------------------
def img2text(image_file):
    raw_image = Image.open(image_file).convert("RGB")
    inputs = blip_processor(raw_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=30)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def generate_story(caption):
    prompt = (
        f"Expand this caption into a cheerful bedtime story for children aged 3–10. "
        f"Keep it wholesome, imaginative, and 50–100 words long. Caption: {caption}"
    )
    inputs = text_tokenizer(prompt, return_tensors="pt").to(device)
    output = text_model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=text_tokenizer.eos_token_id,
    )
    return text_tokenizer.decode(output[0], skip_special_tokens=True).strip()

def story_to_audio(story_text):
    audio_out = tts(story_text)
    return audio_out["audio"]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📖 Bedtime Story Generator")
st.write("Upload an image and let the app create a magical story with audio narration!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    caption = img2text(uploaded_file)
    st.subheader("Generated Caption")
    st.write(caption)

    story = generate_story(caption)
    st.subheader("Generated Story")
    st.write(story)

    audio_bytes = story_to_audio(story)
    st.subheader("Audio Narration")
    st.audio(io.BytesIO(audio_bytes), format="audio/wav")
