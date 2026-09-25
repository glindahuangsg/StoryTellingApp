import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from PIL import Image

# ✅ Force CPU mode
device = torch.device("cpu")

# -----------------------------------------------------------
# Function: load_models
# Purpose: Load lightweight text + caption models
# -----------------------------------------------------------
@st.cache_resource
def load_models():
    # Tiny caption model (uses Hugging Face pipeline for simplicity)
    captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

    # Smaller text model (DistilGPT2 instead of Flan-T5)
    text_model_id = "distilgpt2"
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    text_model = AutoModelForCausalLM.from_pretrained(text_model_id).to(device)

    return captioner, text_tokenizer, text_model

captioner, text_tokenizer, text_model = load_models()

# -----------------------------------------------------------
# Function: img2text
# Purpose: Generate a caption from an uploaded image
# -----------------------------------------------------------
def img2text(image_file):
    return captioner(image_file)[0]["generated_text"]

# -----------------------------------------------------------
# Function: generate_story
# Purpose: Generate a short bedtime story in parental voice
# -----------------------------------------------------------
def generate_story(caption, text_tokenizer=text_tokenizer, text_model=text_model, device=device):
    prompt = (
        f"Once upon a time, my dear, let me tell you a gentle bedtime story. "
        f"This story is about {caption}. "
        f"It should sound like a parent speaking softly to their child, "
        f"with a clear beginning, middle, and a happy ending."
    )

    inputs = text_tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = text_model.generate(
        inputs,
        max_length=120,   # shorter output for speed
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )
    story = text_tokenizer.decode(output[0], skip_special_tokens=True)
    return story.strip()

# -----------------------------------------------------------
# Function: main
# Purpose: Build kid-friendly UI
# -----------------------------------------------------------
def main():
    st.set_page_config(page_title="Kids Story Generator", page_icon="📖", layout="centered")

    # 🎨 Kid-friendly header
    st.markdown(
        "<h1 style='text-align:center; color:#FF69B4;'>🌈 Magical Storytime 🌈</h1>",
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
            # Show friendly waiting message
            with st.spinner("✨ Hold on tight! Your magical bedtime story is being written... ✨"):
                caption = img2text(uploaded_file)
                story = generate_story(caption)

            # Display caption + story
            st.success(f"📝 Magic Caption: {caption}")
            st.markdown(
                f"<div style='background-color:#FFFACD; padding:20px; border-radius:15px; font-size:18px;'>"
                f"<b>📖 Your Story:</b><br>{story}</div>",
                unsafe_allow_html=True
            )
            st.info("💡 Tip: Imagine yourself in the story while you read it!")

# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    main()
