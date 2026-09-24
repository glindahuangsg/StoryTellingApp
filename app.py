import streamlit as st
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import requests
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.display import Audio
from PIL import Image
from io import BytesIO

st.title("Image to Audio Text Generation")

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKENS = os.getenv("api_token")

def img2text(image):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(image)[0]['generated_text']
    st.text("Generated Story from Image:")
    st.write(text)

    st.image(image, use_column_width=True)
    return text

st.text("You can choose any one of the following:")
image_url = st.text_input("1. Enter the URL of the image:")
st.write("or")
image_file = st.file_uploader("2. Upload an image", type=["jpg", "jpeg", "png"])

if image_url and st.button("Generate Text from Image (URL)"):
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
        image_caption = img2text(image)
    except Exception as e:
        st.error("Error: Invalid URL or unsupported image format.")

if image_file and st.button("Generate Text from Image (Upload)"):
    image = Image.open(image_file)
    image_caption = img2text(image)

if "image_caption" in locals():
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKENS}"}
    # text_generative =pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")



    def query(prompt, max_new_tokens=200):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens
            }
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    data = query(image_caption, max_new_tokens=250)
    if data and isinstance(data, list) and data[0] and isinstance(data[0], dict):
        generated_text = data[0].get("generated_text", "")
    else:
        generated_text = ""

    st.text("Generated Text:")
    st.write(generated_text)

    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKENS}"}

    def generate_and_play_audio(text, sampling_rate=22050):
        payload = {"inputs": text}
        response = requests.post(API_URL, headers=headers, json=payload)
        audio = response.content

        # Create an Audio object to play the binary audio data
        return Audio(audio, rate=sampling_rate)

    if "generated_text" in locals():
        text_to_speak = generated_text
        audio_object = generate_and_play_audio(text_to_speak)
        st.write("Audio Response:")
        st.audio(audio_object.data, format="audio/wav")

st.text("")
