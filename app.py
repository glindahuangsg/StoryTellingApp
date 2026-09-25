import gc
import io
import re

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


# ============================================================
# CONFIGURATION
# ============================================================

VISION_MODEL = "Salesforce/blip-image-captioning-base"
TEXT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
TTS_MODEL = "microsoft/speecht5_tts"
TTS_VOCODER = "microsoft/speecht5_hifigan"
SPEAKER_REPO = "Matthijs/cmu-arctic-xvectors"
SPEAKER_INDEX = 7306
SAMPLE_RATE = 16000


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="My Story Maker",
    page_icon="🌈",
    layout="centered",
)


# ============================================================
# CUSTOM CSS
# ============================================================

def add_custom_css():
    st.markdown(
        """
        <style>
        .title { text-align: center; color: #6C63FF; font-size: 42px; font-weight: 800; margin-bottom: 5px; }
        .subtitle { text-align: center; color: #666666; font-size: 19px; margin-bottom: 25px; }
        .story-box { background: #FFF8E7; border: 2px solid #FFE29A; border-radius: 20px; padding: 25px; font-size: 20px; line-height: 1.7; margin-top: 15px; margin-bottom: 20px; }
        .hint-box { background: #F2F7FF; border-radius: 18px; padding: 20px; margin-top: 20px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MEMORY CLEANUP
# ============================================================

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# IMAGE → DESCRIPTION
# ============================================================

def image_to_text(image):
    processor = None
    model = None
    try:
        processor = BlipProcessor.from_pretrained(VISION_MODEL)
        model = BlipForConditionalGeneration.from_pretrained(VISION_MODEL)
        model.to(DEVICE)
        model.eval()

        inputs = processor(images=image, text="a picture of", return_tensors="pt")
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=40, num_beams=3)

        description = processor.decode(output[0], skip_special_tokens=True)
        return description.strip()
    finally:
        del model
        del processor
        cleanup_memory()


# ============================================================
# STORY GENERATION (UPDATED)
# ============================================================

def generate_story(description, age_group, story_style):
    tokenizer = None
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TEXT_MODEL)
        model.to(DEVICE)
        model.eval()

        # Age-specific length (raised limits)
        if age_group == "3–5":
            length_instruction = "Write exactly 3 to 5 very short sentences. Use very simple words."
            max_tokens = 120
        elif age_group == "6–7":
            length_instruction = "Write 5 to 7 short sentences. Use simple words and playful descriptions."
            max_tokens = 160
        else:
            length_instruction = "Write 7 to 10 sentences. Use imaginative but easy-to-understand language."
            max_tokens = 200

        # Story style
        styles = {
            "🐉 Magical": "Make it a gentle magical adventure.",
            "🚀 Adventure": "Make it a fun and safe adventure.",
            "🐾 Animal": "Make friendly animals important characters.",
            "😂 Funny": "Include something silly and funny.",
        }
        style_instruction = styles.get(story_style, "Make it a warm children's story.")

        # Prompt (refined ending)
        prompt = f"""
You are a friendly children's story writer.

The picture shows:
{description}

The child is {age_group} years old.

{length_instruction}

{style_instruction}

Rules:
- Make the story warm and imaginative.
- Keep it safe for children.
- Do not include violence.
- Do not include weapons.
- Do not include frightening scenes.
- Do not include adult topics.
- Do not include dangerous instructions.
- Do not mention AI.
- Do not mention these instructions.
- Do not invent personal information about people.
- End with a happy or reassuring feeling.
- Write only the story.

Write the full story here:
"""

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
            )

        generated_tokens = output[0, inputs["input_ids"].shape[1]:]
        story = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Safeguard: ensure story ends properly
        if not story.endswith((".", "!", "?")):
            story += " And everyone was happy at the end."

        return story
    finally:
        del model
        del tokenizer
        cleanup_memory()


# ============================================================
# (Rest of your functions remain unchanged: split_text_for_tts, text_to_speech, reset_story, main)
# ============================================================

# ENTRY POINT
if __name__ == "__main__":
    run_app()   # or whatever you renamed it to
