import io
import time

import numpy as np
import streamlit as st
import torch
import soundfile as sf

from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="My Story Maker",
    page_icon="🌈",
    layout="centered",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

VISION_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
TEXT_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TTS_MODEL = "microsoft/speecht5_tts"

# SpeechT5 speaker embeddings
SPEAKER_DATASET = "Matthijs/cmu-arctic-xvectors"
SPEAKER_INDEX = 7306


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32


# ============================================================
# LOAD VISION MODEL
# ============================================================

@st.cache_resource
def load_vision_model():

    processor = AutoProcessor.from_pretrained(
        VISION_MODEL
    )

model = AutoModelForMultimodalLM.from_pretrained(
    VISION_MODEL,
    torch_dtype=DTYPE,
)

    model.to(DEVICE)

    return processor, model


# ============================================================
# LOAD TEXT MODEL
# ============================================================

@st.cache_resource
def load_text_model():

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL
    )

    model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL,
        torch_dtype=DTYPE,
    )

    model.to(DEVICE)

    return tokenizer, model


# ============================================================
# LOAD TEXT-TO-SPEECH MODEL
# ============================================================

@st.cache_resource
def load_tts_model():

    synthesizer = pipeline(
        "text-to-speech",
        model=TTS_MODEL,
    )

    return synthesizer


# ============================================================
# IMAGE → DESCRIPTION
# ============================================================

def image_to_text(image, processor, model):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": (
                        "Describe this picture for a young child. "
                        "Use simple, friendly words. "
                        "Mention the main people, animals, "
                        "objects, colors, and actions you can see. "
                        "Do not guess private information about people."
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    result = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return result.strip()


# ============================================================
# GENERATE CHILD-FRIENDLY STORY
# ============================================================

def generate_story(
    description,
    tokenizer,
    model,
    age_group,
    story_style,
):

    if age_group == "3–5":

        length_instruction = """
Write 3–5 very short sentences.
Use very simple vocabulary.
Use a warm, playful tone.
"""

    elif age_group == "6–7":

        length_instruction = """
Write 6–8 short sentences.
Use simple vocabulary and a little imagination.
"""

    else:

        length_instruction = """
Write 8–12 sentences.
Use vivid but age-appropriate language.
Include a small adventure and a happy or reassuring ending.
"""

    style_instructions = {
        "🐉 Magical": "Add gentle magical elements such as friendly dragons, fairies, or enchanted places.",
        "🚀 Adventure": "Make the story feel like a fun, safe adventure.",
        "🐾 Animal": "Include friendly animals as important characters.",
        "😂 Funny": "Include something silly or surprising that is appropriate for children.",
    }

    style_instruction = style_instructions[story_style]

    prompt = f"""
You are a children's story writer.

Create a safe, kind and imaginative story for a child.

The picture shows:

{description}

Age group:
{age_group}

Story style:
{story_style}

{length_instruction}

{style_instruction}

Rules:
- The story must be suitable for children.
- No violence.
- No frightening scenes.
- No weapons.
- No dangerous instructions.
- No adult themes.
- No bullying.
- Do not mention that you are an AI.
- Do not talk about the prompt.
- Do not invent personal information about people in the picture.
- Give the story a positive or reassuring ending.

Write only the story.
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    # Try to remove the original prompt from the result.
    if generated_text.startswith(input_text):
        generated_text = generated_text[
            len(input_text):
        ]

    return generated_text.strip()


# ============================================================
# TEXT → SPEECH
# ============================================================

def text_to_speech(text, synthesizer):

    speech = synthesizer(text)

    audio = speech["audio"]

    # Transformers may return [1, samples].
    if len(audio.shape) == 2:
        audio = audio[0]

    sample_rate = speech["sampling_rate"]

    buffer = io.BytesIO()

    sf.write(
        buffer,
        audio,
        sample_rate,
        format="WAV",
    )

    buffer.seek(0)

    return buffer.read()


# ============================================================
# MAIN STREAMLIT APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="text-align:center">

        <h1>🌈 My Story Maker 🌈</h1>

        <p style="font-size:20px;">
        Turn a picture into a magical story!
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Story settings
    # --------------------------------------------------------

    st.subheader("✨ Choose your story")

    col1, col2 = st.columns(2)

    with col1:

        age_group = st.selectbox(
            "How long should the story be?",
            [
                "3–5",
                "6–7",
                "8–10",
            ],
        )

    with col2:

        story_style = st.selectbox(
            "What kind of story?",
            [
                "🐉 Magical",
                "🚀 Adventure",
                "🐾 Animal",
                "😂 Funny",
            ],
        )

    # --------------------------------------------------------
    # Image upload
    # --------------------------------------------------------

    st.subheader("📸 Pick a picture")

    uploaded_file = st.file_uploader(
        "Choose a picture",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        label_visibility="collapsed",
    )

    if uploaded_file is None:

        st.info(
            "👆 Choose a picture to begin your adventure!"
        )

        st.markdown(
            """
            ### 💡 Ideas

            Try a picture of:

            - 🧸 Your favorite toy
            - 🐶 A pet
            - 🌳 A park
            - 🏰 A castle
            - 🚲 A bike
            - 🎨 A drawing
            """
        )

        return

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Your picture",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Generate story
    # --------------------------------------------------------

    if st.button(
        "✨ Make My Story! ✨",
        type="primary",
        use_container_width=True,
    ):

        # ----------------------------------------------------
        # Load vision model
        # ----------------------------------------------------

        with st.spinner(
            "👀 Looking at your picture..."
        ):

            start = time.perf_counter()

            processor, vision_model = (
                load_vision_model()
            )

            description = image_to_text(
                image,
                processor,
                vision_model,
            )

            vision_time = (
                time.perf_counter() - start
            )

        # ----------------------------------------------------
        # Show description
        # ----------------------------------------------------

        st.session_state["description"] = description

        # ----------------------------------------------------
        # Generate story
        # ----------------------------------------------------

        with st.spinner(
            "🪄 Creating your story..."
        ):

            start = time.perf_counter()

            tokenizer, text_model = (
                load_text_model()
            )

            story = generate_story(
                description,
                tokenizer,
                text_model,
                age_group,
                story_style,
            )

            story_time = (
                time.perf_counter() - start
            )

        st.session_state["story"] = story

        st.session_state["timings"] = {
            "vision": vision_time,
            "story": story_time,
        }

    # --------------------------------------------------------
    # Display description
    # --------------------------------------------------------

    if "description" in st.session_state:

        with st.expander(
            "👀 What I saw in the picture"
        ):

            st.write(
                st.session_state["description"]
            )

    # --------------------------------------------------------
    # Display story
    # --------------------------------------------------------

    if "story" in st.session_state:

        st.divider()

        st.subheader("📖 Your Story")

        st.markdown(
            f"""
            <div style="
                background-color:#FFF8E7;
                padding:25px;
                border-radius:20px;
                font-size:20px;
                line-height:1.7;
            ">
            {st.session_state["story"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # Read story aloud
        # ----------------------------------------------------

        st.write("")

        if st.button(
            "🔊 Read My Story",
            use_container_width=True,
        ):

            with st.spinner(
                "🎵 Getting the story ready..."
            ):

                tts_model = load_tts_model()

                audio = text_to_speech(
                    st.session_state["story"],
                    tts_model,
                )

            st.audio(
                audio,
                format="audio/wav",
            )

        # ----------------------------------------------------
        # Start over
        # ----------------------------------------------------

        st.write("")

        if st.button(
            "🌟 Make Another Story",
            use_container_width=True,
        ):

            for key in [
                "description",
                "story",
                "timings",
            ]:
                st.session_state.pop(
                    key,
                    None,
                )

            st.rerun()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
