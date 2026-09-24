import gc
import io
import time

import streamlit as st
import torch
import soundfile as sf

from PIL import Image

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)

from datasets import load_dataset


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="🌈 My Story Maker",
    page_icon="🌈",
    layout="centered",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Image → Text
VISION_MODEL = "Salesforce/blip-image-captioning-base"

# Text → Story
TEXT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Text → Speech
TTS_MODEL = "microsoft/speecht5_tts"
TTS_VOCODER = "microsoft/speecht5_hifigan"

# Public speaker-embedding dataset used by SpeechT5.
SPEAKER_DATASET = "Matthijs/cmu-arctic-xvectors"

# One of the voices in the dataset.
SPEAKER_INDEX = 7306


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HELPER: CLEAN UP MEMORY
# ============================================================

def cleanup_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# IMAGE → TEXT
# ============================================================

def image_to_text(image):

    processor = None
    model = None

    try:

        processor = BlipProcessor.from_pretrained(
            VISION_MODEL
        )

        model = BlipForConditionalGeneration.from_pretrained(
            VISION_MODEL
        )

        model.to(DEVICE)
        model.eval()

        # BLIP is an image-captioning model rather than a
        # general vision-language model.
        prompt = (
            "a friendly picture of"
        )

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=3,
                early_stopping=True,
            )

        description = processor.decode(
            output[0],
            skip_special_tokens=True,
        )

        return description.strip()

    finally:

        # Don't keep the vision model in RAM.
        del model
        del processor

        cleanup_memory()


# ============================================================
# STORY GENERATION
# ============================================================

def generate_story(
    description,
    age_group,
    story_style,
):

    tokenizer = None
    model = None

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            TEXT_MODEL
        )

        model = AutoModelForCausalLM.from_pretrained(
            TEXT_MODEL
        )

        model.to(DEVICE)
        model.eval()

        # ----------------------------------------------------
        # Story length
        # ----------------------------------------------------

        if age_group == "3–5":

            length = (
                "Write 3 to 5 very short sentences. "
                "Use very simple words."
            )

        elif age_group == "6–7":

            length = (
                "Write 5 to 7 short sentences. "
                "Use simple words and playful descriptions."
            )

        else:

            length = (
                "Write 7 to 10 sentences. "
                "Use imaginative but easy-to-understand language."
            )

        # ----------------------------------------------------
        # Story style
        # ----------------------------------------------------

        styles = {

            "🐉 Magical":
                "Make it a gentle magical adventure.",

            "🚀 Adventure":
                "Make it a fun and safe adventure.",

            "🐾 Animal":
                "Make friendly animals important characters.",

            "😂 Funny":
                "Include something silly and funny.",
        }

        style_instruction = styles.get(
            story_style,
            "Make it a warm children's story.",
        )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = f"""
You are a children's story writer.

Create a kind, imaginative story for a child.

The picture description is:

{description}

The child is in the {age_group} age group.

{length}

{style_instruction}

Important rules:

- The story must be safe for children.
- Do not include violence.
- Do not include weapons.
- Do not include scary or frightening scenes.
- Do not include adult topics.
- Do not include bullying.
- Do not include dangerous instructions.
- Do not mention being an AI.
- Do not talk about the instructions.
- Do not invent names or personal information about people.
- End the story in a happy, warm, or reassuring way.
- Write only the story.

Story:
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

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=140,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        # Only decode the newly generated tokens.
        generated_tokens = output[
            0,
            inputs["input_ids"].shape[1]:
        ]

        story = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        story = story.strip()

        # ----------------------------------------------------
        # Safety-oriented cleanup
        # ----------------------------------------------------

        if story.lower().startswith("story:"):

            story = story[6:].strip()

        return story

    finally:

        del model
        del tokenizer

        cleanup_memory()


# ============================================================
# TEXT → SPEECH
# ============================================================

def text_to_speech(text):

    processor = None
    model = None
    vocoder = None
    embeddings_dataset = None

    try:

        # ----------------------------------------------------
        # Load SpeechT5
        # ----------------------------------------------------

        processor = SpeechT5Processor.from_pretrained(
            TTS_MODEL
        )

        model = SpeechT5ForTextToSpeech.from_pretrained(
            TTS_MODEL
        )

        vocoder = SpeechT5HifiGan.from_pretrained(
            TTS_VOCODER
        )

        model.to(DEVICE)
        vocoder.to(DEVICE)

        model.eval()
        vocoder.eval()

        # ----------------------------------------------------
        # Speaker embedding
        # ----------------------------------------------------

        embeddings_dataset = load_dataset(
            SPEAKER_DATASET,
            split="validation",
        )

        speaker_embedding = torch.tensor(
            embeddings_dataset[SPEAKER_INDEX]["xvector"]
        ).unsqueeze(0)

        speaker_embedding = speaker_embedding.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Prepare text
        # ----------------------------------------------------

        inputs = processor(
            text=text,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(
            DEVICE
        )

        # ----------------------------------------------------
        # Generate speech
        # ----------------------------------------------------

        with torch.no_grad():

            speech = model.generate_speech(
                input_ids,
                speaker_embedding,
                vocoder=vocoder,
            )

        # ----------------------------------------------------
        # Convert to WAV
        # ----------------------------------------------------

        audio_buffer = io.BytesIO()

        sf.write(
            audio_buffer,
            speech.cpu().numpy(),
            16000,
            format="WAV",
        )

        audio_buffer.seek(0)

        return audio_buffer.read()

    finally:

        del model
        del processor
        del vocoder
        del embeddings_dataset

        cleanup_memory()


# ============================================================
# RESET APP
# ============================================================

def reset_story():

    keys_to_remove = [
        "description",
        "story",
        "audio",
    ]

    for key in keys_to_remove:

        if key in st.session_state:

            del st.session_state[key]


# ============================================================
# CHILD-FRIENDLY STYLING
# ============================================================

def add_custom_css():

    st.markdown(
        """
        <style>

        .main-title {
            text-align: center;
            font-size: 42px;
            font-weight: 800;
            color: #6C63FF;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #555555;
            margin-bottom: 25px;
        }

        .story-box {
            background-color: #FFF8E7;
            border-radius: 20px;
            padding: 25px;
            margin-top: 15px;
            margin-bottom: 20px;
            border: 2px solid #FFE4A3;
            font-size: 20px;
            line-height: 1.7;
        }

        .idea-box {
            background-color: #F0F7FF;
            border-radius: 18px;
            padding: 20px;
            margin-top: 20px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    add_custom_css()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.markdown(
        '<div class="main-title">'
        '🌈 My Story Maker 🌈'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Turn a picture into a magical story!'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Story settings
    # --------------------------------------------------------

    st.subheader("✨ Choose your story")

    col1, col2 = st.columns(2)

    with col1:

        age_group = st.selectbox(
            "Story size",
            [
                "3–5",
                "6–7",
                "8–10",
            ],
            help=(
                "This controls how long and complex "
                "the story will be."
            ),
        )

    with col2:

        story_style = st.selectbox(
            "Story type",
            [
                "🐉 Magical",
                "🚀 Adventure",
                "🐾 Animal",
                "😂 Funny",
            ],
        )

    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    st.subheader("📸 Choose a picture")

    uploaded_file = st.file_uploader(
        "Upload a picture",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        label_visibility="collapsed",
    )

    if uploaded_file is None:

        st.markdown(
            """
            <div class="idea-box">

            <h3>💡 Try a picture of...</h3>

            🧸 Your favorite toy<br>
            🐶 A pet<br>
            🌳 A park<br>
            🏰 A castle<br>
            🚲 A bike<br>
            🎨 A drawing

            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Hmm... I couldn't open that picture. "
            "Please try another image."
        )

        return

    st.image(
        image,
        caption="Your picture",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Make story button
    # --------------------------------------------------------

    if st.button(
        "✨ Make My Story! ✨",
        type="primary",
        use_container_width=True,
    ):

        # ----------------------------------------------------
        # IMAGE → TEXT
        # ----------------------------------------------------

        with st.spinner(
            "👀 Looking at your picture..."
        ):

            start = time.perf_counter()

            description = image_to_text(
                image
            )

            elapsed = (
                time.perf_counter() - start
            )

        st.session_state[
            "description"
        ] = description

        # ----------------------------------------------------
        # TEXT → STORY
        # ----------------------------------------------------

        with st.spinner(
            "🪄 Creating your story..."
        ):

            start = time.perf_counter()

            story = generate_story(
                description,
                age_group,
                story_style,
            )

            elapsed_story = (
                time.perf_counter() - start
            )

        st.session_state[
            "story"
        ] = story

        st.session_state[
            "audio"
        ] = None

    # --------------------------------------------------------
    # IMAGE DESCRIPTION
    # --------------------------------------------------------

    if "description" in st.session_state:

        with st.expander(
            "👀 What I saw in the picture"
        ):

            st.write(
                st.session_state["description"]
            )

    # --------------------------------------------------------
    # STORY
    # --------------------------------------------------------

    if "story" in st.session_state:

        st.divider()

        st.subheader("📖 Your Story")

        st.markdown(
            f"""
            <div class="story-box">
            {st.session_state["story"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # READ STORY
        # ----------------------------------------------------

        if st.button(
            "🔊 Read My Story",
            use_container_width=True,
        ):

            with st.spinner(
                "🎵 Getting the story ready..."
            ):

                try:

                    audio = text_to_speech(
                        st.session_state["story"]
                    )

                    st.session_state[
                        "audio"
                    ] = audio

                except Exception as error:

                    st.warning(
                        "I couldn't make the audio "
                        "right now. You can still read "
                        "the story! 😊"
                    )

                    st.session_state[
                        "audio"
                    ] = None

        # ----------------------------------------------------
        # AUDIO PLAYER
        # ----------------------------------------------------

        if st.session_state.get(
            "audio"
        ):

            st.audio(
                st.session_state["audio"],
                format="audio/wav",
            )

        # ----------------------------------------------------
        # NEW STORY
        # ----------------------------------------------------

        st.write("")

        if st.button(
            "🌟 Make Another Story",
            use_container_width=True,
        ):

            reset_story()

            st.rerun()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
