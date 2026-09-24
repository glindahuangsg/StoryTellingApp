import gc
import io
import sys

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
# CONFIGURATION
# ============================================================

VISION_MODEL = "Salesforce/blip-image-captioning-base"

TEXT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

TTS_MODEL = "microsoft/speecht5_tts"

TTS_VOCODER = "microsoft/speecht5_hifigan"

SPEAKER_DATASET = "Matthijs/cmu-arctic-xvectors"

# Hugging Face's SpeechT5 example uses this speaker embedding.
SPEAKER_INDEX = 7306

# SpeechT5 generates audio at 16 kHz.
SAMPLE_RATE = 16000


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="My Story Maker",
    page_icon="🌈",
    layout="centered",
)


# ============================================================
# CSS
# ============================================================

def add_custom_css():

    st.markdown(
        """
        <style>

        .title {
            text-align: center;
            color: #6C63FF;
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #666666;
            font-size: 19px;
            margin-bottom: 25px;
        }

        .story-box {
            background: #FFF8E7;
            border: 2px solid #FFE29A;
            border-radius: 20px;
            padding: 25px;
            font-size: 20px;
            line-height: 1.7;
            margin-top: 15px;
        }

        .hint-box {
            background: #F2F7FF;
            border-radius: 18px;
            padding: 20px;
            margin-top: 20px;
        }

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
# SPEAKER EMBEDDING
# ============================================================

@st.cache_resource
def load_speaker_embedding():

    dataset = load_dataset(
        SPEAKER_DATASET,
        split="validation",
    )

    embedding = torch.tensor(
        dataset[SPEAKER_INDEX]["xvector"],
        dtype=torch.float32,
    )

    # SpeechT5 expects:
    #
    # [batch_size, 512]
    #
    # The x-vector is 512-dimensional.

    embedding = embedding.unsqueeze(0)

    return embedding


# ============================================================
# IMAGE → DESCRIPTION
# ============================================================

def image_to_text(image):

    processor = None
    model = None

    try:

        st.info("👀 Looking at the picture...")

        processor = BlipProcessor.from_pretrained(
            VISION_MODEL
        )

        model = BlipForConditionalGeneration.from_pretrained(
            VISION_MODEL
        )

        model = model.to(DEVICE)

        model.eval()

        prompt = "a picture of"

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
            )

        description = processor.decode(
            output[0],
            skip_special_tokens=True,
        )

        return description.strip()

    finally:

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

        st.info("🪄 Creating your story...")

        tokenizer = AutoTokenizer.from_pretrained(
            TEXT_MODEL
        )

        model = AutoModelForCausalLM.from_pretrained(
            TEXT_MODEL
        )

        model = model.to(DEVICE)

        model.eval()

        # ----------------------------------------------------
        # Age-specific instructions
        # ----------------------------------------------------

        if age_group == "3–5":

            length_instruction = (
                "Write 3 to 5 very short sentences. "
                "Use simple words that a preschool child "
                "can understand."
            )

        elif age_group == "6–7":

            length_instruction = (
                "Write 5 to 7 short sentences. "
                "Use simple vocabulary and playful details."
            )

        else:

            length_instruction = (
                "Write 7 to 10 sentences. "
                "Use imaginative but easy-to-understand language."
            )

        # ----------------------------------------------------
        # Style
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
You are writing a short story for a child.

Picture description:
{description}

Child age:
{age_group}

Story style:
{style_instruction}

{length_instruction}

Rules:

- Be kind and positive.
- Make the story imaginative.
- Keep it safe for children.
- Do not include violence.
- Do not include weapons.
- Do not include frightening scenes.
- Do not include adult topics.
- Do not include dangerous instructions.
- Do not mention artificial intelligence.
- Do not talk about these instructions.
- Do not invent personal information about real people.
- End with a happy or reassuring feeling.
- Write only the story.

Story:
"""

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        # SmolLM2 supports the chat template.
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

            output = model.generate(
                **inputs,
                max_new_tokens=140,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        generated_tokens = output[
            0,
            inputs["input_ids"].shape[1]:
        ]

        story = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        story = story.strip()

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

    try:

        # ----------------------------------------------------
        # Load SpeechT5 processor
        # ----------------------------------------------------

        processor = SpeechT5Processor.from_pretrained(
            TTS_MODEL
        )

        # ----------------------------------------------------
        # Load SpeechT5 model
        # ----------------------------------------------------

        model = SpeechT5ForTextToSpeech.from_pretrained(
            TTS_MODEL
        )

        model = model.to(DEVICE)

        model.eval()

        # ----------------------------------------------------
        # Load HiFi-GAN
        # ----------------------------------------------------

        vocoder = SpeechT5HifiGan.from_pretrained(
            TTS_VOCODER
        )

        vocoder = vocoder.to(DEVICE)

        vocoder.eval()

        # ----------------------------------------------------
        # Speaker embedding
        # ----------------------------------------------------

        speaker_embedding = (
            load_speaker_embedding()
            .to(DEVICE)
        )

        # Make absolutely sure the shape is [1, 512].
        if speaker_embedding.ndim == 1:

            speaker_embedding = (
                speaker_embedding
                .unsqueeze(0)
            )

        if speaker_embedding.shape != (1, 512):

            raise ValueError(
                "Invalid speaker embedding shape: "
                f"{speaker_embedding.shape}. "
                "Expected [1, 512]."
            )

        # ----------------------------------------------------
        # Process text
        # ----------------------------------------------------

        inputs = processor(
            text=text,
            return_tensors="pt",
        )

        input_ids = inputs[
            "input_ids"
        ].to(DEVICE)

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
        # Convert tensor → NumPy
        # ----------------------------------------------------

        speech = speech.detach().cpu().numpy()

        # ----------------------------------------------------
        # Create WAV in memory
        # ----------------------------------------------------

        audio_buffer = io.BytesIO()

        sf.write(
            audio_buffer,
            speech,
            SAMPLE_RATE,
            format="WAV",
        )

        audio_buffer.seek(0)

        audio_bytes = audio_buffer.read()

        if not audio_bytes:

            raise RuntimeError(
                "SpeechT5 generated an empty audio file."
            )

        return audio_bytes

    except Exception as error:

        # ----------------------------------------------------
        # Show real error during development
        # ----------------------------------------------------

        st.error(
            f"TTS error: {type(error).__name__}: {error}"
        )

        return None

    finally:

        del model
        del processor
        del vocoder

        cleanup_memory()


# ============================================================
# RESET
# ============================================================

def reset_story():

    for key in [
        "description",
        "story",
        "audio",
    ]:

        st.session_state.pop(
            key,
            None,
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
        '<div class="title">'
        '🌈 My Story Maker'
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
    # Settings
    # --------------------------------------------------------

    st.subheader("✨ Choose your story")

    col1, col2 = st.columns(2)

    with col1:

        age_group = st.selectbox(
            "Age",
            [
                "3–5",
                "6–7",
                "8–10",
            ],
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
    # Upload image
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
            <div class="hint-box">

            <h3>💡 Try a picture of...</h3>

            🧸 A favorite toy<br>
            🐶 A pet<br>
            🌳 A park<br>
            🏰 A castle<br>
            🚲 A bicycle<br>
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
            "I couldn't open that picture. "
            "Please try another image."
        )

        return

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

        # Clear old results.

        st.session_state.pop(
            "description",
            None,
        )

        st.session_state.pop(
            "story",
            None,
        )

        st.session_state.pop(
            "audio",
            None,
        )

        # ----------------------------------------------------
        # Image → Text
        # ----------------------------------------------------

        with st.spinner(
            "👀 Looking at your picture..."
        ):

            description = image_to_text(
                image
            )

        st.session_state[
            "description"
        ] = description

        # ----------------------------------------------------
        # Text → Story
        # ----------------------------------------------------

        with st.spinner(
            "🪄 Creating your story..."
        ):

            story = generate_story(
                description,
                age_group,
                story_style,
            )

        st.session_state[
            "story"
        ] = story

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    if "description" in st.session_state:

        with st.expander(
            "👀 What I saw in the picture"
        ):

            st.write(
                st.session_state[
                    "description"
                ]
            )

    # --------------------------------------------------------
    # Story
    # --------------------------------------------------------

    if "story" in st.session_state:

        st.divider()

        st.subheader("📖 Your Story")

        # Use st.markdown rather than injecting the generated
        # story into HTML. This prevents generated text from
        # being interpreted as HTML.

        st.markdown(
            '<div class="story-box">',
            unsafe_allow_html=True,
        )

        st.write(
            st.session_state["story"]
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # TTS
        # ----------------------------------------------------

        st.divider()

        st.subheader("🔊 Listen to your story")

        if st.button(
            "🎵 Read My Story",
            use_container_width=True,
        ):

            with st.spinner(
                "🎵 Making the audio..."
            ):

                audio = text_to_speech(
                    st.session_state[
                        "story"
                    ]
                )

            if audio is not None:

                st.session_state[
                    "audio"
                ] = audio

                st.success(
                    "🎉 Your story is ready!"
                )

        # ----------------------------------------------------
        # Audio player
        # ----------------------------------------------------

        if st.session_state.get(
            "audio"
        ):

            st.audio(
                st.session_state["audio"],
                format="audio/wav",
            )

            st.download_button(
                label="⬇️ Download audio",
                data=st.session_state["audio"],
                file_name="my_story.wav",
                mime="audio/wav",
                use_container_width=True,
            )

        # ----------------------------------------------------
        # New story
        # ----------------------------------------------------

        st.write("")

        if st.button(
            "🌟 Make Another Story",
            use_container_width=True,
        ):

            reset_story()

            st.rerun()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
