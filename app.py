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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
# CUSTOM CSS
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
            margin-bottom: 20px;
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

    """
    Download one SpeechT5 speaker embedding.

    We intentionally DO NOT use:

        datasets.load_dataset()

    because the original CMU Arctic repository contains
    an old Python dataset loading script that newer versions
    of the datasets package no longer support.
    """

    zip_path = hf_hub_download(
        repo_id=SPEAKER_REPO,
        filename="spkrec-xvect.zip",
        repo_type="dataset",
    )

    import zipfile

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:

        npy_files = [
            name
            for name in archive.namelist()
            if name.endswith(".npy")
        ]

        npy_files.sort()

        if SPEAKER_INDEX >= len(npy_files):

            raise RuntimeError(
                f"Speaker index {SPEAKER_INDEX} is unavailable. "
                f"Found {len(npy_files)} speaker files."
            )

        selected_file = npy_files[SPEAKER_INDEX]

        with archive.open(
            selected_file
        ) as file:

            embedding = np.load(file)

    embedding = torch.tensor(
        embedding,
        dtype=torch.float32,
    )

    if embedding.ndim == 1:

        embedding = embedding.unsqueeze(0)

    if embedding.shape != (1, 512):

        raise RuntimeError(
            "Invalid speaker embedding shape: "
            f"{embedding.shape}. "
            "Expected (1, 512)."
        )

    return embedding


# ============================================================
# IMAGE → DESCRIPTION
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

        inputs = processor(
            images=image,
            text="a picture of",
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

        tokenizer = AutoTokenizer.from_pretrained(
            TEXT_MODEL
        )

        model = AutoModelForCausalLM.from_pretrained(
            TEXT_MODEL
        )

        model.to(DEVICE)
        model.eval()

        # ----------------------------------------------------
        # Age-specific length
        # ----------------------------------------------------

        if age_group == "3–5":

            length_instruction = (
                "Write exactly 3 to 5 very short sentences. "
                "Use very simple words."
            )

            max_tokens = 80

        elif age_group == "6–7":

            length_instruction = (
                "Write 5 to 7 short sentences. "
                "Use simple words and playful descriptions."
            )

            max_tokens = 110

        else:

            length_instruction = (
                "Write 7 to 10 sentences. "
                "Use imaginative but easy-to-understand language."
            )

            max_tokens = 140

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

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
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
# SPLIT TEXT FOR SPEECHT5
# ============================================================

def split_text_for_tts(
    text,
    max_chars=150,
):

    """
    SpeechT5 has a maximum sequence length.

    Instead of sending an entire story to SpeechT5,
    divide it into short chunks and synthesize each
    chunk separately.

    max_chars is deliberately conservative.
    """

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:

        return []

    # Split into sentences.

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    chunks = []

    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:

            continue

        candidate = (
            f"{current} {sentence}".strip()
        )

        if len(candidate) <= max_chars:

            current = candidate

        else:

            if current:

                chunks.append(current)

            # If a single sentence is too long,
            # split it by words.

            if len(sentence) > max_chars:

                words = sentence.split()

                current = ""

                for word in words:

                    candidate = (
                        f"{current} {word}".strip()
                    )

                    if len(candidate) <= max_chars:

                        current = candidate

                    else:

                        if current:

                            chunks.append(
                                current
                            )

                        current = word

            else:

                current = sentence

    if current:

        chunks.append(current)

    return chunks


# ============================================================
# TEXT → SPEECH
# ============================================================

def text_to_speech(text):

    processor = None
    model = None
    vocoder = None

    try:

        text = text.strip()

        if not text:

            raise ValueError(
                "There is no story to read."
            )

        # ----------------------------------------------------
        # Split the story BEFORE tokenization.
        # ----------------------------------------------------

        chunks = split_text_for_tts(
            text,
            max_chars=150,
        )

        if not chunks:

            raise ValueError(
                "The story could not be split into "
                "readable pieces."
            )

        st.info(
            f"🔊 Reading the story in "
            f"{len(chunks)} short parts..."
        )

        # ----------------------------------------------------
        # Speaker embedding
        # ----------------------------------------------------

        speaker_embedding = (
            load_speaker_embedding()
        )

        speaker_embedding = (
            speaker_embedding.to(DEVICE)
        )

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        processor = SpeechT5Processor.from_pretrained(
            TTS_MODEL
        )

        # ----------------------------------------------------
        # TTS model
        # ----------------------------------------------------

        model = SpeechT5ForTextToSpeech.from_pretrained(
            TTS_MODEL
        )

        model.to(DEVICE)
        model.eval()

        # ----------------------------------------------------
        # Vocoder
        # ----------------------------------------------------

        vocoder = SpeechT5HifiGan.from_pretrained(
            TTS_VOCODER
        )

        vocoder.to(DEVICE)
        vocoder.eval()

        # ----------------------------------------------------
        # Generate audio chunk by chunk
        # ----------------------------------------------------

        audio_chunks = []

        progress = st.progress(
            0,
            text="Preparing the storyteller..."
        )

        for index, chunk in enumerate(chunks):

            progress.progress(
                (index + 1) / len(chunks),
                text=(
                    f"🔊 Reading part "
                    f"{index + 1} of "
                    f"{len(chunks)}..."
                ),
            )

            # ----------------------------------------------
            # Tokenize ONE chunk
            # ----------------------------------------------

            inputs = processor(
                text=chunk,
                return_tensors="pt",
            )

            input_ids = inputs[
                "input_ids"
            ].to(DEVICE)

            token_count = input_ids.shape[1]

            # ----------------------------------------------
            # Safety check.
            #
            # SpeechT5 has a 600-token limit. We use a
            # conservative 450-token limit here.
            # ----------------------------------------------

            if token_count >= 450:

                raise ValueError(
                    "Speech chunk is still too long: "
                    f"{token_count} tokens. "
                    "Try making the story shorter."
                )

            # ----------------------------------------------
            # Generate speech
            # ----------------------------------------------

            with torch.no_grad():

                speech = model.generate_speech(
                    input_ids,
                    speaker_embedding,
                    vocoder=vocoder,
                )

            # ----------------------------------------------
            # CPU NumPy
            # ----------------------------------------------

            speech = (
                speech
                .detach()
                .cpu()
                .numpy()
            )

            audio_chunks.append(
                speech
            )

            del inputs
            del input_ids
            del speech

        progress.empty()

        # ----------------------------------------------------
        # Combine chunks
        # ----------------------------------------------------

        if not audio_chunks:

            raise RuntimeError(
                "SpeechT5 did not generate any audio."
            )

        combined_audio = np.concatenate(
            audio_chunks
        )

        # ----------------------------------------------------
        # Create WAV
        # ----------------------------------------------------

        audio_buffer = io.BytesIO()

        sf.write(
            audio_buffer,
            combined_audio,
            SAMPLE_RATE,
            format="WAV",
        )

        audio_buffer.seek(0)

        audio_bytes = audio_buffer.read()

        if not audio_bytes:

            raise RuntimeError(
                "The generated WAV file is empty."
            )

        return audio_bytes

    except Exception as error:

        st.error(
            "TTS error: "
            f"{type(error).__name__}: {error}"
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
# MAIN APPLICATION
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
    # Image
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

        reset_story()

        # ----------------------------------------------------
        # Image → Description
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
        # Description → Story
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
    # Image description
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
        # Audio
        # ----------------------------------------------------

        st.subheader(
            "🔊 Listen to your story"
        )

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
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
