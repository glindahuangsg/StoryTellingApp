import gc
import io
import re
import zipfile

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

# Small model suitable for Streamlit Cloud CPU.
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
# STREAMLIT
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
            color: #666;
            font-size: 19px;
            margin-bottom: 25px;
        }

        .story-box {
            background: #FFF8E7;
            border: 2px solid #FFE29A;
            border-radius: 20px;
            padding: 25px;
            font-size: 20px;
            line-height: 1.8;
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
# MEMORY
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
    Download the CMU Arctic x-vector archive directly.

    We deliberately do NOT use datasets.load_dataset().
    """

    zip_path = hf_hub_download(
        repo_id=SPEAKER_REPO,
        filename="spkrec-xvect.zip",
        repo_type="dataset",
    )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:

        npy_files = sorted(
            [
                name
                for name in archive.namelist()
                if name.endswith(".npy")
            ]
        )

        if SPEAKER_INDEX >= len(npy_files):

            raise RuntimeError(
                f"Speaker index {SPEAKER_INDEX} unavailable. "
                f"Found {len(npy_files)} files."
            )

        selected_file = npy_files[
            SPEAKER_INDEX
        ]

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

@st.cache_resource
def load_vision_model():

    processor = BlipProcessor.from_pretrained(
        VISION_MODEL
    )

    model = BlipForConditionalGeneration.from_pretrained(
        VISION_MODEL
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


def image_to_text(image):

    processor, model = load_vision_model()

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


# ============================================================
# TEXT MODEL
# ============================================================

@st.cache_resource
def load_text_model():

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL
    )

    model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ============================================================
# SENTENCE CLEANING
# ============================================================

def clean_sentence(text):

    if not text:
        return ""

    text = text.strip()

    # Remove common prefixes.

    prefixes = [
        "Story:",
        "story:",
        "Sentence:",
        "sentence:",
        "Next sentence:",
        "Next:",
        "The next sentence is:",
    ]

    for prefix in prefixes:

        if text.startswith(prefix):

            text = text[
                len(prefix):
            ].strip()

    # Remove markdown.

    text = re.sub(
        r"^[-*]\s*",
        "",
        text,
    )

    text = re.sub(
        r"^#+\s*",
        "",
        text,
    )

    # Convert newlines to spaces.

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # If model generated multiple sentences,
    # keep only the first complete sentence.

    match = re.search(
        r"^(.+?[.!?])(?:\s|$)",
        text,
    )

    if match:

        text = match.group(1)

    # Make sure sentence has punctuation.

    if text and text[-1] not in ".!?":

        text += "."

    return text.strip()


# ============================================================
# SENTENCE DUPLICATE CHECK
# ============================================================

def normalize_sentence(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def is_duplicate_sentence(
    sentence,
    existing_sentences,
):

    normalized = normalize_sentence(
        sentence
    )

    if not normalized:
        return True

    for previous in existing_sentences:

        previous_normalized = normalize_sentence(
            previous
        )

        if normalized == previous_normalized:

            return True

    return False


# ============================================================
# GENERATE ONE SENTENCE
# ============================================================

def generate_one_sentence(
    tokenizer,
    model,
    prompt,
    max_new_tokens=45,
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=850,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,

            max_new_tokens=max_new_tokens,

            min_new_tokens=8,

            do_sample=True,

            temperature=0.65,

            top_p=0.88,

            repetition_penalty=1.18,

            no_repeat_ngram_size=4,

            eos_token_id=tokenizer.eos_token_id,

            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )

    generated_tokens = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    result = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return clean_sentence(result)


# ============================================================
# STORY GENERATION
# ============================================================

def generate_story(
    description,
    age_group,
    story_style,
):

    tokenizer, model = load_text_model()

    # --------------------------------------------------------
    # Number of sentences
    # --------------------------------------------------------

    if age_group == "3–5":

        sentence_count = 4

    elif age_group == "6–7":

        sentence_count = 6

    else:

        sentence_count = 8

    # --------------------------------------------------------
    # Style
    # --------------------------------------------------------

    styles = {

        "🐉 Magical":
            "gentle magical adventure",

        "🚀 Adventure":
            "fun and safe adventure",

        "🐾 Animal":
            "friendly animal adventure",

        "😂 Funny":
            "funny and silly adventure",
    }

    style = styles.get(
        story_style,
        "warm children's adventure",
    )

    # --------------------------------------------------------
    # Story structure
    # --------------------------------------------------------

    if sentence_count == 4:

        roles = [
            "Introduce the main character and setting.",
            "Introduce a small fun surprise or problem.",
            "Show the character solving the problem.",
            "End with a happy and reassuring ending.",
        ]

    elif sentence_count == 6:

        roles = [
            "Introduce the main character and setting.",
            "Introduce an interesting discovery.",
            "Describe a small problem or challenge.",
            "Show the character trying to solve it.",
            "Show the character succeeding.",
            "End with a happy and reassuring ending.",
        ]

    else:

        roles = [
            "Introduce the main character and setting.",
            "Introduce an interesting discovery or mystery.",
            "Describe something surprising that happens.",
            "Show the character beginning a gentle adventure.",
            "Add a small safe challenge.",
            "Show the character finding a clever solution.",
            "Show what happens after the problem is solved.",
            "End with a warm, happy and reassuring ending.",
        ]

    story_sentences = []

    progress = st.progress(
        0,
        text="Creating the story..."
    )

    for index in range(
        sentence_count
    ):

        role = roles[index]

        previous_story = " ".join(
            story_sentences
        )

        # ----------------------------------------------------
        # Build context.
        # ----------------------------------------------------

        if previous_story:

            context = (
                "Story written so far:\n"
                f"{previous_story}\n\n"
            )

        else:

            context = ""

        # ----------------------------------------------------
        # Ending instruction
        # ----------------------------------------------------

        if index == sentence_count - 1:

            ending_instruction = """
This is the FINAL sentence.
Finish the story completely.
Give it a happy, warm and reassuring ending.
Do not introduce a new problem.
Do not leave the story unfinished.
"""

        else:

            ending_instruction = """
This is NOT the final sentence.
Continue the story naturally.
Do not end the story yet.
"""

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = f"""
You are writing a story for a child aged {age_group}.

The picture shows:
{description}

The story style is:
{style}

{context}

Write exactly ONE new sentence.

The purpose of this sentence is:
{role}

Rules:

- Write only one sentence.
- Do not repeat an earlier sentence.
- Do not repeat the same event.
- Use simple language.
- Keep the story imaginative and positive.
- Keep it safe for children.
- No violence.
- No weapons.
- No frightening scenes.
- No adult topics.
- No dangerous instructions.
- Do not mention AI.
- Do not write a title.
- Do not write "Story:".
- Do not use bullet points.
- Do not use quotation marks around the sentence.

{ending_instruction}

New sentence:
"""

        # ----------------------------------------------------
        # Try up to 3 times if duplicate/empty.
        # ----------------------------------------------------

        sentence = ""

        for attempt in range(3):

            candidate = generate_one_sentence(
                tokenizer,
                model,
                prompt,
                max_new_tokens=50,
            )

            if (
                candidate
                and not is_duplicate_sentence(
                    candidate,
                    story_sentences,
                )
            ):

                sentence = candidate
                break

            # Make retry more explicit.

            prompt += f"""

IMPORTANT RETRY:
The previous attempt was not acceptable.
Write a completely different sentence.
Do not repeat these sentences:

{" ".join(story_sentences)}
"""

        if not sentence:

            # ------------------------------------------------
            # Safe fallback.
            #
            # We would rather have a complete short story
            # than silently display an incomplete one.
            # ------------------------------------------------

            fallback_sentences = [
                "The adventure continued with a bright smile.",
                "Everyone discovered something wonderful together.",
                "They laughed and shared the happy moment.",
                "At last, they returned home feeling proud and happy.",
            ]

            for fallback in fallback_sentences:

                if not is_duplicate_sentence(
                    fallback,
                    story_sentences,
                ):

                    sentence = fallback
                    break

        story_sentences.append(
            sentence
        )

        progress.progress(
            (index + 1) / sentence_count,
            text=(
                f"Writing sentence "
                f"{index + 1} of "
                f"{sentence_count}..."
            ),
        )

    progress.empty()

    # --------------------------------------------------------
    # Final story
    # --------------------------------------------------------

    story = " ".join(
        story_sentences
    ).strip()

    # --------------------------------------------------------
    # Final duplicate protection
    # --------------------------------------------------------

    cleaned_sentences = []

    for sentence in story_sentences:

        if not is_duplicate_sentence(
            sentence,
            cleaned_sentences,
        ):

            cleaned_sentences.append(
                sentence
            )

    story = " ".join(
        cleaned_sentences
    ).strip()

    # --------------------------------------------------------
    # Ensure complete ending.
    # --------------------------------------------------------

    if cleaned_sentences:

        last_sentence = cleaned_sentences[-1]

        ending_words = (
            "happy",
            "smile",
            "home",
            "wonderful",
            "joy",
            "proud",
            "together",
        )

        # If the final generated sentence doesn't appear
        # to contain an ending, append a gentle ending.
        #
        # This is intentionally conservative.
        if not any(
            word in last_sentence.lower()
            for word in ending_words
        ):

            if age_group == "8–10":

                cleaned_sentences[-1] = (
                    last_sentence.rstrip(".!?")
                    + ". At the end of the adventure, "
                    "everyone went home with happy hearts."
                )

            else:

                cleaned_sentences[-1] = (
                    last_sentence.rstrip(".!?")
                    + ". Everyone ended the day "
                    "with a big happy smile."
                )

    story = " ".join(
        cleaned_sentences
    ).strip()

    cleanup_memory()

    return story


# ============================================================
# TTS
# ============================================================

def split_text_for_tts(
    text,
    max_chars=90,
):

    """
    One short sentence per TTS chunk.

    Keeping chunks short dramatically reduces SpeechT5
    repetition problems.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    chunks = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) <= max_chars:

            chunks.append(sentence)

            continue

        # Long sentence → split by words.

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
                    chunks.append(current)

                current = word

        if current:
            chunks.append(current)

    return chunks


# ============================================================
# AUDIO
# ============================================================

def add_silence(
    audio,
    milliseconds=100,
):

    length = int(
        SAMPLE_RATE
        * milliseconds
        / 1000
    )

    silence = np.zeros(
        length,
        dtype=np.float32,
    )

    return np.concatenate(
        [audio, silence]
    )


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

        chunks = split_text_for_tts(
            text,
            max_chars=90,
        )

        if not chunks:

            raise ValueError(
                "No readable sentences found."
            )

        st.info(
            f"🔊 Reading {len(chunks)} "
            "short parts..."
        )

        # ----------------------------------------------------
        # Speaker
        # ----------------------------------------------------

        speaker = load_speaker_embedding()

        speaker = speaker.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        processor = SpeechT5Processor.from_pretrained(
            TTS_MODEL
        )

        model = SpeechT5ForTextToSpeech.from_pretrained(
            TTS_MODEL
        )

        model.to(DEVICE)
        model.eval()

        vocoder = SpeechT5HifiGan.from_pretrained(
            TTS_VOCODER
        )

        vocoder.to(DEVICE)
        vocoder.eval()

        audio_parts = []

        progress = st.progress(
            0,
            text="Preparing audio..."
        )

        for index, chunk in enumerate(
            chunks
        ):

            progress.progress(
                (index + 1) / len(chunks),
                text=(
                    f"🔊 Reading part "
                    f"{index + 1} of "
                    f"{len(chunks)}..."
                ),
            )

            # ------------------------------------------------
            # Tokenize
            # ------------------------------------------------

            inputs = processor(
                text=chunk,
                return_tensors="pt",
            )

            input_ids = inputs[
                "input_ids"
            ].to(DEVICE)

            # ------------------------------------------------
            # Safety limit
            # ------------------------------------------------

            if input_ids.shape[1] >= 300:

                raise ValueError(
                    "One sentence is too long "
                    "for reliable speech generation."
                )

            # ------------------------------------------------
            # Generate
            # ------------------------------------------------

            with torch.no_grad():

                speech = model.generate_speech(
                    input_ids,
                    speaker,
                    vocoder=vocoder,
                    maxlenratio=8.0,
                    minlenratio=1.0,
                )

            speech = (
                speech
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if speech.size == 0:

                raise RuntimeError(
                    "SpeechT5 generated empty audio."
                )

            # ------------------------------------------------
            # Pause between sentences.
            # ------------------------------------------------

            speech = add_silence(
                speech,
                milliseconds=120,
            )

            audio_parts.append(
                speech
            )

            del inputs
            del input_ids
            del speech

            cleanup_memory()

        progress.empty()

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        combined = np.concatenate(
            audio_parts
        )

        # ----------------------------------------------------
        # WAV
        # ----------------------------------------------------

        buffer = io.BytesIO()

        sf.write(
            buffer,
            combined,
            SAMPLE_RATE,
            format="WAV",
        )

        buffer.seek(0)

        audio_bytes = buffer.read()

        if not audio_bytes:

            raise RuntimeError(
                "Generated audio is empty."
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

    st.subheader(
        "✨ Choose your story"
    )

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
    # Upload
    # --------------------------------------------------------

    st.subheader(
        "📸 Choose a picture"
    )

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
    # Image
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "I couldn't open that picture."
        )

        return

    st.image(
        image,
        caption="Your picture",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    if st.button(
        "✨ Make My Story! ✨",
        type="primary",
        use_container_width=True,
    ):

        reset_story()

        # ----------------------------------------------------
        # Vision
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
        # Story
        # ----------------------------------------------------

        story = generate_story(
            description,
            age_group,
            story_style,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Store the exact final story that is displayed.
        # TTS later reads this exact string.
        # ----------------------------------------------------

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

        st.subheader(
            "📖 Your Story"
        )

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

            if audio:

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
                st.session_state[
                    "audio"
                ],
                format="audio/wav",
            )

            st.download_button(
                "⬇️ Download audio",
                data=st.session_state[
                    "audio"
                ],
                file_name="my_story.wav",
                mime="audio/wav",
                use_container_width=True,
            )

        st.write("")

        # ----------------------------------------------------
        # New story
        # ----------------------------------------------------

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
