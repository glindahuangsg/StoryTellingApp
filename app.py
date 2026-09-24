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

# Small enough for Streamlit Cloud CPU.
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
# PAGE CONFIG
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
    Load one SpeechT5 speaker embedding.

    IMPORTANT:
    We do not use datasets.load_dataset().

    The original CMU Arctic repository contains an old
    dataset-loading Python script that causes:

        Dataset scripts are no longer supported

    Instead, we download the x-vector archive directly.
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
# TEXT CLEANING
# ============================================================

def clean_story_text(text):

    """
    Clean common artifacts produced by small language models.
    """

    if not text:

        return ""

    text = text.strip()

    # Remove common prefixes.

    prefixes = [
        "Story:",
        "story:",
        "Here is the story:",
        "Here is a story:",
        "Here’s the story:",
        "Here’s a story:",
    ]

    for prefix in prefixes:

        if text.startswith(prefix):

            text = text[
                len(prefix):
            ].strip()

    # Remove markdown headings.

    text = re.sub(
        r"^#+\s*",
        "",
        text,
    )

    # Normalize whitespace.

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):

    """
    Split English children's story into sentences.
    """

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:

        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# REMOVE REPEATED SENTENCES
# ============================================================

def remove_repeated_sentences(
    sentences,
):

    """
    Remove exact or near-exact repeated sentences.

    This is particularly useful with small language models
    which sometimes generate:

        The dog ran home.
        The dog ran home.
        The dog ran home.

    or slightly modified repetitions.
    """

    result = []

    seen = set()

    for sentence in sentences:

        normalized = sentence.lower()

        normalized = re.sub(
            r"[^a-z0-9\s]",
            "",
            normalized,
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        ).strip()

        if not normalized:

            continue

        # Exact duplicate.

        if normalized in seen:

            continue

        # Detect immediate repetition of a sentence.
        if result:

            previous = result[-1]

            previous_normalized = re.sub(
                r"[^a-z0-9\s]",
                "",
                previous.lower(),
            )

            previous_normalized = re.sub(
                r"\s+",
                " ",
                previous_normalized,
            ).strip()

            if (
                normalized == previous_normalized
            ):

                continue

        seen.add(normalized)

        result.append(sentence)

    return result


# ============================================================
# FINALIZE STORY
# ============================================================

def finalize_story(
    raw_story,
    age_group,
):

    """
    Convert the model output into a clean, complete story.

    We intentionally preserve the complete generated text
    instead of arbitrarily taking only the first paragraph.
    """

    story = clean_story_text(
        raw_story
    )

    sentences = split_sentences(
        story
    )

    sentences = remove_repeated_sentences(
        sentences
    )

    # --------------------------------------------------------
    # Age-specific sentence limits.
    #
    # These are limits, not truncation rules.
    #
    # We only apply them if the model generated MORE than
    # requested. We never cut a sentence in half.
    # --------------------------------------------------------

    if age_group == "3–5":

        maximum_sentences = 5

    elif age_group == "6–7":

        maximum_sentences = 7

    else:

        maximum_sentences = 10

    # Keep complete sentences only.

    if len(sentences) > maximum_sentences:

        sentences = sentences[
            :maximum_sentences
        ]

    story = " ".join(
        sentences
    ).strip()

    # --------------------------------------------------------
    # Remove a repeated ending.
    #
    # Example:
    #
    # They went home happily.
    # They went home happily.
    #
    # This catches some cases where punctuation differs.
    # --------------------------------------------------------

    if len(sentences) >= 2:

        last = re.sub(
            r"[^a-z0-9\s]",
            "",
            sentences[-1].lower(),
        )

        second_last = re.sub(
            r"[^a-z0-9\s]",
            "",
            sentences[-2].lower(),
        )

        if last == second_last:

            sentences = sentences[:-1]

            story = " ".join(
                sentences
            ).strip()

    return story


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
        # Age-specific instructions
        # ----------------------------------------------------

        if age_group == "3–5":

            length_instruction = (
                "Write exactly 4 short sentences."
            )

            max_tokens = 100

        elif age_group == "6–7":

            length_instruction = (
                "Write exactly 6 short sentences."
            )

            max_tokens = 140

        else:

            length_instruction = (
                "Write exactly 8 complete sentences. "
                "Each sentence should be short enough "
                "for a child to understand."
            )

            max_tokens = 180

        # ----------------------------------------------------
        # Styles
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

Create one complete story based on this picture:

{description}

The reader is {age_group} years old.

{length_instruction}

{style_instruction}

IMPORTANT:

- Finish the story.
- Do not stop in the middle.
- Do not repeat any sentence.
- Do not repeat the same event several times.
- Do not write a title.
- Do not write "Story:".
- Do not use bullet points.
- Do not use numbered lists.
- Write exactly the requested number of complete sentences.
- Every sentence must end naturally with punctuation.
- Use simple vocabulary.
- Make the story warm, playful and imaginative.
- Keep it safe for children.
- Do not include violence.
- Do not include weapons.
- Do not include frightening scenes.
- Do not include adult topics.
- Do not include dangerous instructions.
- End with a happy or reassuring ending.
- Output ONLY the story.

"""

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        # ----------------------------------------------------
        # Chat template
        # ----------------------------------------------------

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

                max_new_tokens=max_tokens,

                min_new_tokens=50,

                do_sample=True,

                temperature=0.65,

                top_p=0.88,

                repetition_penalty=1.15,

                no_repeat_ngram_size=4,

                eos_token_id=tokenizer.eos_token_id,

                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Decode ONLY newly generated tokens.
        # ----------------------------------------------------

        generated_tokens = output[
            0,
            inputs["input_ids"].shape[1]:
        ]

        raw_story = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        # ----------------------------------------------------
        # Clean and validate
        # ----------------------------------------------------

        story = finalize_story(
            raw_story,
            age_group,
        )

        # ----------------------------------------------------
        # If the model stopped too early, make one retry.
        # ----------------------------------------------------

        sentence_count = len(
            split_sentences(story)
        )

        if (
            age_group == "8–10"
            and sentence_count < 6
        ):

            # A small model can occasionally stop early.
            # Retry with a stronger instruction.

            retry_prompt = f"""
Write a complete children's story about:

{description}

The child is 8 to 10 years old.

Write exactly 8 complete sentences.

Style:
{style_instruction}

IMPORTANT:
Finish the entire story.
Do not stop early.
Do not repeat sentences.
Do not repeat events.
Use simple language.
End with a happy ending.

Output only the 8 story sentences.
"""

            retry_messages = [
                {
                    "role": "user",
                    "content": retry_prompt,
                }
            ]

            retry_input = (
                tokenizer.apply_chat_template(
                    retry_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            retry_inputs = tokenizer(
                retry_input,
                return_tensors="pt",
            )

            retry_inputs = {
                key: value.to(DEVICE)
                for key, value in retry_inputs.items()
            }

            with torch.no_grad():

                retry_output = model.generate(
                    **retry_inputs,

                    max_new_tokens=180,

                    min_new_tokens=70,

                    do_sample=True,

                    temperature=0.60,

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

            retry_tokens = retry_output[
                0,
                retry_inputs["input_ids"].shape[1]:
            ]

            retry_story = tokenizer.decode(
                retry_tokens,
                skip_special_tokens=True,
            )

            retry_story = finalize_story(
                retry_story,
                age_group,
            )

            retry_sentence_count = len(
                split_sentences(
                    retry_story
                )
            )

            if (
                retry_sentence_count
                > sentence_count
            ):

                story = retry_story

            del retry_output
            del retry_inputs

        # ----------------------------------------------------
        # Final validation
        # ----------------------------------------------------

        story = finalize_story(
            story,
            age_group,
        )

        if not story:

            raise RuntimeError(
                "The story generator returned an empty story."
            )

        return story

    finally:

        del model
        del tokenizer

        cleanup_memory()


# ============================================================
# TTS CHUNKING
# ============================================================

def split_text_for_tts(
    text,
    max_chars=90,
):

    """
    Split the story into VERY short pieces.

    SpeechT5 is much more reliable with short inputs.

    We intentionally use ~90 characters rather than 150.
    """

    sentences = split_sentences(
        text
    )

    chunks = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:

            continue

        # ----------------------------------------------------
        # A normal short sentence can be one TTS chunk.
        # ----------------------------------------------------

        if len(sentence) <= max_chars:

            chunks.append(sentence)

            continue

        # ----------------------------------------------------
        # Long sentence → split by words.
        # ----------------------------------------------------

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

        if current:

            chunks.append(
                current
            )

    return chunks


# ============================================================
# REMOVE AUDIO CHUNK OVERLAP
# ============================================================

def add_audio_silence(
    audio,
    milliseconds=80,
):

    """
    Add a tiny pause between sentences.

    This makes the final audio sound natural and also
    prevents the end of one SpeechT5 generation from
    sounding like it runs into the next one.
    """

    silence_length = int(
        SAMPLE_RATE
        * milliseconds
        / 1000
    )

    silence = np.zeros(
        silence_length,
        dtype=np.float32,
    )

    return np.concatenate(
        [
            audio,
            silence,
        ]
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

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Use the FINAL story shown on screen.
        #
        # We don't regenerate or reconstruct the story here.
        # ----------------------------------------------------

        chunks = split_text_for_tts(
            text,
            max_chars=90,
        )

        if not chunks:

            raise ValueError(
                "The story contains no readable text."
            )

        st.info(
            f"🔊 Reading "
            f"{len(chunks)} short sentences..."
        )

        # ----------------------------------------------------
        # Speaker
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
        # Audio
        # ----------------------------------------------------

        audio_chunks = []

        progress = st.progress(
            0,
            text="Preparing the storyteller..."
        )

        for index, chunk in enumerate(
            chunks
        ):

            progress.progress(
                (index + 1) / len(chunks),
                text=(
                    f"🔊 Reading sentence "
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

            token_count = (
                input_ids.shape[1]
            )

            # ------------------------------------------------
            # Extra safety.
            #
            # We are intentionally well below 600.
            # ------------------------------------------------

            if token_count >= 300:

                raise ValueError(
                    "A sentence is unexpectedly long "
                    f"({token_count} tokens). "
                    "Please try a shorter story."
                )

            # ------------------------------------------------
            # Generate speech.
            #
            # maxlenratio limits how long SpeechT5 can
            # continue generating audio.
            #
            # This is important for preventing a generated
            # sentence from repeating its ending.
            # ------------------------------------------------

            with torch.no_grad():

                speech = model.generate_speech(
                    input_ids,
                    speaker_embedding,
                    vocoder=vocoder,

                    # Prevent excessively long output.
                    maxlenratio=10.0,

                    # Avoid extremely short output.
                    minlenratio=1.0,
                )

            speech = (
                speech
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            if speech.size == 0:

                raise RuntimeError(
                    f"SpeechT5 returned empty audio "
                    f"for sentence {index + 1}."
                )

            # ------------------------------------------------
            # Add a small pause.
            # ------------------------------------------------

            speech = add_audio_silence(
                speech,
                milliseconds=80,
            )

            audio_chunks.append(
                speech
            )

            del inputs
            del input_ids
            del speech

            cleanup_memory()

        progress.empty()

        # ----------------------------------------------------
        # Combine audio
        # ----------------------------------------------------

        if not audio_chunks:

            raise RuntimeError(
                "No audio was generated."
            )

        combined_audio = np.concatenate(
            audio_chunks
        )

        # ----------------------------------------------------
        # WAV
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
    # Image
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
        # Image → description
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
        # Description → story
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

            if audio is not None:

                st.session_state[
                    "audio"
                ] = audio

                st.success(
                    "🎉 Your story is ready!"
                )

        # ----------------------------------------------------
        # Player
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
                label="⬇️ Download audio",
                data=st.session_state[
                    "audio"
                ],
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
