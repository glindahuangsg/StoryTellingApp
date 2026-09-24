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
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoModelForCausalLM,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)

from huggingface_hub import hf_hub_download


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Vision-language model.
#
# Unlike BLIP, this model can directly answer a detailed
# question about the uploaded image.
VISION_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"

# Small text model used to turn the visual facts into a story.
TEXT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Text-to-speech
TTS_MODEL = "microsoft/speecht5_tts"
TTS_VOCODER = "microsoft/speecht5_hifigan"

# Speaker embeddings
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
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="My Story Maker",
    page_icon="🌈",
    layout="centered",
)


# ============================================================
# CSS
# ============================================================

def add_css():

    st.markdown(
        """
        <style>

        .title {
            text-align: center;
            color: #6C63FF;
            font-size: 42px;
            font-weight: 800;
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
        }

        .visual-box {
            background: #F2F7FF;
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 20px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MEMORY
# ============================================================

def cleanup():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# VISION MODEL
# ============================================================

@st.cache_resource
def load_vision_model():

    processor = AutoProcessor.from_pretrained(
        VISION_MODEL
    )

    model = AutoModelForVision2Seq.from_pretrained(
        VISION_MODEL
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


# ============================================================
# IMAGE UNDERSTANDING
# ============================================================

def understand_image(image):

    processor, model = load_vision_model()

    prompt = """
Look carefully at this picture.

You are helping create a gentle story for a child.

Describe ONLY things that you can actually see.

Give me these facts:

MAIN SUBJECTS:
- What people, children, animals, or characters are visible?

OBJECTS:
- What important objects are visible?

SETTING:
- Where does the picture appear to take place?

COLORS:
- What important colors can you see?

ACTIONS:
- What are the visible subjects doing?

MOOD:
- Does the picture look happy, calm, playful, curious, etc.?

IMPORTANT:
- Do not invent names.
- Do not invent people or animals that are not visible.
- Do not invent objects that are not visible.
- Do not guess private information.
- Do not guess what people are thinking.
- Do not invent an exact location.
- If something is unclear, say "unclear".
- Describe the picture rather than writing a story.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text_prompt,
        images=[image],
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )

    input_length = inputs[
        "input_ids"
    ].shape[-1]

    generated = output[
        0,
        input_length:
    ]

    description = processor.decode(
        generated,
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
# AGE RULES
# ============================================================

def age_settings(age):

    if age == "3–5":

        return {
            "sentences": 4,
            "words": "very simple words",
            "tone": "gentle, playful and warm",
            "max_tokens": 90,
        }

    if age == "6–7":

        return {
            "sentences": 6,
            "words": "simple children's vocabulary",
            "tone": "playful, curious and warm",
            "max_tokens": 130,
        }

    return {
        "sentences": 8,
        "words": "easy-to-understand children's vocabulary",
        "tone": "imaginative, playful and warm",
        "max_tokens": 180,
    }


# ============================================================
# STORY SAFETY
# ============================================================

def contains_unsafe_content(text):

    text = text.lower()

    unsafe_words = [
        "gun",
        "weapon",
        "shoot",
        "murder",
        "blood",
        "kill",
        "killed",
        "dead body",
        "suicide",
        "drug",
        "cocaine",
        "knife",
        "bomb",
        "explosion",
        "terrorist",
        "porn",
        "sex",
        "sexual",
    ]

    for word in unsafe_words:

        if re.search(
            r"\b" + re.escape(word) + r"\b",
            text,
        ):

            return True

    return False


# ============================================================
# SENTENCES
# ============================================================

def split_sentences(text):

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
        x.strip()
        for x in sentences
        if x.strip()
    ]


def normalize(text):

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


def remove_duplicates(sentences):

    result = []

    seen = set()

    for sentence in sentences:

        key = normalize(
            sentence
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)

        result.append(sentence)

    return result


# ============================================================
# STORY GENERATION
# ============================================================

def generate_story(
    visual_description,
    age_group,
    style,
):

    tokenizer, model = load_text_model()

    settings = age_settings(
        age_group
    )

    # --------------------------------------------------------
    # VERY IMPORTANT:
    #
    # The visual description is placed prominently in the
    # prompt and the model is told that it may ONLY use
    # those visual facts.
    # --------------------------------------------------------

    prompt = f"""
You are a professional children's story writer.

Write a complete story based on the picture description below.

PICTURE FACTS:
{visual_description}

CHILD AGE:
{age_group}

STORY STYLE:
{style}

LENGTH:
Write exactly {settings["sentences"]} complete sentences.

LANGUAGE:
Use {settings["words"]}.

TONE:
The story should be {settings["tone"]}.

STRICT PICTURE RULES:

1. The story MUST be clearly connected to the picture.
2. Use the main people, animals, or objects that are actually
   visible in the picture.
3. Use the visible setting.
4. Use at least three concrete visual details from the picture.
5. Do NOT introduce a completely unrelated setting.
6. Do NOT replace the visible main subject with another subject.
7. Do NOT invent dangerous objects.
8. Do NOT invent frightening events.
9. Do NOT invent adult themes.
10. Do NOT invent violence.
11. Do NOT invent weapons.
12. Do NOT invent drugs.
13. Do NOT make the child character do anything dangerous.

STORY RULES:

- Give the visible subject a gentle adventure.
- You may use imagination, but the imagination must begin
  with things visible in the picture.
- A visible object can become part of the imaginary adventure.
- Keep the story easy to understand.
- Do not write a title.
- Do not write "Story:".
- Do not write a moral.
- Do not use bullet points.
- Do not repeat sentences.
- Do not repeat the same event.
- Finish the story.
- The final sentence must give the story a warm ending.

OUTPUT ONLY THE STORY.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1000,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,

            max_new_tokens=settings[
                "max_tokens"
            ],

            min_new_tokens=35,

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

    generated = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    story = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    story = story.strip()

    story = re.sub(
        r"^(Story:|story:)\s*",
        "",
        story,
    )

    sentences = split_sentences(
        story
    )

    sentences = remove_duplicates(
        sentences
    )

    # Keep requested number of COMPLETE sentences.

    sentences = sentences[
        :settings["sentences"]
    ]

    story = " ".join(
        sentences
    ).strip()

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if contains_unsafe_content(
        story
    ):

        return None

    # --------------------------------------------------------
    # Completeness check
    # --------------------------------------------------------

    if len(sentences) < settings[
        "sentences"
    ]:

        return None

    return story


# ============================================================
# STORY FALLBACK
# ============================================================

def fallback_story(
    visual_description,
    age_group,
):

    """
    If the language model fails, create a simple story from
    the visual facts instead of showing an unrelated story.
    """

    # Extract lines from the vision model output.

    lines = [
        line.strip()
        for line in visual_description.split(
            "\n"
        )
        if line.strip()
    ]

    facts = " ".join(
        lines
    )

    if age_group == "3–5":

        return (
            f"We looked at the picture and found "
            f"something wonderful: {facts}. "
            f"It was a happy day for everyone in the picture. "
            f"They enjoyed looking around together. "
            f"At the end, everyone smiled and felt happy."
        )

    if age_group == "6–7":

        return (
            f"In the picture, there was "
            f"{facts}. "
            f"Everyone began a little adventure together. "
            f"They noticed many interesting things around them. "
            f"Something surprising made them smile. "
            f"They explored carefully and happily. "
            f"Then they discovered something special. "
            f"The adventure ended with everyone feeling happy."
        )

    return (
        f"The picture showed {facts}. "
        f"It became the beginning of a wonderful little adventure. "
        f"The characters noticed several interesting details around them. "
        f"They decided to explore the scene together. "
        f"Along the way, they discovered something unexpected. "
        f"They used their imagination to make the moment even more special. "
        f"Everyone enjoyed the adventure and shared a happy moment. "
        f"At the end, they went home with wonderful memories."
    )


# ============================================================
# FINAL STORY VALIDATION
# ============================================================

def validate_story(
    story,
    visual_description,
    age_group,
):

    if not story:

        return False

    if contains_unsafe_content(
        story
    ):

        return False

    sentences = split_sentences(
        story
    )

    sentences = remove_duplicates(
        sentences
    )

    settings = age_settings(
        age_group
    )

    if len(sentences) < settings[
        "sentences"
    ]:

        return False

    # --------------------------------------------------------
    # Check visual connection.
    #
    # We take important words from the visual description.
    # If none occur in the story, the story is probably generic.
    # --------------------------------------------------------

    visual_words = re.findall(
        r"\b[a-zA-Z]{4,}\b",
        visual_description.lower(),
    )

    story_lower = story.lower()

    matches = 0

    ignored = {
        "picture",
        "image",
        "there",
        "appears",
        "visible",
        "something",
        "unclear",
        "looks",
        "color",
        "colors",
        "main",
        "subject",
        "setting",
        "action",
        "mood",
        "people",
    }

    for word in set(
        visual_words
    ):

        if word in ignored:
            continue

        if word in story_lower:

            matches += 1

    # Require at least two visual connections.

    if matches < 2:

        return False

    return True


# ============================================================
# GENERATE COMPLETE STORY
# ============================================================

def create_story(
    image,
    age_group,
    style,
):

    # --------------------------------------------------------
    # Step 1: Understand image
    # --------------------------------------------------------

    with st.spinner(
        "👀 Looking carefully at your picture..."
    ):

        visual_description = (
            understand_image(
                image
            )
        )

    # Show visual information to the parent/teacher.

    with st.expander(
        "🔎 What the app noticed"
    ):

        st.write(
            visual_description
        )

    # --------------------------------------------------------
    # Step 2: Generate story
    # --------------------------------------------------------

    with st.spinner(
        "🪄 Creating a story from your picture..."
    ):

        story = generate_story(
            visual_description,
            age_group,
            style,
        )

    # --------------------------------------------------------
    # Step 3: Validate
    # --------------------------------------------------------

    if story is not None:

        valid = validate_story(
            story,
            visual_description,
            age_group,
        )

        if valid:

            return (
                story,
                visual_description,
            )

    # --------------------------------------------------------
    # Step 4: Fallback
    # --------------------------------------------------------

    st.info(
        "The storyteller needed a little help, "
        "so I'm making a simpler picture-based story."
    )

    story = fallback_story(
        visual_description,
        age_group,
    )

    return (
        story,
        visual_description,
    )


# ============================================================
# SPEAKER
# ============================================================

@st.cache_resource
def load_speaker():

    zip_path = hf_hub_download(
        repo_id=SPEAKER_REPO,
        filename="spkrec-xvect.zip",
        repo_type="dataset",
    )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:

        files = sorted(
            [
                name
                for name in archive.namelist()
                if name.endswith(".npy")
            ]
        )

        if SPEAKER_INDEX >= len(files):

            raise RuntimeError(
                "Requested speaker embedding does not exist."
            )

        selected = files[
            SPEAKER_INDEX
        ]

        with archive.open(
            selected
        ) as file:

            embedding = np.load(
                file
            )

    embedding = torch.tensor(
        embedding,
        dtype=torch.float32,
    )

    if embedding.ndim == 1:

        embedding = embedding.unsqueeze(0)

    return embedding


# ============================================================
# TTS
# ============================================================

@st.cache_resource
def load_tts():

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

    return (
        processor,
        model,
        vocoder,
    )


def split_for_tts(text):

    sentences = split_sentences(
        text
    )

    chunks = []

    for sentence in sentences:

        if len(sentence) <= 90:

            chunks.append(
                sentence
            )

        else:

            words = sentence.split()

            current = ""

            for word in words:

                candidate = (
                    f"{current} {word}".strip()
                )

                if len(candidate) <= 90:

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


def text_to_speech(text):

    try:

        chunks = split_for_tts(
            text
        )

        if not chunks:

            return None

        speaker = load_speaker()
        speaker = speaker.to(
            DEVICE
        )

        processor, model, vocoder = (
            load_tts()
        )

        audio_parts = []

        progress = st.progress(
            0,
            text="🔊 Getting the storyteller ready..."
        )

        for index, chunk in enumerate(
            chunks
        ):

            progress.progress(
                (index + 1) / len(chunks),
                text=(
                    f"🔊 Reading "
                    f"{index + 1} of "
                    f"{len(chunks)}..."
                ),
            )

            inputs = processor(
                text=chunk,
                return_tensors="pt",
            )

            input_ids = inputs[
                "input_ids"
            ].to(DEVICE)

            with torch.no_grad():

                speech = (
                    model.generate_speech(
                        input_ids,
                        speaker,
                        vocoder=vocoder,
                        maxlenratio=8.0,
                        minlenratio=1.0,
                    )
                )

            audio = (
                speech
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            # Pause between sentences.

            pause = np.zeros(
                int(
                    SAMPLE_RATE
                    * 0.12
                ),
                dtype=np.float32,
            )

            audio_parts.append(
                np.concatenate(
                    [
                        audio,
                        pause,
                    ]
                )
            )

        progress.empty()

        combined = np.concatenate(
            audio_parts
        )

        buffer = io.BytesIO()

        sf.write(
            buffer,
            combined,
            SAMPLE_RATE,
            format="WAV",
        )

        buffer.seek(0)

        return buffer.read()

    except Exception as error:

        st.error(
            f"TTS error: "
            f"{type(error).__name__}: "
            f"{error}"
        )

        return None


# ============================================================
# RESET
# ============================================================

def reset():

    for key in [
        "story",
        "description",
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

    add_css()

    st.markdown(
        '<div class="title">'
        '🌈 My Story Maker'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Take a picture and turn it into a story!'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Settings
    # --------------------------------------------------------

    st.subheader(
        "1️⃣ Choose your age"
    )

    age_group = st.radio(
        "Age",
        [
            "3–5",
            "6–7",
            "8–10",
        ],
        horizontal=True,
    )

    st.subheader(
        "2️⃣ Choose a story style"
    )

    style = st.selectbox(
        "Style",
        [
            "🐉 Magical",
            "🐾 Animal adventure",
            "🚀 Space adventure",
            "😂 Funny",
            "🌳 Nature adventure",
        ],
        label_visibility="collapsed",
    )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    st.subheader(
        "3️⃣ Choose a picture"
    )

    uploaded = st.file_uploader(
        "Upload a picture",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        label_visibility="collapsed",
    )

    if uploaded is None:

        st.info(
            "📸 Try a photo of a pet, toy, drawing, "
            "park, family-friendly activity, or favorite object."
        )

        return

    try:

        image = Image.open(
            uploaded
        ).convert("RGB")

    except Exception:

        st.error(
            "I couldn't read that image."
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
        "✨ Make My Story ✨",
        type="primary",
        use_container_width=True,
    ):

        reset()

        story, description = (
            create_story(
                image,
                age_group,
                style,
            )
        )

        st.session_state[
            "story"
        ] = story

        st.session_state[
            "description"
        ] = description

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
            st.session_state[
                "story"
            ]
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        if st.button(
            "🔊 Read My Story",
            use_container_width=True,
        ):

            with st.spinner(
                "🎵 Making the story audio..."
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

        # ----------------------------------------------------
        # New story
        # ----------------------------------------------------

        if st.button(
            "🌟 Make Another Story",
            use_container_width=True,
        ):

            reset()

            st.rerun()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
