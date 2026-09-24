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
    AutoModelForImageTextToText,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)

from huggingface_hub import hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

# One vision-language model does both:
#
#   IMAGE -> UNDERSTANDING -> STORY
#
# This removes the additional SmolLM2 model and reduces
# memory usage on Streamlit Cloud.
VISION_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"

# Text-to-speech
TTS_MODEL = "microsoft/speecht5_tts"
TTS_VOCODER = "microsoft/speecht5_hifigan"

# Speaker embedding
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

def add_css():

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
# MEMORY CLEANUP
# ============================================================

def cleanup_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# AGE SETTINGS
# ============================================================

def get_age_settings(age_group):

    if age_group == "3–5":

        return {
            "sentences": 4,
            "max_new_tokens": 180,
            "language": (
                "Use very simple words and short sentences."
            ),
            "tone": (
                "gentle, cheerful, playful and reassuring"
            ),
        }

    if age_group == "6–7":

        return {
            "sentences": 6,
            "max_new_tokens": 240,
            "language": (
                "Use simple children's vocabulary and "
                "clear sentences."
            ),
            "tone": (
                "playful, curious, warm and imaginative"
            ),
        }

    return {
        "sentences": 8,
        "max_new_tokens": 320,
        "language": (
            "Use easy-to-understand vocabulary suitable "
            "for children aged 8 to 10."
        ),
        "tone": (
            "imaginative, funny, adventurous and warm"
        ),
    }


# ============================================================
# VISION-LANGUAGE MODEL
# ============================================================

@st.cache_resource
def load_vision_model():

    processor = AutoProcessor.from_pretrained(
        VISION_MODEL
    )

    model = AutoModelForImageTextToText.from_pretrained(
        VISION_MODEL
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


# ============================================================
# IMAGE DESCRIPTION
# ============================================================

def analyze_image(image):

    processor, model = load_vision_model()

    prompt = """
Look very carefully at this picture.

You are preparing a children's story based ONLY on what
can actually be seen in the picture.

Describe the important visual facts.

Include:

1. MAIN SUBJECTS
   - people
   - children
   - animals
   - toys
   - other important subjects

2. IMPORTANT OBJECTS
   - books
   - balls
   - bicycles
   - food
   - furniture
   - vehicles
   - toys
   - etc.

3. SETTING
   - indoors or outdoors
   - room, garden, park, beach, school, etc.
   - only say something if it is reasonably visible

4. COLORS
   - important colors

5. ACTIONS
   - what the visible subjects appear to be doing

6. VISUAL DETAILS
   - clothing
   - size
   - shapes
   - interesting details

IMPORTANT:

- Only describe things that are visible.
- Do not invent people.
- Do not invent animals.
- Do not invent objects.
- Do not invent a location that cannot reasonably be inferred.
- Do not guess people's names.
- Do not guess private information.
- Do not guess thoughts or feelings as facts.
- If something is unclear, say "unclear".
- Do NOT write a story yet.
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

    cleanup_memory()

    return description.strip()


# ============================================================
# STORY SAFETY
# ============================================================

def contains_unsafe_content(text):

    text_lower = text.lower()

    unsafe_patterns = [
        r"\bgun\b",
        r"\bguns\b",
        r"\bweapon\b",
        r"\bweapons\b",
        r"\bshoot\b",
        r"\bshooting\b",
        r"\bmurder\b",
        r"\bkill\b",
        r"\bkilled\b",
        r"\bblood\b",
        r"\bdead body\b",
        r"\bsuicide\b",
        r"\bdrug\b",
        r"\bdrugs\b",
        r"\bcocaine\b",
        r"\bknife\b",
        r"\bknives\b",
        r"\bbomb\b",
        r"\bterrorist\b",
        r"\bporn\b",
        r"\bsexual\b",
    ]

    for pattern in unsafe_patterns:

        if re.search(
            pattern,
            text_lower,
        ):

            return True

    return False


# ============================================================
# SENTENCE FUNCTIONS
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
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


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


def remove_duplicate_sentences(
    sentences
):

    result = []
    seen = set()

    for sentence in sentences:

        normalized = normalize_sentence(
            sentence
        )

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(sentence)

    return result


# ============================================================
# STORY GENERATION
# ============================================================

def generate_story(
    image,
    visual_description,
    age_group,
    story_style,
):

    processor, model = load_vision_model()

    settings = get_age_settings(
        age_group
    )

    prompt = f"""
You are a kind and imaginative children's storyteller.

Look at the picture and the visual notes below.

VISUAL NOTES:
{visual_description}

Now write a complete story inspired by THIS PICTURE.

CHILD'S AGE:
{age_group}

STORY STYLE:
{story_style}

LENGTH:
Write exactly {settings["sentences"]} complete sentences.

LANGUAGE:
{settings["language"]}

TONE:
The story should be {settings["tone"]}.

==================================================
MOST IMPORTANT RULE: STAY CONNECTED TO THE PICTURE
==================================================

The story must clearly describe and use things that are
actually visible in the picture.

Use several concrete visual details, such as:

- the main person or child
- an animal
- an important object
- a toy
- a book
- a ball
- a bicycle
- a visible color
- the visible setting
- a visible action

The story should feel as though the adventure begins
INSIDE THIS PICTURE.

You may use gentle imagination.

For example:

If there is a red ball, the ball can become magical.

If there is a dog, the dog can become the child's
adventure companion.

If there are books, the books can lead to an imaginary
adventure.

If there is a garden, the garden can become an enchanted
garden.

But do NOT replace the picture with a completely
unrelated story.

==================================================
CHILD SAFETY
==================================================

The story must be suitable for children aged {age_group}.

Do NOT include:

- violence
- weapons
- fighting
- blood
- death
- murder
- frightening horror
- dangerous instructions
- drugs
- adult themes
- sexual content
- hateful content
- cruelty
- self-harm

Keep the adventure safe and reassuring.

==================================================
WRITING RULES
==================================================

- Write exactly {settings["sentences"]} sentences.
- Every sentence must be complete.
- Do not repeat a sentence.
- Do not repeat the same event.
- Do not use a title.
- Do not use bullet points.
- Do not say "Story:".
- Do not mention AI.
- Do not explain what you are doing.
- Do not leave the story unfinished.
- The final sentence must provide a satisfying, happy ending.
- Output ONLY the story.

Begin the story now.
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

            max_new_tokens=settings[
                "max_new_tokens"
            ],

            do_sample=True,

            temperature=0.65,

            top_p=0.90,

            repetition_penalty=1.15,

            no_repeat_ngram_size=4,

            eos_token_id=(
                processor.tokenizer.eos_token_id
                if hasattr(
                    processor,
                    "tokenizer",
                )
                else None
            ),
        )

    input_length = inputs[
        "input_ids"
    ].shape[-1]

    generated = output[
        0,
        input_length:
    ]

    story = processor.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    # --------------------------------------------------------
    # Remove accidental prefixes.
    # --------------------------------------------------------

    story = re.sub(
        r"^(Story:|story:)\s*",
        "",
        story,
    )

    story = re.sub(
        r"^Here is.*?:\s*",
        "",
        story,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Split and remove duplicates.
    # --------------------------------------------------------

    sentences = split_sentences(
        story
    )

    sentences = remove_duplicate_sentences(
        sentences
    )

    # Keep only complete requested sentences.

    sentences = sentences[
        :settings["sentences"]
    ]

    story = " ".join(
        sentences
    ).strip()

    cleanup_memory()

    return story


# ============================================================
# STORY VALIDATION
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

    settings = get_age_settings(
        age_group
    )

    sentences = split_sentences(
        story
    )

    sentences = remove_duplicate_sentences(
        sentences
    )

    # --------------------------------------------------------
    # Must contain enough complete sentences.
    # --------------------------------------------------------

    if len(sentences) < settings[
        "sentences"
    ]:

        return False

    # --------------------------------------------------------
    # Check that the story is actually grounded in the image.
    #
    # Look for concrete words shared between the visual
    # description and story.
    # --------------------------------------------------------

    visual_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            visual_description.lower(),
        )
    )

    story_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            story.lower(),
        )
    )

    ignored_words = {
        "picture",
        "image",
        "visible",
        "unclear",
        "something",
        "there",
        "appears",
        "looks",
        "main",
        "subject",
        "setting",
        "action",
        "mood",
        "people",
        "objects",
        "colors",
        "color",
        "children",
    }

    useful_visual_words = (
        visual_words
        - ignored_words
    )

    matches = (
        useful_visual_words
        & story_words
    )

    # Require at least two concrete visual connections.

    if len(matches) < 2:

        return False

    return True


# ============================================================
# FALLBACK STORY
# ============================================================

def fallback_story(
    visual_description,
    age_group,
):

    """
    Used if the model produces an incomplete or unsafe story.

    This is intentionally simple but guarantees that the
    application has a complete story instead of showing
    an unfinished generation.
    """

    settings = get_age_settings(
        age_group
    )

    # Keep the visual description compact.

    description = re.sub(
        r"\s+",
        " ",
        visual_description,
    ).strip()

    if age_group == "3–5":

        sentences = [
            f"In the picture, we can see {description}.",
            "It looked like the beginning of a happy little adventure.",
            "Everyone enjoyed discovering the wonderful scene together.",
            "At the end, everyone smiled and felt happy.",
        ]

    elif age_group == "6–7":

        sentences = [
            f"The picture showed {description}.",
            "It looked like the perfect place for a small adventure.",
            "Everyone noticed something interesting in the scene.",
            "They explored carefully and happily.",
            "Soon, they discovered a special moment together.",
            "The adventure ended with everyone smiling.",
        ]

    else:

        sentences = [
            f"The picture showed {description}.",
            "It looked like the beginning of a wonderful adventure.",
            "The characters noticed several interesting details around them.",
            "They decided to explore the scene together.",
            "Along the way, something unexpected caught their attention.",
            "They used their imagination to turn the moment into a special adventure.",
            "Everyone enjoyed the experience and shared a happy moment.",
            "At the end, they went home with wonderful memories.",
        ]

    return " ".join(
        sentences[
            :settings["sentences"]
        ]
    )


# ============================================================
# COMPLETE STORY CREATION
# ============================================================

def create_story(
    image,
    age_group,
    story_style,
):

    # --------------------------------------------------------
    # 1. Analyze image
    # --------------------------------------------------------

    with st.spinner(
        "👀 Looking carefully at your picture..."
    ):

        visual_description = analyze_image(
            image
        )

    # --------------------------------------------------------
    # Show what the model saw.
    # --------------------------------------------------------

    with st.expander(
        "🔎 What I noticed in the picture"
    ):

        st.write(
            visual_description
        )

    # --------------------------------------------------------
    # 2. Generate story
    # --------------------------------------------------------

    with st.spinner(
        "🪄 Creating a story from your picture..."
    ):

        story = generate_story(
            image=image,
            visual_description=visual_description,
            age_group=age_group,
            story_style=story_style,
        )

    # --------------------------------------------------------
    # 3. Validate
    # --------------------------------------------------------

    if validate_story(
        story,
        visual_description,
        age_group,
    ):

        return (
            story,
            visual_description,
        )

    # --------------------------------------------------------
    # 4. Retry once with stronger instruction.
    # --------------------------------------------------------

    st.info(
        "🪄 The first story wasn't quite right, "
        "so I'm trying again with the picture details."
    )

    story = generate_story(
        image=image,
        visual_description=visual_description,
        age_group=age_group,
        story_style=story_style,
    )

    if validate_story(
        story,
        visual_description,
        age_group,
    ):

        return (
            story,
            visual_description,
        )

    # --------------------------------------------------------
    # 5. Guaranteed fallback.
    # --------------------------------------------------------

    story = fallback_story(
        visual_description,
        age_group,
    )

    return (
        story,
        visual_description,
    )


# ============================================================
# SPEAKER EMBEDDING
# ============================================================

@st.cache_resource
def load_speaker_embedding():

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
                "Speaker embedding not found."
            )

        selected_file = files[
            SPEAKER_INDEX
        ]

        with archive.open(
            selected_file
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
# TTS MODELS
# ============================================================

@st.cache_resource
def load_tts_models():

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


# ============================================================
# TTS SENTENCE SPLITTING
# ============================================================

def split_for_tts(text):

    sentences = split_sentences(
        text
    )

    chunks = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Keep chunks short enough for SpeechT5.

        if len(sentence) <= 90:

            chunks.append(
                sentence
            )

            continue

        # ----------------------------------------------------
        # Split long sentences by words.
        # ----------------------------------------------------

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


# ============================================================
# TEXT TO SPEECH
# ============================================================

def text_to_speech(text):

    try:

        chunks = split_for_tts(
            text
        )

        if not chunks:

            raise ValueError(
                "There is no text to read."
            )

        speaker = load_speaker_embedding()

        speaker = speaker.to(
            DEVICE
        )

        (
            processor,
            model,
            vocoder,
        ) = load_tts_models()

        audio_parts = []

        progress = st.progress(
            0,
            text="🔊 Preparing the storyteller..."
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
                .astype(np.float32)
            )

            # Small pause between sentences.

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

        combined_audio = np.concatenate(
            audio_parts
        )

        buffer = io.BytesIO()

        sf.write(
            buffer,
            combined_audio,
            SAMPLE_RATE,
            format="WAV",
        )

        buffer.seek(0)

        audio_bytes = buffer.read()

        if not audio_bytes:

            raise RuntimeError(
                "No audio was generated."
            )

        return audio_bytes

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

def reset_app():

    for key in [
        "story",
        "description",
        "audio",
    ]:

        if key in st.session_state:

            del st.session_state[key]


# ============================================================
# MAIN
# ============================================================

def main():

    add_css()

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
        'Turn your picture into a magical story!'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    st.subheader(
        "1️⃣ Who is the story for?"
    )

    age_group = st.radio(
        "Choose an age",
        [
            "3–5",
            "6–7",
            "8–10",
        ],
        horizontal=True,
    )

    # --------------------------------------------------------
    # Style
    # --------------------------------------------------------

    st.subheader(
        "2️⃣ Choose an adventure"
    )

    story_style = st.selectbox(
        "Story style",
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
    # Upload
    # --------------------------------------------------------

    st.subheader(
        "3️⃣ Choose a picture"
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
            "I couldn't open this picture. "
            "Please try a JPG or PNG image."
        )

        return

    st.image(
        image,
        caption="Your picture",
        width="stretch",
    )

    # --------------------------------------------------------
    # Generate story
    # --------------------------------------------------------

    if st.button(
        "✨ Make My Story ✨",
        type="primary",
        width="stretch",
    ):

        reset_app()

        try:

            (
                story,
                description,
            ) = create_story(
                image=image,
                age_group=age_group,
                story_style=story_style,
            )

            # Store EXACTLY what is displayed.

            st.session_state[
                "story"
            ] = story

            st.session_state[
                "description"
            ] = description

        except Exception as error:

            st.error(
                "I couldn't make the story right now."
            )

            st.exception(error)

            return

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

        st.subheader(
            "🔊 Listen to your story"
        )

        if st.button(
            "🎵 Read My Story",
            width="stretch",
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
                "⬇️ Download audio",
                data=st.session_state[
                    "audio"
                ],
                file_name="my_story.wav",
                mime="audio/wav",
                width="stretch",
            )

        # ----------------------------------------------------
        # New story
        # ----------------------------------------------------

        st.write("")

        if st.button(
            "🌟 Make Another Story",
            width="stretch",
        ):

            reset_app()

            st.rerun()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
