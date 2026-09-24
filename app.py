import base64
import io
import re

import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient


# ============================================================
# CONFIGURATION
# ============================================================

# We DO NOT load this model with transformers locally.
# Hugging Face Inference Providers will run the model remotely.

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

MAX_IMAGE_SIZE = 1024


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

def add_css():

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
            font-size: 19px;
            color: #666666;
            margin-bottom: 25px;
        }

        .story-box {
            background: #FFF8E7;
            border: 2px solid #FFD86B;
            border-radius: 20px;
            padding: 25px;
            font-size: 20px;
            line-height: 1.8;
            margin-top: 15px;
            margin-bottom: 20px;
        }

        .tip-box {
            background: #F1F7FF;
            border-radius: 18px;
            padding: 20px;
            margin-top: 15px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

@st.cache_resource
def get_hf_client(token):

    return InferenceClient(
        provider="auto",
        api_key=token,
    )


# ============================================================
# GET TOKEN
# ============================================================

def get_hf_token():

    # Recommended:
    #
    # Streamlit Cloud:
    # Settings → Secrets
    #
    # HF_TOKEN = "hf_xxxxxxxxxxxxx"

    try:

        token = st.secrets["HF_TOKEN"]

    except Exception:

        token = None

    if not token:

        st.error(
            "Hugging Face API token is missing."
        )

        st.info(
            "Add HF_TOKEN to your Streamlit secrets."
        )

        st.code(
            'HF_TOKEN = "hf_your_token_here"'
        )

        st.stop()

    return token


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    image.thumbnail(
        (
            MAX_IMAGE_SIZE,
            MAX_IMAGE_SIZE,
        ),
        Image.Resampling.LANCZOS,
    )

    return image


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_data_url(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=85,
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# AGE SETTINGS
# ============================================================

def get_age_settings(age_group):

    if age_group == "3–5":

        return {
            "sentences": 5,
            "max_tokens": 220,
            "language": (
                "Use very simple vocabulary, "
                "short sentences, and ideas that "
                "a preschool child can understand."
            ),
        }

    if age_group == "6–7":

        return {
            "sentences": 7,
            "max_tokens": 300,
            "language": (
                "Use simple vocabulary and clear "
                "sentences suitable for young children."
            ),
        }

    return {
        "sentences": 9,
        "max_tokens": 380,
        "language": (
            "Use vocabulary and sentence structures "
            "suitable for children aged 8 to 10."
        ),
    }


# ============================================================
# STORY PROMPT
# ============================================================

def create_story_prompt(
    age_group,
    story_style,
):

    settings = get_age_settings(
        age_group
    )

    return f"""
You are a warm, creative children's storyteller.

Look carefully at the uploaded picture.

Create a COMPLETE children's story that is clearly
connected to THIS SPECIFIC PICTURE.

CHILD AGE:
{age_group}

STORY STYLE:
{story_style}

LANGUAGE:
{settings["language"]}

==================================================
PICTURE CONNECTION
==================================================

The story must use several things that are actually
visible in the picture.

Pay particular attention to:

- people
- children
- animals
- toys
- books
- vehicles
- furniture
- clothing
- colors
- objects
- scenery
- visible actions
- the visible setting

The story should feel as though the adventure begins
inside the picture.

You may use imagination, but imagination must grow
from things visible in the picture.

For example:

If there is a dog, the dog can become an adventure
friend.

If there is a book, the book can become magical.

If there is a ball, the ball can start a playful
adventure.

If there is a garden, the garden can become an
interesting place to explore.

DO NOT create a completely unrelated story.

DO NOT invent a completely different setting.

==================================================
CHILD SAFETY
==================================================

This story is for children.

Do not include:

- violence
- fighting
- weapons
- blood
- death
- murder
- horror
- frightening scenes
- drugs
- sexual content
- hateful content
- self-harm
- dangerous instructions

The story should be gentle, positive and reassuring.

==================================================
STORY STRUCTURE
==================================================

Write exactly {settings["sentences"]} complete sentences.

The story should have:

1. A beginning connected to the picture.
2. A small adventure or discovery.
3. A playful middle.
4. A satisfying ending.

The FINAL sentence must clearly finish the story.

==================================================
OUTPUT RULES
==================================================

Output ONLY the story.

Do not include:

- a title
- bullet points
- numbering
- "Story:"
- explanations
- comments about the picture
- comments about being an AI
- unfinished sentences

Do not repeat sentences.

Do not repeat the same event.

Write exactly {settings["sentences"]} sentences.

Now write the story.
"""


# ============================================================
# SENTENCE PROCESSING
# ============================================================

def split_sentences(text):

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return []

    return [
        item.strip()
        for item in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if item.strip()
    ]


def normalize_sentence(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def clean_story(text):

    text = text.strip()

    # Remove common model prefixes.

    text = re.sub(
        r"^\s*(story|answer)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^\s*here is (your|the) story\s*:?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    sentences = split_sentences(
        text
    )

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

        seen.add(
            normalized
        )

        result.append(
            sentence
        )

    return result


# ============================================================
# CHILD SAFETY FILTER
# ============================================================

def contains_unsafe_content(text):

    unsafe_words = [
        "murder",
        "suicide",
        "porn",
        "sexual",
        "weapon",
        "weapons",
        "gun",
        "guns",
        "blood",
        "kill",
        "killed",
        "shoot",
        "shooting",
        "cocaine",
        "heroin",
        "terrorist",
    ]

    lower = text.lower()

    return any(
        word in lower
        for word in unsafe_words
    )


# ============================================================
# GENERATE STORY USING HF
# ============================================================

def generate_story(
    image,
    age_group,
    story_style,
):

    token = get_hf_token()

    client = get_hf_client(
        token
    )

    image_data_url = image_to_data_url(
        image
    )

    prompt = create_story_prompt(
        age_group,
        story_style,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    settings = get_age_settings(
        age_group
    )

    try:

        response = (
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=settings[
                    "max_tokens"
                ],
                temperature=0.7,
                top_p=0.9,
            )
        )

    except Exception as error:

        raise RuntimeError(
            "Hugging Face inference failed: "
            f"{type(error).__name__}: {error}"
        ) from error

    try:

        story = (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as error:

        raise RuntimeError(
            "Hugging Face returned an unexpected "
            "response format."
        ) from error

    if not story:

        raise RuntimeError(
            "The model returned an empty story."
        )

    sentences = clean_story(
        story
    )

    # --------------------------------------------------------
    # Remove accidental excess sentences.
    # --------------------------------------------------------

    required = settings[
        "sentences"
    ]

    sentences = sentences[
        :required
    ]

    final_story = " ".join(
        sentences
    )

    # --------------------------------------------------------
    # Safety check.
    # --------------------------------------------------------

    if contains_unsafe_content(
        final_story
    ):

        raise RuntimeError(
            "The generated story did not pass "
            "the basic child-safety filter."
        )

    # --------------------------------------------------------
    # Completeness check.
    # --------------------------------------------------------

    if len(sentences) < required:

        raise RuntimeError(
            "The model returned an incomplete story "
            f"({len(sentences)} sentences instead of "
            f"{required})."
        )

    return final_story


# ============================================================
# FALLBACK
# ============================================================

def fallback_story(age_group):

    if age_group == "3–5":

        return (
            "The picture looked like the beginning "
            "of a lovely little adventure. "
            "Everyone noticed something interesting "
            "in the scene. "
            "They explored together with happy smiles. "
            "Soon they discovered a wonderful surprise. "
            "Everyone finished the adventure feeling happy."
        )

    if age_group == "6–7":

        return (
            "The picture looked like the beginning "
            "of a wonderful adventure. "
            "Everyone noticed something interesting "
            "in the scene. "
            "They decided to explore together. "
            "Their little adventure became more exciting "
            "as they looked around. "
            "Soon they discovered a playful surprise. "
            "Everyone laughed and shared the moment. "
            "The adventure ended with happy smiles."
        )

    return (
        "The picture looked like the beginning "
        "of an unexpected adventure. "
        "Everyone noticed several interesting details "
        "in the scene. "
        "They decided to explore together and follow "
        "their curiosity. "
        "Their imagination turned the ordinary moment "
        "into something special. "
        "Along the way, they discovered a delightful "
        "surprise. "
        "Everyone shared the moment with smiles "
        "and laughter. "
        "The adventure gave them a wonderful memory "
        "to remember. "
        "At the end, everyone felt happy about "
        "the adventure they had shared."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    add_css()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        '<div class="main-title">'
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
    # AGE
    # --------------------------------------------------------

    st.subheader(
        "👧 1. Choose an age"
    )

    age_group = st.radio(
        "Age group",
        [
            "3–5",
            "6–7",
            "8–10",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    # --------------------------------------------------------
    # STYLE
    # --------------------------------------------------------

    st.subheader(
        "✨ 2. Choose an adventure"
    )

    story_style = st.selectbox(
        "Story style",
        [
            "Magical adventure 🪄",
            "Animal adventure 🐶",
            "Funny adventure 😂",
            "Space adventure 🚀",
            "Nature adventure 🌳",
        ],
        label_visibility="collapsed",
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    st.subheader(
        "📸 3. Choose a picture"
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
            <div class="tip-box">

            <h3>💡 Try a picture of:</h3>

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
    # IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image = prepare_image(
            image
        )

    except Exception as error:

        st.error(
            "I couldn't open this picture."
        )

        st.exception(
            error
        )

        return

    st.image(
        image,
        caption="Your picture",
        width="stretch",
    )

    # --------------------------------------------------------
    # GENERATE BUTTON
    # --------------------------------------------------------

    if st.button(
        "✨ Make My Story ✨",
        type="primary",
        width="stretch",
    ):

        st.session_state.pop(
            "story",
            None,
        )

        try:

            with st.spinner(
                "🪄 Looking at your picture "
                "and writing your story..."
            ):

                story = generate_story(
                    image=image,
                    age_group=age_group,
                    story_style=story_style,
                )

            st.session_state[
                "story"
            ] = story

            st.success(
                "🎉 Your complete story is ready!"
            )

        except Exception as error:

            st.error(
                "I couldn't make the story right now."
            )

            st.exception(
                error
            )

            st.info(
                "If this happens again, copy the "
                "error shown above and send it to me."
            )

            return

    # --------------------------------------------------------
    # DISPLAY STORY
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
        # AUDIO PLACEHOLDER
        # ----------------------------------------------------

        st.info(
            "🔊 Audio is temporarily disabled while "
            "we make the picture-to-story part reliable."
        )

        # ----------------------------------------------------
        # NEW STORY
        # ----------------------------------------------------

        if st.button(
            "🌟 Make Another Story",
            width="stretch",
        ):

            st.session_state.clear()

            st.rerun()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
