import gc
import re

import streamlit as st
import torch

from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"

MAX_IMAGE_SIZE = 1024


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="My Story Maker",
    page_icon="🌈",
    layout="centered",
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
            background-color: #FFF8E7;
            border: 2px solid #FFD86B;
            border-radius: 20px;
            padding: 25px;
            font-size: 20px;
            line-height: 1.8;
        }

        .tip-box {
            background-color: #F1F7FF;
            border-radius: 18px;
            padding: 20px;
            margin-top: 15px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    width, height = image.size

    largest_side = max(
        width,
        height,
    )

    if largest_side > MAX_IMAGE_SIZE:

        scale = (
            MAX_IMAGE_SIZE
            / largest_side
        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        image = image.resize(
            (
                new_width,
                new_height,
            ),
            Image.Resampling.LANCZOS,
        )

    return image


# ============================================================
# SENTENCE HANDLING
# ============================================================

def split_sentences(text):

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        p.strip()
        for p in parts
        if p.strip()
    ]


def clean_story(text):

    # Remove common unwanted prefixes.

    text = re.sub(
        r"^\s*(story|answer)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^\s*here is.*?:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove markdown headings.

    text = re.sub(
        r"^#+\s*.*?\n",
        "",
        text,
    )

    sentences = split_sentences(
        text
    )

    # Remove exact duplicate sentences.

    seen = set()
    clean = []

    for sentence in sentences:

        key = re.sub(
            r"[^a-z0-9 ]",
            "",
            sentence.lower(),
        )

        key = re.sub(
            r"\s+",
            " ",
            key,
        ).strip()

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)

        clean.append(
            sentence
        )

    return clean


# ============================================================
# AGE SETTINGS
# ============================================================

def get_age_settings(age_group):

    if age_group == "3–5":

        return {
            "sentence_count": 5,
            "tokens": 180,
            "language": (
                "Use very simple words and short sentences."
            ),
        }

    if age_group == "6–7":

        return {
            "sentence_count": 7,
            "tokens": 230,
            "language": (
                "Use simple vocabulary and short, "
                "easy-to-follow sentences."
            ),
        }

    return {
        "sentence_count": 9,
        "tokens": 300,
        "language": (
            "Use vocabulary suitable for children "
            "aged 8 to 10."
        ),
    }


# ============================================================
# STORY PROMPT
# ============================================================

def make_prompt(
    age_group,
    story_style,
):

    settings = get_age_settings(
        age_group
    )

    return f"""
You are a friendly children's storyteller.

Look carefully at the picture.

Write a COMPLETE children's story based on what you
can actually see in the picture.

The child is {age_group} years old.

Story style:
{story_style}

{settings["language"]}

IMPORTANT:
The story MUST be connected to the picture.

Use several visible details from the picture.

For example, if you can see:
- a child, include the child;
- a dog, include the dog;
- books, include the books;
- a ball, include the ball;
- a bicycle, include the bicycle;
- a garden, use the garden;
- particular colors, you may mention them.

Do not invent a completely different scene.

You may add gentle imagination to visible objects.

For example:
A visible book can become a magical book.
A visible dog can become an adventure friend.
A visible ball can lead to a playful adventure.

But the story must still clearly relate to the picture.

SAFETY:
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

Make the story warm, playful and reassuring.

WRITING REQUIREMENTS:

Write exactly {settings["sentence_count"]} complete sentences.

Every sentence must be complete.

Do not repeat sentences.

Do not repeat the same event.

Do not create a title.

Do not use bullet points.

Do not explain your answer.

Do not say "Here is your story".

Do not mention AI.

The final sentence must provide a happy and complete ending.

OUTPUT ONLY THE STORY.

Start now.
"""


# ============================================================
# GENERATE STORY
# ============================================================

def generate_story(
    image,
    age_group,
    story_style,
):

    processor, model = load_model()

    prompt = make_prompt(
        age_group,
        story_style,
    )

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

    # --------------------------------------------------------
    # Let the processor create the complete multimodal input.
    # This follows the current SmolVLM usage pattern.
    # --------------------------------------------------------

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=get_age_settings(
                age_group
            )["tokens"],
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
        )

    # --------------------------------------------------------
    # Remove prompt tokens.
    # --------------------------------------------------------

    input_length = (
        inputs["input_ids"].shape[-1]
    )

    generated_tokens = output[
        0,
        input_length:
    ]

    story = processor.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    story = clean_story(
        story
    )

    settings = get_age_settings(
        age_group
    )

    # --------------------------------------------------------
    # Keep the requested number of sentences.
    # --------------------------------------------------------

    story = story[
        :settings["sentence_count"]
    ]

    # --------------------------------------------------------
    # Join the final story.
    # --------------------------------------------------------

    final_story = " ".join(
        story
    ).strip()

    # --------------------------------------------------------
    # Cleanup temporary tensors.
    # --------------------------------------------------------

    del inputs
    del output

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_story


# ============================================================
# FALLBACK
# ============================================================

def fallback_story(age_group):

    if age_group == "3–5":

        return (
            "The picture was the beginning of a "
            "wonderful little adventure. "
            "Everyone looked around and noticed "
            "the interesting things nearby. "
            "They explored together with big smiles. "
            "Soon they discovered something fun. "
            "Everyone went home feeling happy."
        )

    if age_group == "6–7":

        return (
            "The picture looked like the beginning "
            "of a wonderful adventure. "
            "The characters noticed something "
            "interesting nearby. "
            "They decided to explore together. "
            "Their little adventure became more "
            "exciting with every step. "
            "They laughed and shared ideas. "
            "Soon they discovered a lovely surprise. "
            "Everyone enjoyed the adventure. "
            "At the end, they went home with happy memories."
        )

    return (
        "The picture looked like the beginning "
        "of an unexpected adventure. "
        "The characters noticed several interesting "
        "details around them. "
        "They decided to explore the scene together. "
        "As they looked around, their imagination "
        "turned the ordinary moment into something special. "
        "They followed their curiosity and discovered "
        "a delightful surprise. "
        "Everyone shared the moment with smiles and laughter. "
        "The adventure gave them a wonderful memory "
        "to talk about later. "
        "By the end of the day, everyone felt happy "
        "about the adventure they had shared."
    )


# ============================================================
# STORY VALIDATION
# ============================================================

def validate_story(
    story,
    age_group,
):

    if not story:
        return False

    sentences = clean_story(
        story
    )

    required = get_age_settings(
        age_group
    )["sentence_count"]

    # We require the complete requested story.

    if len(sentences) < required:

        return False

    # Detect repeated sentences.

    normalized = [
        re.sub(
            r"[^a-z0-9 ]",
            "",
            s.lower(),
        )
        for s in sentences
    ]

    if len(normalized) != len(
        set(normalized)
    ):

        return False

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    add_css()

    # --------------------------------------------------------
    # Header
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
    # Age
    # --------------------------------------------------------

    st.subheader(
        "👧 1. How old is the storyteller?"
    )

    age_group = st.radio(
        "Age",
        [
            "3–5",
            "6–7",
            "8–10",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    # --------------------------------------------------------
    # Style
    # --------------------------------------------------------

    st.subheader(
        "✨ 2. Choose a story style"
    )

    story_style = st.selectbox(
        "Style",
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
    # Image
    # --------------------------------------------------------

    st.subheader(
        "📸 3. Choose a picture"
    )

    uploaded_file = st.file_uploader(
        "Upload your picture",
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

            🧸 Your favorite toy<br>
            🐶 Your pet<br>
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
    # Load image
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
    # Generate
    # --------------------------------------------------------

    if st.button(
        "✨ Make My Story ✨",
        type="primary",
        width="stretch",
    ):

        # Remove old result.

        st.session_state.pop(
            "story",
            None,
        )

        try:

            with st.spinner(
                "👀 Looking at your picture..."
            ):

                story = generate_story(
                    image=image,
                    age_group=age_group,
                    story_style=story_style,
                )

            if not validate_story(
                story,
                age_group,
            ):

                st.warning(
                    "The story was incomplete, "
                    "so I made a new simple story."
                )

                story = fallback_story(
                    age_group
                )

            st.session_state[
                "story"
            ] = story

        except Exception as error:

            st.error(
                "The story generator encountered "
                "an error."
            )

            # VERY IMPORTANT during development:
            # Streamlit Cloud normally hides details.
            # This shows the actual exception so that
            # we can diagnose the next issue.

            st.exception(
                error
            )

            st.info(
                "If you send me the error shown above, "
                "I can identify the exact problem."
            )

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

        st.success(
            "🎉 Your story is complete!"
        )

        # ----------------------------------------------------
        # TTS intentionally disabled for this diagnostic
        # version.
        # ----------------------------------------------------

        st.info(
            "🔊 Audio will be added after story generation "
            "is working reliably."
        )


# ============================================================
# START APP
# ============================================================

if __name__ == "__main__":

    main()
