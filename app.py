"""
=====================================================================================
 app.py — "Story Time!" — An AI Storytelling App for Kids (ages 3-10)
 ------------------------------------------------------------------------------------
 Course : ISOM5240 — Individual Assignment
 Task   : Storytelling Application using Hugging Face Pipelines

 WHAT THIS APP DOES, STEP BY STEP:
   1. The user uploads a photo, or takes one with their camera.
   2. STEP 1 - CAPTION: An AI model looks at the photo and writes a short
      caption, e.g. "a dog playing with a red ball".
   3. STEP 2 - STORY: A second AI model turns that caption into a short
      children's story.
   4. STEP 3 - AUDIO: A third AI model reads the story out loud and turns it
      into an audio file.
   5. The story and audio are shown on screen for the child to read/listen to.

 HOW THIS FILE IS ORGANIZED (read top to bottom):
   - First, some settings (CONSTANTS) that are easy to find and change.
   - Then, three small "load the AI model" functions (one per step above).
   - Then, three small "do the work" functions (one per step above).
   - Then, the Streamlit screens (what the user actually sees and clicks).
   - Finally, main() at the very bottom, which is what actually runs.

 A NOTE FOR DEBUGGING:
   If something breaks, the error message at the bottom of the page will
   usually point to one specific function below. Since each function only
   does ONE small job (get a caption, OR write a story, OR make audio), you
   can test that one function by itself to find the problem.
=====================================================================================
"""

import io

import numpy as np
import soundfile as sf
import streamlit as st
from PIL import Image
from transformers import pipeline


# -------------------------------------------------------------------------------------
# PAGE SETUP
# This has to be the very first Streamlit command in the whole file.
# -------------------------------------------------------------------------------------
st.set_page_config(page_title="Story Time!", page_icon="🧸", layout="centered")


# -------------------------------------------------------------------------------------
# SETTINGS (CONSTANTS)
# All the "things you might want to change" are collected here at the top,
# instead of being scattered through the code, so they're easy to find.
# -------------------------------------------------------------------------------------

# Which Hugging Face model does each job:
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"  # looks at the photo
STORY_MODEL_NAME = "google/flan-t5-small"                     # writes the story
AUDIO_MODEL_NAME = "facebook/mms-tts-eng"                      # reads it aloud

# The two story-length choices the child can pick between.
# min_new_tokens / max_new_tokens tell the AI model roughly how much text to
# write (this is a bit more than the word count, since the model thinks in
# small pieces of words called "tokens", not whole words).
QUICK_STORY = {
    "label": "🐣 Quick Story  (~1 minute)",
    "min_words": 50,
    "max_words": 100,
    "min_new_tokens": 60,
    "max_new_tokens": 180,
}
BIG_ADVENTURE = {
    "label": "🐉 Big Adventure  (~2 minutes)",
    "min_words": 200,
    "max_words": 260,
    "min_new_tokens": 240,
    "max_new_tokens": 420,
}
STORY_LENGTH_CHOICES = [QUICK_STORY, BIG_ADVENTURE]


# -------------------------------------------------------------------------------------
# STEP 0: MAKE THE APP LOOK COLOURFUL AND FUN
# This just adds some CSS (styling rules) so the app looks bright and playful
# for kids. It doesn't affect how the app works — if you want to see the
# plainer, unstyled version, just skip calling this function in main().
# -------------------------------------------------------------------------------------
def make_the_app_colourful():
    st.markdown(
        """
        <style>
        /* Use a playful, rounded font everywhere */
        @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;700;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Baloo 2', sans-serif !important;
            font-size: 18px;
        }

        /* Soft rainbow background instead of plain white */
        .stApp {
            background: linear-gradient(160deg, #FFF6E5, #FFE3EC, #E6F4FF, #EAFBEA);
        }

        /* Big, colourful title */
        h1 {
            text-align: center;
            color: #FF6F91;
        }

        /* Big rounded buttons that are easy for little fingers to tap */
        .stButton > button, .stDownloadButton > button {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            color: white !important;
            background: linear-gradient(135deg, #FF9A76, #FF6F91) !important;
            border: none !important;
            border-radius: 30px !important;
            padding: 0.8em 1em !important;
            width: 100%;
        }

        /* Rounded picture frame around the uploaded photo */
        div[data-testid="stImage"] img {
            border-radius: 20px;
            border: 6px solid white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------------------------
# LOAD THE THREE AI MODELS
#
# @st.cache_resource means: "only actually load this model ONE time, then
# reuse it." Without this, Streamlit would re-download and reload the model
# every single time the user clicks a button, which would be very slow.
# -------------------------------------------------------------------------------------

@st.cache_resource
def load_caption_model():
    """Loads the AI model that looks at a photo and describes it in words."""
    return pipeline("image-to-text", model=CAPTION_MODEL_NAME)


@st.cache_resource
def load_story_model():
    """Loads the AI model that writes a story from a description."""
    return pipeline("text2text-generation", model=STORY_MODEL_NAME)


@st.cache_resource
def load_audio_model():
    """Loads the AI model that turns text into spoken audio."""
    return pipeline("text-to-speech", model=AUDIO_MODEL_NAME)


# -------------------------------------------------------------------------------------
# STEP 1: LOOK AT THE PHOTO AND WRITE A CAPTION
# -------------------------------------------------------------------------------------
def get_caption_from_photo(photo):
    """
    Give the photo to the caption model and get back a short description.

    Example: give it a photo of a dog, and it might return
    "a dog running in a park".
    """
    caption_model = load_caption_model()

    # The model gives back a list with one result in it, like this:
    #   [{"generated_text": "a dog running in a park"}]
    # so we grab item [0], then the "generated_text" part of it.
    result = caption_model(photo)
    caption = result[0]["generated_text"]

    return caption.strip()  # .strip() removes extra spaces at the start/end


# -------------------------------------------------------------------------------------
# STEP 2: TURN THE CAPTION INTO A STORY
# -------------------------------------------------------------------------------------
def write_story_from_caption(caption, story_settings):
    """
    Give the caption to the story model and get back a children's story.

    `story_settings` is one of QUICK_STORY or BIG_ADVENTURE from the
    settings section above — it tells this function how long the story
    should be.
    """
    story_model = load_story_model()

    # This is the instruction we give the AI model. Changing this sentence
    # changes the *style* of story you get back — try editing it!
    instructions = (
        "Write a fun, imaginative, and gentle story for young children "
        "aged 3 to 10 years old. No scary or violent content. The story "
        f"should be between {story_settings['min_words']} and "
        f"{story_settings['max_words']} words long. Base the story on this "
        f"scene: '{caption}'. Give the main character a name and a happy ending."
    )

    # We may need to try a couple of times if the story comes back too short.
    story_text = ""
    number_of_tries = 0
    while number_of_tries < 3:
        result = story_model(
            instructions,
            max_new_tokens=story_settings["max_new_tokens"],
            min_new_tokens=story_settings["min_new_tokens"],
            do_sample=True,       # lets the model be a little creative/random
            temperature=0.9,      # higher = more creative, lower = more plain
            top_p=0.95,
            # These next two settings stop the AI model from getting stuck
            # and repeating the same sentence over and over, which can
            # happen when we ask it to write more than it naturally would.
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
        )
        story_text = result[0]["generated_text"].strip()

        word_count = len(story_text.split())
        if word_count >= story_settings["min_words"]:
            break  # long enough, stop trying

        number_of_tries += 1

    # Safety net: if a sentence somehow still repeats, remove the repeat.
    story_text = remove_repeated_sentences(story_text)

    # If the story ended up too long, cut it down to the word limit.
    words = story_text.split()
    if len(words) > story_settings["max_words"]:
        words = words[: story_settings["max_words"]]
        story_text = " ".join(words)
        if not story_text.endswith("."):
            story_text += "."

    return story_text


def clean_up_for_comparing(sentence):
    """
    Turns a sentence into a plain, simple form just for COMPARING two
    sentences to each other (we still keep the original wording when we
    build the final story — this cleaned-up version is only used to decide
    "are these two sentences basically the same?").
    """
    text = sentence.strip().lower()
    text = text.rstrip(".!?")  # ignore a trailing period/!/? when comparing
    return text


def remove_repeated_sentences(story_text):
    """
    Removes a sentence if it's an exact repeat of the sentence right before it.

    Small AI models can sometimes get "stuck" and repeat the same sentence
    many times in a row instead of writing something new. This function
    cleans that up as a safety net, just in case it happens.
    """
    sentences = story_text.split(". ")  # split the story into sentences

    cleaned_sentences = []
    previous_sentence = ""
    for sentence in sentences:
        if clean_up_for_comparing(sentence) != clean_up_for_comparing(previous_sentence):
            cleaned_sentences.append(sentence)
        previous_sentence = sentence

    return ". ".join(cleaned_sentences)


# -------------------------------------------------------------------------------------
# STEP 3: TURN THE STORY INTO AUDIO
# -------------------------------------------------------------------------------------
def make_audio_from_story(story_text):
    """
    Give the story text to the audio model and get back a playable audio file.

    Returns the audio as "WAV bytes" — just a chunk of data that Streamlit's
    audio player and download button both know how to use directly.
    """
    audio_model = load_audio_model()

    # The model gives back a dictionary like:
    #   {"audio": <numbers representing sound>, "sampling_rate": 16000}
    result = audio_model(story_text)
    audio_numbers = result["audio"]
    sample_rate = result["sampling_rate"]

    # np.squeeze removes any extra "wrapper" layer around the audio numbers,
    # so we end up with a plain 1D list of numbers that soundfile understands.
    audio_numbers = np.squeeze(np.asarray(audio_numbers))

    # Write those numbers into a real WAV audio file, but keep it in memory
    # (io.BytesIO) instead of saving it to disk, since we don't need a file
    # on disk — we just need to hand the bytes to Streamlit.
    audio_file_in_memory = io.BytesIO()
    sf.write(audio_file_in_memory, audio_numbers, sample_rate, format="WAV")

    return audio_file_in_memory.getvalue()  # .getvalue() gives us the raw bytes


# -------------------------------------------------------------------------------------
# SCREEN 1: ASK FOR A PHOTO AND STORY LENGTH
#
# The app shows either THIS screen, or SCREEN 2 below — never both at once —
# so the user doesn't have to scroll through everything on one giant page.
# -------------------------------------------------------------------------------------
def show_photo_input_screen():
    st.title("🧸✨ Story Time! ✨🧸")
    st.write("Take or upload a picture, and watch it turn into a story you can listen to!")
    st.markdown("---")

    # `uploader_generation` is a little counter we use to reset the photo
    # widgets below. Streamlit remembers old widget values by their "key",
    # so to force a *blank* upload box again (e.g. after making a story),
    # we just give the widget a new key by counting up. You don't need to
    # touch this — it's only read here and changed in start_a_new_story().
    if "uploader_generation" not in st.session_state:
        st.session_state.uploader_generation = 0
    widget_key_number = st.session_state.uploader_generation

    how_to_add_photo = st.radio(
        "How do you want to add a picture?",
        ["📁 Upload a Photo", "🤳 Take a Photo"],
        horizontal=True,
    )

    if how_to_add_photo == "📁 Upload a Photo":
        uploaded_file = st.file_uploader(
            "Choose a picture:",
            type=["png", "jpg", "jpeg"],
            key=f"uploader_{widget_key_number}",
        )
    else:
        uploaded_file = st.camera_input(
            "Take a picture:",
            key=f"camera_{widget_key_number}",
        )

    if uploaded_file is None:
        st.info("👆 Add a picture above to get started!")
        return  # nothing more to do until the user adds a photo

    # Turn the uploaded file into an image we can show and pass to the AI model.
    photo = Image.open(uploaded_file).convert("RGB")

    # Show a small, centered preview of the photo (not stretched full-width).
    left_space, middle, right_space = st.columns([1, 2, 1])
    with middle:
        st.image(photo, caption="Your picture", width=260)

    # Let the user pick a story length. st.radio shows the "label" text for
    # each choice, and gives back the matching dictionary from our list.
    chosen_length = st.radio(
        "How long should the story be?",
        STORY_LENGTH_CHOICES,
        format_func=lambda choice: choice["label"],
        horizontal=True,
    )

    if st.button("✨ Make My Story! ✨"):
        make_the_story_and_save_it(photo, chosen_length)
        st.rerun()  # redraw the page — this time it will show Screen 2


def make_the_story_and_save_it(photo, chosen_length):
    """
    Runs all three AI steps, then saves the results into st.session_state.

    We SAVE the results instead of just displaying them here so that they
    stick around even after the page reruns later (for example, when the
    user clicks the "Save the Story" download button, which also causes
    Streamlit to rerun the whole script from the top).
    """
    try:
        with st.spinner("👀 Looking at your picture..."):
            caption = get_caption_from_photo(photo)

        with st.spinner("✍️ Writing your story..."):
            story = write_story_from_caption(caption, chosen_length)

        with st.spinner("🔊 Recording the story..."):
            audio_bytes = make_audio_from_story(story)

        # Turn the photo into plain bytes too, so we can show it again on
        # Screen 2 without needing the original upload widget.
        photo_bytes_holder = io.BytesIO()
        photo.save(photo_bytes_holder, format="PNG")

        # Save everything into session_state using simple, separate variables
        # (rather than one big combined structure) so each piece is easy to
        # inspect on its own — for example, in a debug helper you could just
        # write: st.write(st.session_state.story_text)
        st.session_state.story_caption = caption
        st.session_state.story_text = story
        st.session_state.story_audio_bytes = audio_bytes
        st.session_state.story_photo_bytes = photo_bytes_holder.getvalue()

    except Exception as error:
        # Show a friendly message for the child/parent...
        st.error("Oops! Something went wrong while making your story. Please try again.")
        # ...but also show the real technical error, which is what YOU need
        # when debugging. This is exactly the message you'd paste to me.
        st.exception(error)


# -------------------------------------------------------------------------------------
# SCREEN 2: SHOW THE FINISHED STORY
# -------------------------------------------------------------------------------------
def show_finished_story_screen():
    st.title("🧸✨ Your Story! ✨🧸")

    left_space, middle, right_space = st.columns([1, 2, 1])
    with middle:
        st.image(st.session_state.story_photo_bytes, width=200)

    st.write(st.session_state.story_text)
    word_count = len(st.session_state.story_text.split())
    st.caption(f"Word count: {word_count} words")

    # autoplay=True starts the audio playing by itself — helpful for a child
    # who can't yet read the word "play".
    st.audio(st.session_state.story_audio_bytes, format="audio/wav", autoplay=True)

    st.download_button(
        "⬇️ 🎵 Save the Story",
        data=st.session_state.story_audio_bytes,
        file_name="my_story.wav",
        mime="audio/wav",
    )

    with st.expander("🔍 See what the AI noticed in your picture"):
        st.write(f"Image caption: *{st.session_state.story_caption}*")

    if st.button("🔁 Make Another Story!"):
        start_a_new_story()


def start_a_new_story():
    """Clears the saved story and picture so the app goes back to Screen 1."""
    st.session_state.story_text = None
    # Bump the widget-key counter (see the comment in show_photo_input_screen)
    # so the photo upload/camera boxes come back empty for the next photo.
    st.session_state.uploader_generation = st.session_state.get("uploader_generation", 0) + 1
    st.rerun()


# -------------------------------------------------------------------------------------
# MAIN: THIS IS WHAT ACTUALLY RUNS WHEN THE APP STARTS
# -------------------------------------------------------------------------------------
def main():
    make_the_app_colourful()

    # "story_text" only exists in session_state once a story has been made.
    # We check it here to decide which of the two screens to show.
    if "story_text" not in st.session_state:
        st.session_state.story_text = None

    if st.session_state.story_text is not None:
        show_finished_story_screen()
    else:
        show_photo_input_screen()


# This line means: "only run main() if this file is being run directly"
# (which is how `streamlit run app.py` works).
if __name__ == "__main__":
    main()
