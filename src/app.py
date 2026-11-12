
from pathlib import Path

import streamlit as st
from search_engine import SearchEngine


# Project paths
ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT / "data" / "videos" / "lecture.mp4"


st.set_page_config(
    page_title="MomentMap",
    page_icon="🎬",
    layout="wide"
)


# Load the AI model only once and reuse it.
@st.cache_resource
def load_search_engine():
    return SearchEngine()


def format_time(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    return f"{minutes:02d}:{remaining_seconds:02d}"


# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = []

if "selected_start" not in st.session_state:
    st.session_state.selected_start = 0


# Interface
st.title("🎬 MomentMap")
st.subheader("Find any moment in your video using natural language.")

st.divider()

st.subheader("Your video")

if VIDEO_PATH.exists():
    st.video(
        str(VIDEO_PATH),
        start_time=st.session_state.selected_start
    )
else:
    st.error("lecture.mp4 was not found in data/videos.")

st.divider()

st.subheader("Search inside your video")

query = st.text_input(
    "What moment are you looking for?",
    placeholder="e.g. A professor explaining gradient descent"
)

if st.button("🔍 Search", type="primary"):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching your video..."):
            try:
                engine = load_search_engine()
                st.session_state.results = engine.search(query)
                st.session_state.selected_start = 0
                st.session_state.last_query = query

            except Exception as error:
                st.error(f"Search failed: {error}")

# Display results
if st.session_state.results:

    st.divider()

    st.subheader(
        f"Results for: {st.session_state.get('last_query', '')}"
    )

    for index, result in enumerate(st.session_state.results):

        start = result["start"]
        end = result["end"]
        score = result["score"]

        with st.container(border=True):

            st.markdown(
                f"### Result {index + 1}"
            )

            st.write(
                f"**Time:** {format_time(start)} – "
                f"{format_time(end)}"
            )

            st.write(
                f"**Similarity:** {score:.3f}"
            )

            if st.button(
                f"▶ Play from {format_time(start)}",
                key=f"play_{index}"
            ):
                st.session_state.selected_start = int(start)
                st.rerun()

elif st.session_state.get("last_query"):
    st.info("No matching moments found. Try another search.")
