
# MomentMap 

**AI-Powered Semantic Video Search**

MomentMap is a personal AI/ML learning project that allows users
to search through long videos using natural-language queries.

Instead of manually scrubbing through hours of footage, users
can describe a moment and retrieve relevant timestamps.

## Features

- Extract frames from videos using OpenCV.
- Generate visual embeddings using OpenAI CLIP.
- Search video frames using natural-language queries.
- Rank relevant moments using cosine similarity.
- Group nearby matching frames into video segments.
- View search results through a Streamlit interface.
- Jump directly to selected timestamps in the video.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- OpenAI CLIP (ViT-B/32)
- OpenCV
- NumPy
- Streamlit

## Architecture

Video
  ↓
Frame Extraction (OpenCV)
  ↓
Image Embeddings (CLIP)
  ↓
Embedding Storage (NumPy)
  ↓
Text Query
  ↓
Text Embedding (CLIP)
  ↓
Cosine Similarity Search
  ↓
Timestamp-based Segment Retrieval
  ↓
Streamlit Interface

## Setup

Clone the repository:

    git clone https://github.com/Raghunandan-Kotary/MemoryMap.git

    cd MemoryMap

Create a virtual environment:

    python -m venv .venv

Activate it on Windows:

    .venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install -r requirements.txt

## Running the Application

Place your video and generated frame data in the
appropriate data directories.

After generating frame embeddings, launch the application:

    streamlit run src/app.py

## How It Works

MomentMap uses CLIP to map images and natural-language
descriptions into a shared embedding space.

The similarity between a text query and stored frame embeddings
is calculated using cosine similarity.

Matching frames are grouped into timestamped segments,
which users can navigate to through the Streamlit interface.

## Project Status

Functional personal MVP.

## Future Improvements

- Better temporal localization.
- More accurate semantic retrieval.
- Support for multiple videos.
- Improved result previews and thumbnails.
- More advanced video understanding.

## Author

Raghunandan Kotary
