from pathlib import Path
import csv

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    # Load saved frame embeddings.
    embeddings = np.load("data/frame_embeddings.npy")

    # Read the original frame timestamps.
    metadata_files = sorted(
        Path("data/frames").rglob("frames.csv")
    )

    frames = []

    for metadata_file in metadata_files:
        with metadata_file.open(
            newline="", encoding="utf-8"
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                frames.append({
                    "frame": str(
                        metadata_file.parent / row["frame"]
                    ),
                    "timestamp": float(
                        row["timestamp_seconds"]
                    ),
                })

    if len(frames) != len(embeddings):
        raise ValueError(
            f"Found {len(frames)} frames but "
            f"{len(embeddings)} embeddings."
        )

    # Enter a natural-language search query.
    query = input("Search your video: ")
    
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        text_output = model.text_model(**inputs)

        query_features = model.text_projection(
            text_output.pooler_output
        )

        query_features = query_features / query_features.norm(
            dim=-1, keepdim=True
        )

    query_embedding = query_features.cpu().numpy()[0]

    # Calculate cosine similarity against every frame.
    similarities = embeddings @ query_embedding

    # Retrieve the five most relevant frames.
        # Group consecutive matching frames into segments.
    threshold = 0.3
    max_gap_seconds = 10.0

    # Find frames with sufficiently high similarity.
    matching_indices = np.where(similarities >= threshold)[0]

    if len(matching_indices) == 0:
        print("No matching moments found.")
        return

    # Sort by timestamp.
    matching_indices = sorted(
        matching_indices,
        key=lambda i: frames[i]["timestamp"]
    )

    segments = []
    current_segment = [matching_indices[0]]

    for index in matching_indices[1:]:
        previous_index = current_segment[-1]

        time_gap = (
            frames[index]["timestamp"]
            - frames[previous_index]["timestamp"]
        )

        if time_gap <= max_gap_seconds:
            current_segment.append(index)
        else:
            segments.append(current_segment)
            current_segment = [index]

    segments.append(current_segment)

    # Score and rank each segment.
    results = []

    for segment in segments:
        start_time = frames[segment[0]]["timestamp"]
        end_time = frames[segment[-1]]["timestamp"]

        segment_score = float(
            np.mean(similarities[segment])
        )

        results.append({
            "start": start_time,
            "end": end_time,
            "score": segment_score,
        })

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    print("\nMost relevant video segments:\n")

    for result in results[:5]:
        print(
            f"{result['start']:.1f}s - "
            f"{result['end']:.1f}s | "
            f"Similarity: {result['score']:.3f}"
        )


if __name__ == "__main__":
    main()