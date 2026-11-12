
from pathlib import Path
import csv

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"

ROOT = Path(__file__).resolve().parent.parent


class SearchEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading CLIP model...")

        self.model = CLIPModel.from_pretrained(
            MODEL_NAME
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            MODEL_NAME
        )

        self.model.eval()

        # Load saved frame embeddings
        self.embeddings = np.load(
            ROOT / "data" / "frame_embeddings.npy"
        )

        # Load frame timestamps
        metadata_files = sorted(
            (ROOT / "data" / "frames").rglob("frames.csv")
        )

        self.frames = []

        for metadata_file in metadata_files:
            with metadata_file.open(
                newline="", encoding="utf-8"
            ) as f:
                reader = csv.DictReader(f)

                for row in reader:
                    self.frames.append({
                        "frame": str(
                            metadata_file.parent / row["frame"]
                        ),
                        "timestamp": float(
                            row["timestamp_seconds"]
                        )
                    })

        if len(self.frames) != len(self.embeddings):
            raise ValueError(
                "Frame and embedding counts do not match."
            )

        print("Search engine ready!")

    def search(self, query):
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            text_output = self.model.text_model(**inputs)

            features = self.model.text_projection(
                text_output.pooler_output
            )

            features = features / features.norm(
                dim=-1, keepdim=True
            )

        query_embedding = features.cpu().numpy()[0]

        similarities = self.embeddings @ query_embedding

        threshold = 0.3
        max_gap_seconds = 10.0

        matching_indices = np.where(
            similarities >= threshold
        )[0]

        if len(matching_indices) == 0:
            return []

        matching_indices = sorted(
            matching_indices,
            key=lambda i: self.frames[i]["timestamp"]
        )

        segments = []
        current_segment = [matching_indices[0]]

        for index in matching_indices[1:]:
            previous_index = current_segment[-1]

            time_gap = (
                self.frames[index]["timestamp"]
                - self.frames[previous_index]["timestamp"]
            )

            if time_gap <= max_gap_seconds:
                current_segment.append(index)
            else:
                segments.append(current_segment)
                current_segment = [index]

        segments.append(current_segment)

        results = []

        for segment in segments:
            results.append({
                "start": self.frames[segment[0]]["timestamp"],
                "end": self.frames[segment[-1]]["timestamp"],
                "score": float(
                    np.mean(similarities[segment])
                )
            })

        results.sort(
            key=lambda result: result["score"],
            reverse=True
        )

        return results[:5]
