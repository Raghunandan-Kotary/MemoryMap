from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    frames_dir = Path("data/frames")
    frames = sorted(frames_dir.rglob("*.jpg"))

    if not frames:
        raise FileNotFoundError(
            "No extracted frames found in data/frames"
        )

    embeddings = []

    with torch.no_grad():
        for frame_path in frames:
            image = Image.open(frame_path).convert("RGB")

            inputs = processor(
                images=image,
                return_tensors="pt",
            ).to(device)

            vision_output = model.vision_model(**inputs)

            features = model.visual_projection(
                vision_output.pooler_output
            )

            features = features / features.norm(
                dim=-1, keepdim=True
            )

            embeddings.append(features.cpu().numpy()[0])

    embeddings = np.array(embeddings)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    np.save(data_dir / "frame_embeddings.npy", embeddings)

    print(f"Embedded {len(frames)} frames.")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()