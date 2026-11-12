import argparse
import csv
from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float = 5.0,
) -> None:
    """Extract video frames at regular time intervals."""

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if interval_seconds <= 0:
        raise ValueError("Interval must be greater than zero.")

    output_dir.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0 or total_frames <= 0:
        video.release()
        raise RuntimeError("Could not read video properties.")

    duration = total_frames / fps

    print(f"Video: {video_path.name}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"FPS: {fps:.2f}")

    metadata_path = output_dir / "frames.csv"

    with metadata_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(["frame", "timestamp_seconds"])

        timestamp = 0.0
        frame_count = 0

        while timestamp < duration:

            frame_number = round(timestamp * fps)

            video.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number,
            )

            success, frame = video.read()

            if not success:
                break

            filename = f"frame_{frame_number:08d}.jpg"
            frame_path = output_dir / filename

            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(
                    f"Could not save frame: {frame_path}"
                )

            writer.writerow([filename, timestamp])

            frame_count += 1
            timestamp += interval_seconds

    video.release()

    print(f"Extracted {frame_count} frames.")
    print(f"Frames saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video."
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Path to the input video.",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between sampled frames.",
    )

    args = parser.parse_args()

    output_dir = (
        Path("data")
        / "frames"
        / args.video.stem
    )

    extract_frames(
        args.video,
        output_dir,
        args.interval,
    )


if __name__ == "__main__":
    main()