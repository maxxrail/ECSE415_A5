import argparse
import csv
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from deep_sort_realtime.deepsort_tracker import DeepSort


# -----------------------------
# Utils
# -----------------------------
def load_detector(device: str = "cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device = "cpu"

    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    model.to(device)
    model.eval()
    return model, device


def init_deepsort():
    """
    Initialize DeepSort tracker.
    You can tweak these for better tracking, but defaults are already quite good.
    """
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=True,
    )
    return tracker


# -----------------------------
# Main processing
# -----------------------------
def process_frames_to_videos_and_outputs(
    frames_dir: Path,
    input_video_path: Path,
    tracked_video_path: Path,
    tracks_txt_path: Path,
    counts_csv_path: Path,
    fps: float = 14.0,
    device: str = "cuda",
    score_thresh: float = 0.45,
):
    # Load detector + tracker
    model, device = load_detector(device)
    tracker = init_deepsort()

    image_paths = sorted(frames_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg files found in {frames_dir}")

    # Read first frame to get size
    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {image_paths[0]}")

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Video writers
    input_writer = cv2.VideoWriter(
        str(input_video_path), fourcc, fps, (width, height)
    )
    tracked_writer = cv2.VideoWriter(
        str(tracked_video_path), fourcc, fps, (width, height)
    )

    # Outputs
    count_rows = []          # for CSV
    track_lines = []         # for MOT-style txt

    frame_idx = 0

    with torch.no_grad():
        for img_path in image_paths:
            frame_idx += 1

            frame_bgr = cv2.imread(str(img_path))
            if frame_bgr is None:
                print(f"WARNING: could not read {img_path}, skipping.")
                continue

            # Write raw frame to input video
            input_writer.write(frame_bgr)

            # Prepare image for detector (RGB)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor = F.to_tensor(pil_img).to(device)

            outputs = model([tensor])[0]
            boxes = outputs["boxes"].cpu()
            labels = outputs["labels"].cpu()
            scores = outputs["scores"].cpu()

            detections = []
            person_count = 0

            # Build detections for DeepSort
            for box, label, score in zip(boxes, labels, scores):
                if label.item() != 1:  # 1 = "person" in COCO
                    continue
                if score.item() < score_thresh:
                    continue

                x1, y1, x2, y2 = box.tolist()

                # Clamp to frame
                x1 = max(0.0, min(x1, width - 1.0))
                x2 = max(0.0, min(x2, width - 1.0))
                y1 = max(0.0, min(y1, height - 1.0))
                y2 = max(0.0, min(y2, height - 1.0))

                if x2 <= x1 or y2 <= y1:
                    continue

                # IMPORTANT: DeepSort wants [x, y, w, h], NOT [x1, y1, x2, y2]
                w = x2 - x1
                h = y2 - y1

                detections.append(
                    ([x1, y1, w, h], float(score.item()), "person")
                )
                person_count += 1

            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame_bgr)

            # Draw tracks and build MOT lines
            for trk in tracks:
                if not trk.is_confirmed() or trk.time_since_update > 0:
                    continue

                # tlbr = (left, top, right, bottom)
                x1, y1, x2, y2 = map(int, trk.to_ltrb())

                # Clamp
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width - 1))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height - 1))

                bb_left = x1
                bb_top = y1
                bb_width = max(0, x2 - x1)
                bb_height = max(0, y2 - y1)

                track_id = trk.track_id

                # Draw rectangle + ID
                cv2.rectangle(
                    frame_bgr,
                    (bb_left, bb_top),
                    (bb_left + bb_width, bb_top + bb_height),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"ID {track_id}",
                    (bb_left, max(0, bb_top - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                # MOT-style line: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>
                track_lines.append(
                    f"{frame_idx}, {track_id}, {bb_left}, {bb_top}, {bb_width}, {bb_height}\n"
                )

            # Write tracked frame
            tracked_writer.write(frame_bgr)

            # For counting, we stick with detector person_count (not unique IDs)
            count_rows.append({"Number": frame_idx, "count": person_count})
            print(
                f"Frame {frame_idx:4d}: {person_count:2d} people,"
                f" {len(detections)} detections, {len(tracks)} tracks"
            )

    # Release videos
    input_writer.release()
    tracked_writer.release()

    # Write tracks txt
    tracks_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with tracks_txt_path.open("w") as f:
        f.writelines(track_lines)

    # Write counts CSV
    counts_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with counts_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Number", "count"])
        writer.writeheader()
        writer.writerows(count_rows)

    print(f"\nSaved input video to   {input_video_path}")
    print(f"Saved tracked video to {tracked_video_path}")
    print(f"Saved tracks to        {tracks_txt_path}")
    print(f"Saved counts to        {counts_csv_path}")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Task 2: build video from frames, track people with Faster R-CNN + DeepSort, and save outputs."
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Path to folder containing frame images (e.g. C:/.../Task2/images)",
    )
    parser.add_argument(
        "--input_video",
        type=str,
        default="task2_input.mp4",
        help="Path to save the raw input video.",
    )
    parser.add_argument(
        "--tracked_video",
        type=str,
        default="task2.mp4",
        help="Path to save the tracked output video.",
    )
    parser.add_argument(
        "--tracks_txt",
        type=str,
        default="task2_tracks.txt",
        help="Path to save tracking results text file.",
    )
    parser.add_argument(
        "--counts_csv",
        type=str,
        default="task2_count.csv",
        help="Path to save frame-wise people counts CSV.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=14.0,
        help="FPS for the output videos (default: 14).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device: "cuda" or "cpu". Default: cuda (GPU if available).',
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.45,
        help="Score threshold for person detections (default 0.45).",
    )

    args = parser.parse_args()

    process_frames_to_videos_and_outputs(
        frames_dir=Path(args.frames_dir),
        input_video_path=Path(args.input_video),
        tracked_video_path=Path(args.tracked_video),
        tracks_txt_path=Path(args.tracks_txt),
        counts_csv_path=Path(args.counts_csv),
        fps=args.fps,
        device=args.device,
        score_thresh=args.score_thresh,
    )


if __name__ == "__main__":
    main()
