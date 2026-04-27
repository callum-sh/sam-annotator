"""Render annotated copies of frames so you can scrub through and spot-check.

Reads a ground_truth.json (in drone-detect schema) + a frames dir, draws each
target's bbox on the frame in a color-coded box with the target's name,
writes the result as annotated_<frame>.jpg into an output dir.

Usage:
    python inspect_annotations.py \\
        --frames /Users/callum/Downloads/demo_frames \\
        --gt /Users/callum/Downloads/demo_frames/ground_truth.json \\
        --out /Users/callum/Downloads/demo_annotated
"""

import argparse
import json
from pathlib import Path

import cv2

TAG_COLORS_BGR = [
    (0, 255, 0),
    (0, 100, 255),
    (255, 150, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 128),
    (0, 128, 255),
]


def color_for(tag: str, tag_index: dict[str, int]) -> tuple[int, int, int]:
    if tag not in tag_index:
        tag_index[tag] = len(tag_index)
    return TAG_COLORS_BGR[tag_index[tag] % len(TAG_COLORS_BGR)]


def draw_frame(
    img: cv2.typing.MatLike,
    per_target: dict[str, dict],
    tag_index: dict[str, int],
) -> tuple[cv2.typing.MatLike, dict]:
    out = img.copy()
    found_count = 0
    legend_y = 24
    for tag in sorted(per_target):
        entry = per_target[tag]
        if not isinstance(entry, dict):
            continue
        color = color_for(tag, tag_index)
        if entry.get("found"):
            cx = float(entry["center_x"])
            cy = float(entry["center_y"])
            w = float(entry["w"])
            h = float(entry["h"])
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, tag, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            found_count += 1
            label = f"{tag}: bbox"
        else:
            label = f"{tag}: absent"
        cv2.putText(out, label, (8, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        legend_y += 18
    return out, {"found_count": found_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, type=Path, help="Directory of source frames")
    parser.add_argument("--gt", required=True, type=Path, help="ground_truth.json path")
    parser.add_argument(
        "--out", required=True, type=Path, help="Output directory for annotated frames"
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Render every Nth annotated frame (default 1; bump up for fast skim)",
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Write annotated_<name>.jpg by default; this flag writes <name>.jpg directly",
    )
    args = parser.parse_args()

    if not args.frames.exists():
        raise SystemExit(f"frames dir not found: {args.frames}")
    if not args.gt.exists():
        raise SystemExit(f"ground truth not found: {args.gt}")

    ground_truth = json.loads(args.gt.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    tag_index: dict[str, int] = {}
    rendered = 0
    skipped = 0
    found_total = 0
    target_total = 0

    sorted_frames = sorted(ground_truth.keys())
    for i, frame_name in enumerate(sorted_frames):
        if i % args.every != 0:
            continue
        src = args.frames / frame_name
        if not src.exists():
            skipped += 1
            continue
        img = cv2.imread(str(src))
        if img is None:
            skipped += 1
            continue
        per_target = ground_truth[frame_name] or {}
        annotated, stats = draw_frame(img, per_target, tag_index)
        target_total += sum(1 for v in per_target.values() if isinstance(v, dict))
        found_total += stats["found_count"]
        out_name = frame_name if args.no_prefix else f"annotated_{frame_name}"
        cv2.imwrite(str(args.out / out_name), annotated)
        rendered += 1

    print(f"rendered {rendered} frames -> {args.out}")
    if skipped:
        print(f"  skipped {skipped} frames (no source image)")
    print(f"  total target-frames: {target_total}, found: {found_total}")
    if tag_index:
        print(f"  tags + colors: {dict(tag_index)}")


if __name__ == "__main__":
    main()
