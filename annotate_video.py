"""SAM 2 video tracking annotator for drone-detect ground truth.

Workflow:
    1. Load a video (or pre-extracted frame folder).
    2. For each known target person:
       a. Scrub to a frame where they're clearly visible.
       b. Click on them — SAM 2 tracks them through every frame.
       c. Refine with extra clicks where the track drifts.
       d. Mark frames where they're genuinely absent (vs occluded).
    3. Export → drone-detect-shaped ground_truth.json + per-frame mask PNGs.

Output format matches DetectClip.ground_truth in evals-api:
    {
        "000000.jpg": {
            "alice": {"found": true, "center_x": 957, "center_y": 318, "w": 138, "h": 204},
            "bob":   {"found": false}
        },
        ...
    }
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import gradio as gr
import numpy as np
import torch

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print(
        "sam2 is not installed. Install with:\n"
        "  pip install git+https://github.com/facebookresearch/sam2.git",
        file=sys.stderr,
    )
    sys.exit(1)


SAM2_CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
SAM2_CHECKPOINTS = {
    "tiny": (
        "sam2_hiera_tiny.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "sam2_hiera_t.yaml",
    ),
    "small": (
        "sam2_hiera_small.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "sam2_hiera_s.yaml",
    ),
    "base_plus": (
        "sam2_hiera_base_plus.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "sam2_hiera_b+.yaml",
    ),
    "large": (
        "sam2_hiera_large.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "sam2_hiera_l.yaml",
    ),
}

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
TAG_COLORS = [
    (0, 255, 0),
    (255, 100, 0),
    (0, 150, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (128, 0, 255),
    (255, 128, 0),
]


def download_checkpoint(model: str) -> Path:
    import urllib.request

    if model not in SAM2_CHECKPOINTS:
        raise ValueError(f"unknown model {model}; choose from {list(SAM2_CHECKPOINTS)}")
    fname, url, _ = SAM2_CHECKPOINTS[model]
    SAM2_CHECKPOINT_DIR.mkdir(exist_ok=True)
    dest = SAM2_CHECKPOINT_DIR / fname
    if dest.exists():
        print(f"checkpoint already present: {dest}")
        return dest
    print(f"downloading {model} checkpoint to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print("done")
    return dest


def extract_frames(
    video_path: Path,
    out_dir: Path,
    stride: int = 1,
    max_frames: int | None = None,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {video_path}")
    frames: list[str] = []
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            name = f"{idx:06d}.jpg"
            cv2.imwrite(str(out_dir / name), frame)
            frames.append(name)
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        idx += 1
    cap.release()
    print(
        f"extracted {saved} frames from {video_path} "
        f"(stride {stride}, max {max_frames}) into {out_dir}"
    )
    return frames


def mask_to_bbox(mask: np.ndarray) -> dict:
    if mask.ndim == 3:
        mask = mask[0]
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return {"found": False}
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return {"found": False}
    return {
        "found": True,
        "center_x": float(x1 + w / 2),
        "center_y": float(y1 + h / 2),
        "w": int(w),
        "h": int(h),
    }


def draw_overlays(
    image: np.ndarray,
    masks_by_tag: dict[str, np.ndarray | None],
    bboxes_by_tag: dict[str, dict | None],
    current_tag: str,
    tag_colors: dict[str, tuple[int, int, int]],
    prompt_points_by_tag: dict[str, list[tuple[int, int, int]]],
) -> np.ndarray:
    overlay = image.copy()
    mask_layer = np.zeros_like(image)
    for tag, mask in masks_by_tag.items():
        color = tag_colors.get(tag, (200, 200, 200))
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[0]
            mask_layer[mask > 0] = color
        bbox = bboxes_by_tag.get(tag)
        if bbox and bbox.get("found"):
            cx, cy = bbox["center_x"], bbox["center_y"]
            w, h = bbox["w"], bbox["h"]
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            thickness = 3 if tag == current_tag else 2
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                overlay,
                tag,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        for x, y, label in prompt_points_by_tag.get(tag, []):
            point_color = color if label == 1 else (0, 0, 255)
            cv2.circle(overlay, (x, y), 6, point_color, -1)
            cv2.circle(overlay, (x, y), 7, (255, 255, 255), 1)
    blended = cv2.addWeighted(overlay, 1.0, mask_layer, 0.35, 0)
    return blended


class VideoAnnotator:
    def __init__(
        self,
        frames_dir: Path,
        tags: list[str],
        model: str,
        offload_to_cpu: bool = True,
        device_override: str | None = None,
    ):
        self.frames_dir = frames_dir
        self.tags = tags
        self.tag_colors = {t: TAG_COLORS[i % len(TAG_COLORS)] for i, t in enumerate(tags)}
        self.tag_to_obj_id = {t: i + 1 for i, t in enumerate(tags)}
        self.obj_id_to_tag = {i + 1: t for i, t in enumerate(tags)}

        self.frames = sorted(
            f.name for f in frames_dir.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTS
        )
        if not self.frames:
            raise RuntimeError(f"no frames found in {frames_dir}")

        self.frame_idx = 0
        self.tag_idx = 0
        self.point_label = 1

        self.prompt_points: dict[str, dict[int, list[tuple[int, int, int]]]] = {t: {} for t in tags}
        self.absent_overrides: dict[str, set[int]] = {t: set() for t in tags}

        self._device = device_override or self._pick_device()
        if self._device == "cuda":
            print("INFO: CUDA detected. Enabling bfloat16 autocast (SAM 2 default).")
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self._device == "mps":
            print(
                "INFO: MPS detected. Forcing float32 + autocast off to avoid "
                "SAM 2's known matmul-dtype assertion."
            )
            torch.set_default_dtype(torch.float32)
        print(f"loading SAM 2 ({model}) on {self._device}...")
        ckpt = download_checkpoint(model)
        cfg = SAM2_CHECKPOINTS[model][2]
        self.predictor = build_sam2_video_predictor(cfg, str(ckpt), device=self._device)
        if self._device == "mps":
            self.predictor.to(torch.float32)
        elif self._device == "cuda":
            self.predictor.to(torch.bfloat16)
        print(
            f"initializing inference state across {len(self.frames)} frames "
            f"(offload_to_cpu={offload_to_cpu})..."
        )
        self.inference_state = self.predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=offload_to_cpu,
            offload_state_to_cpu=offload_to_cpu,
        )
        self.dirty = False

        self.video_segments: dict[int, dict[int, np.ndarray]] = {}

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def current_frame_name(self) -> str:
        return self.frames[self.frame_idx]

    @property
    def current_tag(self) -> str:
        return self.tags[self.tag_idx]

    def _load_image_rgb(self, frame_idx: int) -> np.ndarray:
        path = self.frames_dir / self.frames[frame_idx]
        img = cv2.imread(str(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _masks_at(self, frame_idx: int) -> dict[str, np.ndarray | None]:
        per_obj = self.video_segments.get(frame_idx, {})
        out: dict[str, np.ndarray | None] = {}
        for tag, obj_id in self.tag_to_obj_id.items():
            if frame_idx in self.absent_overrides[tag]:
                out[tag] = None
                continue
            mask = per_obj.get(obj_id)
            out[tag] = mask
        return out

    def _bboxes_at(self, frame_idx: int) -> dict[str, dict | None]:
        masks = self._masks_at(frame_idx)
        out: dict[str, dict | None] = {}
        for tag, mask in masks.items():
            if frame_idx in self.absent_overrides[tag]:
                out[tag] = {"found": False}
            elif mask is None:
                out[tag] = None
            else:
                out[tag] = mask_to_bbox(mask)
        return out

    def _prompts_at(self, frame_idx: int) -> dict[str, list[tuple[int, int, int]]]:
        return {tag: list(self.prompt_points[tag].get(frame_idx, [])) for tag in self.tags}

    def _render(self) -> tuple[np.ndarray, str]:
        img = self._load_image_rgb(self.frame_idx)
        rendered = draw_overlays(
            img,
            self._masks_at(self.frame_idx),
            self._bboxes_at(self.frame_idx),
            self.current_tag,
            self.tag_colors,
            self._prompts_at(self.frame_idx),
        )
        return rendered, self._status_text()

    def _status_text(self) -> str:
        frame_info = (
            f"Frame **{self.frame_idx + 1}/{len(self.frames)}** ({self.current_frame_name})"
        )
        tag_parts = []
        for i, t in enumerate(self.tags):
            n_prompts = sum(len(v) for v in self.prompt_points[t].values())
            n_absent = len(self.absent_overrides[t])
            n_tracked = sum(
                1
                for fi in range(len(self.frames))
                if t in self._bboxes_at(fi)
                and self._bboxes_at(fi)[t]
                and self._bboxes_at(fi)[t].get("found")
            )
            label = f"{t} ({n_prompts}p / {n_tracked}f / {n_absent}a)"
            if i == self.tag_idx:
                tag_parts.append(f"**[{label}]**")
            else:
                tag_parts.append(label)
        click_mode = "POSITIVE" if self.point_label == 1 else "NEGATIVE"
        dirty_marker = " ⚠️ unpropagated changes — click Propagate" if self.dirty else ""
        return (
            f"{frame_info} | click mode: **{click_mode}** | current tag: **{self.current_tag}**{dirty_marker}\n"
            f"Targets (clicks / tracked frames / absent overrides): " + "  ".join(tag_parts)
        )

    def handle_click(self, evt: gr.SelectData) -> tuple[np.ndarray, str]:
        x, y = evt.index
        tag = self.current_tag
        obj_id = self.tag_to_obj_id[tag]
        per_frame = self.prompt_points[tag].setdefault(self.frame_idx, [])
        per_frame.append((int(x), int(y), int(self.point_label)))

        all_pts = []
        all_labels = []
        for px, py, lab in per_frame:
            all_pts.append([px, py])
            all_labels.append(lab)
        points = np.array(all_pts, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int32)

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=self.frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        self.video_segments.setdefault(self.frame_idx, {})
        for i, oid in enumerate(out_obj_ids):
            self.video_segments[self.frame_idx][int(oid)] = (out_mask_logits[i] > 0.0).cpu().numpy()
        self.dirty = True
        return self._render()

    def propagate_now(self) -> tuple[np.ndarray, str]:
        if not any(by_frame for by_frame in self.prompt_points.values()):
            return self._render()
        self._propagate()
        self.dirty = False
        return self._render()

    def _propagate(self) -> None:
        new_segments: dict[int, dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
            self.inference_state
        ):
            new_segments[frame_idx] = {
                int(obj_id): (mask_logits[i] > 0.0).cpu().numpy()
                for i, obj_id in enumerate(obj_ids)
            }
        self.video_segments = new_segments

    def mark_absent_here(self) -> tuple[np.ndarray, str]:
        self.absent_overrides[self.current_tag].add(self.frame_idx)
        return self._render()

    def clear_absent_here(self) -> tuple[np.ndarray, str]:
        self.absent_overrides[self.current_tag].discard(self.frame_idx)
        return self._render()

    def reset_current_tag(self) -> tuple[np.ndarray, str]:
        tag = self.current_tag
        obj_id = self.tag_to_obj_id[tag]
        self.prompt_points[tag] = {}
        self.absent_overrides[tag] = set()
        try:
            self.predictor.reset_state(self.inference_state)
        except Exception:
            print("predictor.reset_state unavailable — full reset; reapplying other tags...")
        self._reapply_all_prompts()
        self._propagate()
        return self._render()

    def _reapply_all_prompts(self) -> None:
        for tag, by_frame in self.prompt_points.items():
            if not by_frame:
                continue
            obj_id = self.tag_to_obj_id[tag]
            for frame_idx, pts in by_frame.items():
                if not pts:
                    continue
                points = np.array([[px, py] for px, py, _ in pts], dtype=np.float32)
                labels = np.array([lab for _, _, lab in pts], dtype=np.int32)
                self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

    def toggle_click_mode(self) -> tuple[np.ndarray, str]:
        self.point_label = 0 if self.point_label == 1 else 1
        return self._render()

    def next_tag(self) -> tuple[np.ndarray, str]:
        self.tag_idx = (self.tag_idx + 1) % len(self.tags)
        return self._render()

    def prev_tag(self) -> tuple[np.ndarray, str]:
        self.tag_idx = (self.tag_idx - 1) % len(self.tags)
        return self._render()

    def goto_frame(self, frame_idx: int) -> tuple[np.ndarray, str]:
        self.frame_idx = max(0, min(int(frame_idx) - 1, len(self.frames) - 1))
        return self._render()

    def export(self, out_path: str) -> str:
        ground_truth: dict[str, dict[str, dict]] = {}
        for fi, frame_name in enumerate(self.frames):
            per_target: dict[str, dict] = {}
            bboxes = self._bboxes_at(fi)
            for tag in self.tags:
                bbox = bboxes.get(tag)
                if bbox is None:
                    per_target[tag] = {"found": False}
                else:
                    per_target[tag] = bbox
            ground_truth[frame_name] = per_target
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(ground_truth, f, indent=2)
        n_total = len(self.frames) * len(self.tags)
        n_found = sum(
            1 for frame in ground_truth.values() for tag in frame.values() if tag.get("found")
        )
        return f"saved {out} ({n_found}/{n_total} target-frames found)"


def build_ui(app: VideoAnnotator, default_export_path: str) -> gr.Blocks:
    with gr.Blocks(title="SAM 2 Video Annotator") as demo:
        gr.Markdown("# SAM 2 Video Annotator")
        gr.Markdown(
            "Click on a target to add a prompt; SAM 2 propagates the mask through every frame. "
            "Refine with extra clicks, scrub via the slider, mark frames as absent when needed."
        )
        status = gr.Markdown(app._status_text())
        image = gr.Image(value=app._render()[0], label="Current frame", interactive=False)
        slider = gr.Slider(minimum=1, maximum=len(app.frames), step=1, value=1, label="Frame")
        with gr.Row():
            prev_tag_btn = gr.Button("← Tag")
            next_tag_btn = gr.Button("Tag →")
            toggle_click_btn = gr.Button("Toggle +/-")
        with gr.Row():
            propagate_btn = gr.Button("Propagate to all frames", variant="primary")
            mark_absent_btn = gr.Button("Mark absent here")
            clear_absent_btn = gr.Button("Clear absent here")
            reset_tag_btn = gr.Button("Reset current tag", variant="stop")
        with gr.Row():
            export_path = gr.Textbox(value=default_export_path, label="Export path", scale=4)
            export_btn = gr.Button("Export GT JSON", variant="primary", scale=1)
        export_msg = gr.Textbox(label="Status", visible=False)

        image.select(app.handle_click, outputs=[image, status])
        slider.change(app.goto_frame, inputs=[slider], outputs=[image, status])
        prev_tag_btn.click(app.prev_tag, outputs=[image, status])
        next_tag_btn.click(app.next_tag, outputs=[image, status])
        toggle_click_btn.click(app.toggle_click_mode, outputs=[image, status])
        propagate_btn.click(app.propagate_now, outputs=[image, status])
        mark_absent_btn.click(app.mark_absent_here, outputs=[image, status])
        clear_absent_btn.click(app.clear_absent_here, outputs=[image, status])
        reset_tag_btn.click(app.reset_current_tag, outputs=[image, status])
        export_btn.click(app.export, inputs=[export_path], outputs=[export_msg]).then(
            lambda: gr.update(visible=True), outputs=[export_msg]
        )

    return demo


def resolve_frames_dir(args: argparse.Namespace) -> Path:
    if args.video:
        video = Path(args.video)
        if not video.exists() or video.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
            raise SystemExit(f"--video must point to an existing {SUPPORTED_VIDEO_EXTS} file")
        out_dir = Path(args.work_dir) if args.work_dir else video.parent / f"{video.stem}_frames"
        if args.force_extract and out_dir.exists():
            shutil.rmtree(out_dir)
        if not out_dir.exists() or not any(out_dir.iterdir()):
            extract_frames(video, out_dir, stride=args.stride, max_frames=args.max_frames)
        else:
            print(f"reusing existing frames in {out_dir} (use --force-extract to re-extract)")
        return out_dir
    if args.folder:
        folder = Path(args.folder)
        if not folder.exists() or not folder.is_dir():
            raise SystemExit("--folder must point to an existing directory")
        return folder
    raise SystemExit("must provide --video or --folder")


def main():
    parser = argparse.ArgumentParser(description="SAM 2 video annotator for drone-detect GT")
    parser.add_argument("--video", type=str, help="Path to video file (mp4/mov/avi/mkv)")
    parser.add_argument("--folder", type=str, help="Folder of pre-extracted frames")
    parser.add_argument(
        "--work-dir", type=str, help="Where to extract frames (default: <video>_frames)"
    )
    parser.add_argument("--stride", type=int, default=1, help="Extract every Nth frame (default 1)")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop extracting after this many frames (default: whole video). "
        "Combine with --stride to process a manageable slice.",
    )
    parser.add_argument(
        "--force-extract", action="store_true", help="Re-extract frames even if they exist"
    )
    parser.add_argument("--tags", type=str, help="Comma-separated target names")
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        choices=list(SAM2_CHECKPOINTS),
        help="SAM 2 model size (default: tiny — 4-5x faster than base_plus, "
        "quality usually fine for body bboxes). Use base_plus or large for harder footage.",
    )
    parser.add_argument(
        "--export", type=str, help="Default export path (default: <frames>/ground_truth.json)"
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--no-offload",
        action="store_true",
        help="Keep all video frames on GPU/MPS (default: offload to CPU; needed for long videos on MPS)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device. Default auto picks cuda > mps > cpu. Use cpu if MPS crashes.",
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Just download the model and exit"
    )
    args = parser.parse_args()

    if args.download_only:
        download_checkpoint(args.model)
        return

    if not args.tags:
        raise SystemExit("--tags is required (e.g. --tags alice,bob,charlie)")

    frames_dir = resolve_frames_dir(args)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    if not tags:
        raise SystemExit("no tags provided")
    default_export = args.export or str(frames_dir / "ground_truth.json")

    device_override = None if args.device == "auto" else args.device
    app = VideoAnnotator(
        frames_dir,
        tags,
        args.model,
        offload_to_cpu=not args.no_offload,
        device_override=device_override,
    )
    demo = build_ui(app, default_export)
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
