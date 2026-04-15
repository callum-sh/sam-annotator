import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
SAM_CHECKPOINT_PATH = SAM_CHECKPOINT_DIR / "sam_vit_b_01ec64.pth"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
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


def download_model():
    SAM_CHECKPOINT_DIR.mkdir(exist_ok=True)
    if SAM_CHECKPOINT_PATH.exists():
        print(f"Checkpoint already exists: {SAM_CHECKPOINT_PATH}")
        return
    print(f"Downloading SAM ViT-B checkpoint to {SAM_CHECKPOINT_PATH}...")
    urllib.request.urlretrieve(SAM_CHECKPOINT_URL, SAM_CHECKPOINT_PATH)
    print("Done.")


def load_sam_predictor() -> SamPredictor:
    if not SAM_CHECKPOINT_PATH.exists():
        print("SAM checkpoint not found. Run with --download-model first.")
        sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT_PATH))
    sam.to(device)
    return SamPredictor(sam)


def mask_to_bbox(mask: np.ndarray) -> dict:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return {"found": False}
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    w = x2 - x1
    h = y2 - y1
    return {
        "found": True,
        "center_x": float(x1 + w / 2),
        "center_y": float(y1 + h / 2),
        "w": w,
        "h": h,
    }


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def draw_annotations(
    image: np.ndarray,
    annotations: dict[str, list[dict]],
    current_tag: str,
    tag_colors: dict[str, tuple[int, int, int]],
    folder: Path,
) -> np.ndarray:
    overlay = image.copy()
    mask_layer = np.zeros_like(image)
    for tag, entries in annotations.items():
        color = tag_colors.get(tag, (200, 200, 200))
        is_current = tag == current_tag
        thickness = 3 if is_current else 2
        for ann in entries:
            if not ann.get("found"):
                continue
            cx, cy, w, h = ann["center_x"], ann["center_y"], ann["w"], ann["h"]
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(overlay, tag, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            mask_file = ann.get("mask_file")
            if mask_file:
                full = folder / mask_file
                if full.exists():
                    m = cv2.imread(str(full), cv2.IMREAD_GRAYSCALE)
                    if m is not None and m.shape[:2] == image.shape[:2]:
                        mask_layer[m > 127] = color
    result = cv2.addWeighted(overlay, 1.0, mask_layer, 0.3, 0)
    return result


def migrate_annotations(annotations: dict) -> dict[str, dict[str, list[dict]]]:
    migrated: dict[str, dict[str, list[dict]]] = {}
    for frame, tags in annotations.items():
        migrated[frame] = {}
        for tag, value in tags.items():
            if isinstance(value, list):
                migrated[frame][tag] = value
            else:
                migrated[frame][tag] = [value]
    return migrated


class AnnotationApp:
    def __init__(self, folder: str, tags: list[str]):
        self.folder = Path(folder)
        self.masks_dir = self.folder / "masks"
        self.masks_dir.mkdir(exist_ok=True)
        self.tags = tags
        self.tag_colors = {t: TAG_COLORS[i % len(TAG_COLORS)] for i, t in enumerate(tags)}
        self.frames = sorted(
            f.name for f in self.folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not self.frames:
            print(f"No images found in {folder}")
            sys.exit(1)

        self.annotations: dict[str, dict[str, list[dict]]] = {}
        self.frame_idx = 0
        self.tag_idx = 0
        self.predictor = load_sam_predictor()
        self._current_image: np.ndarray | None = None

        save_path = self.folder / "annotations.json"
        if save_path.exists():
            with open(save_path) as f:
                raw = json.load(f)
            self.annotations = migrate_annotations(raw)
            for i, frame in enumerate(self.frames):
                if frame not in self.annotations:
                    self.frame_idx = i
                    break
            else:
                self.frame_idx = 0
            print(f"Resumed from {save_path} ({len(self.annotations)} frames annotated)")

    @property
    def current_frame(self) -> str:
        return self.frames[self.frame_idx]

    @property
    def current_tag(self) -> str:
        return self.tags[self.tag_idx]

    @property
    def current_annotations(self) -> dict[str, list[dict]]:
        return self.annotations.setdefault(self.current_frame, {})

    def load_image(self) -> np.ndarray:
        path = self.folder / self.current_frame
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._current_image = img
        self.predictor.set_image(img)
        return img

    def get_display_image(self) -> np.ndarray:
        if self._current_image is None:
            self.load_image()
        return draw_annotations(
            self._current_image,
            self.current_annotations,
            self.current_tag,
            self.tag_colors,
            self.folder,
        )

    def status_text(self) -> str:
        frame_info = f"Frame: {self.current_frame} ({self.frame_idx + 1}/{len(self.frames)})"
        tag_parts = []
        for i, t in enumerate(self.tags):
            entries = self.current_annotations.get(t)
            if entries is None:
                marker = "?"
            elif len(entries) == 1 and not entries[0].get("found"):
                marker = "not found"
            else:
                count = sum(1 for e in entries if e.get("found"))
                marker = f"x{count}"
            if i == self.tag_idx:
                tag_parts.append(f"**[{t} {marker}]**")
            else:
                tag_parts.append(f"{t} {marker}")
        tag_info = "Tags: " + "  ".join(tag_parts)
        return f"{frame_info}\n{tag_info}"

    def _load_mask(self, mask_file: str) -> np.ndarray | None:
        full = self.folder / mask_file
        if not full.exists():
            return None
        m = cv2.imread(str(full), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        return m > 127

    def handle_click(self, evt: gr.SelectData) -> tuple[np.ndarray, str]:
        x, y = evt.index
        point = np.array([[x, y]])
        label = np.array([1])
        masks, scores, _ = self.predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        new_mask = masks[best_idx]

        entries = self.current_annotations.setdefault(self.current_tag, [])
        merged = False
        for entry in entries:
            if not entry.get("found") or not entry.get("mask_file"):
                continue
            existing = self._load_mask(entry["mask_file"])
            if existing is not None and np.any(existing & new_mask):
                combined = existing | new_mask
                save_mask(combined, self.folder / entry["mask_file"])
                bbox = mask_to_bbox(combined)
                bbox["mask_file"] = entry["mask_file"]
                entry.update(bbox)
                merged = True
                break

        if not merged:
            frame_stem = Path(self.current_frame).stem
            existing_found = [e for e in entries if e.get("found")]
            idx = len(existing_found)
            mask_rel = f"masks/{frame_stem}_{self.current_tag}_{idx}.png"
            save_mask(new_mask, self.folder / mask_rel)
            bbox = mask_to_bbox(new_mask)
            bbox["mask_file"] = mask_rel
            entries.append(bbox)

        return self.get_display_image(), self.status_text()

    def mark_not_found(self) -> tuple[np.ndarray, str]:
        self._delete_masks_for_tag(self.current_tag)
        self.current_annotations[self.current_tag] = [{"found": False}]
        return self.get_display_image(), self.status_text()

    def clear_current(self) -> tuple[np.ndarray, str]:
        self._delete_masks_for_tag(self.current_tag)
        self.current_annotations.pop(self.current_tag, None)
        return self.get_display_image(), self.status_text()

    def next_tag(self) -> tuple[np.ndarray, str]:
        self.tag_idx = (self.tag_idx + 1) % len(self.tags)
        return self.get_display_image(), self.status_text()

    def prev_tag(self) -> tuple[np.ndarray, str]:
        self.tag_idx = (self.tag_idx - 1) % len(self.tags)
        return self.get_display_image(), self.status_text()

    def next_frame(self) -> tuple[np.ndarray, str]:
        self._save()
        if self.frame_idx < len(self.frames) - 1:
            self.frame_idx += 1
            self.tag_idx = 0
            self.load_image()
        return self.get_display_image(), self.status_text()

    def prev_frame(self) -> tuple[np.ndarray, str]:
        self._save()
        if self.frame_idx > 0:
            self.frame_idx -= 1
            self.tag_idx = 0
            self.load_image()
        return self.get_display_image(), self.status_text()

    def export(self) -> str:
        self._save()
        out_path = self.folder / "annotations.json"
        annotated = len(self.annotations)
        total = len(self.frames)
        return f"Saved {annotated}/{total} frames to {out_path}"

    def _save(self):
        out_path = self.folder / "annotations.json"
        with open(out_path, "w") as f:
            json.dump(self.annotations, f, indent=2)

    def _delete_masks_for_tag(self, tag: str):
        entries = self.current_annotations.get(tag, [])
        for entry in entries:
            mask_file = entry.get("mask_file")
            if mask_file:
                full = self.folder / mask_file
                if full.exists():
                    full.unlink()


def build_ui(app: AnnotationApp) -> gr.Blocks:
    with gr.Blocks(title="SAM Annotator") as demo:
        gr.Markdown("# SAM Annotator")
        status = gr.Markdown(app.status_text())
        image = gr.Image(
            value=app.get_display_image(),
            label="Click on a person to annotate",
            interactive=False,
        )
        with gr.Row():
            not_found_btn = gr.Button("Not Found")
            clear_btn = gr.Button("Clear")
            prev_tag_btn = gr.Button("← Tag")
            next_tag_btn = gr.Button("Tag →")
        with gr.Row():
            prev_btn = gr.Button("← Prev Frame")
            next_btn = gr.Button("Next Frame →")
            export_btn = gr.Button("Export", variant="primary")
        export_msg = gr.Textbox(label="Export", visible=False)

        image.select(app.handle_click, outputs=[image, status])
        not_found_btn.click(app.mark_not_found, outputs=[image, status])
        clear_btn.click(app.clear_current, outputs=[image, status])
        next_tag_btn.click(app.next_tag, outputs=[image, status])
        prev_tag_btn.click(app.prev_tag, outputs=[image, status])
        next_btn.click(app.next_frame, outputs=[image, status])
        prev_btn.click(app.prev_frame, outputs=[image, status])
        export_btn.click(app.export, outputs=[export_msg]).then(
            lambda: gr.update(visible=True), outputs=[export_msg]
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="SAM-powered annotation tool")
    parser.add_argument("--folder", type=str, help="Folder containing frame images")
    parser.add_argument("--tags", type=str, help="Comma-separated person tags")
    parser.add_argument("--download-model", action="store_true", help="Download SAM checkpoint")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if args.download_model:
        download_model()
        if not args.folder:
            return

    if not args.folder or not args.tags:
        parser.print_help()
        return

    tags = [t.strip() for t in args.tags.split(",")]
    app = AnnotationApp(args.folder, tags)
    demo = build_ui(app)
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
