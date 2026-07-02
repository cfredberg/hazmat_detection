"""
HazmatAnalyzer — Inference Script

Modes:
  image   — run on a single image, print JSON, optionally save annotated output
  video   — run on a video file, save annotated video
  camera  — live webcam feed

Usage:
  python infer.py image  path/to/image.jpg [--viz]
  python infer.py video  path/to/video.mp4 [--out out.mp4] [--skip 2]
  python infer.py camera [--camera 0] [--skip 1]

Camera controls:
  q / ESC  — quit
  s        — save screenshot
  p        — pause / unpause
  + / -    — increase / decrease frame skip
"""

import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pipeline import HazmatPipeline

_COLOR_MAP = {
    "red":    (0,   0,   255),
    "orange": (0,   165, 255),
    "yellow": (0,   255, 255),
    "green":  (0,   200,   0),
    "white":  (220, 220, 220),
}

_SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "captures")


def draw(frame, output):
    img = frame.copy()
    for det in output.get("detections", []):
        x1, y1, x2, y2 = det["bounding_box"]
        color = _COLOR_MAP.get(det.get("color", "white"), (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = (
            f"{det['hazard_class']} | "
            f"{det.get('color','?')} | "
            f"{det.get('symbol','?')} "
            f"({det['confidence']:.2f})"
        )
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    timing = output.get("timing_ms", {})
    if timing:
        t_str = (f"det={timing.get('detection',0):.0f}ms  "
                 f"clf={timing.get('classification',0):.0f}ms  "
                 f"total={timing.get('total',0):.0f}ms")
        cv2.putText(img, t_str, (8, img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return img


# ── image ──────────────────────────────────────────────────────────────────────

def run_image(args):
    pipe   = HazmatPipeline(det_conf=args.conf)
    output = pipe.run(args.source)
    print(json.dumps(output, indent=2))

    if args.viz:
        img      = cv2.imread(args.source)
        out_path = os.path.splitext(args.source)[0] + "_result.jpg"
        cv2.imwrite(out_path, draw(img, output))
        print(f"Saved → {out_path}")


# ── video ──────────────────────────────────────────────────────────────────────

def run_video(args):
    pipe = HazmatPipeline(det_conf=args.conf)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.source}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = args.out or os.path.splitext(args.source)[0] + "_result.mp4"
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx   = 0
    last_result = {"detections": [], "timing_ms": {}}
    print(f"Processing {total} frames → {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % max(args.skip, 1) == 0:
            tmp = "/tmp/_hazmat_video_frame.jpg"
            cv2.imwrite(tmp, frame)
            try:
                last_result = pipe.run(tmp)
            except Exception as e:
                print(f"[frame {frame_idx}] {e}")
        writer.write(draw(frame, last_result))
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total}")
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Done → {out_path}")


# ── camera ─────────────────────────────────────────────────────────────────────

def run_camera(args):
    pipe = HazmatPipeline(det_conf=args.conf)
    print(f"Opening camera {args.camera}…")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("HazmatAnalyzer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HazmatAnalyzer", 1280, 720)

    frame_skip  = max(args.skip, 1)
    frame_idx   = 0
    last_result = {"detections": [], "timing_ms": {}}
    last_frame  = None
    paused      = False
    fps_t0      = time.perf_counter()
    fps_count   = 0
    fps_display = 0.0

    print("q=quit  s=screenshot  p=pause  +/-=frame skip")

    while True:
        ret, raw = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s') and last_frame is not None:
            os.makedirs(_SCREENSHOT_DIR, exist_ok=True)
            path = os.path.join(_SCREENSHOT_DIR,
                                f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(path, last_frame)
            print(f"Screenshot → {path}")
        elif key == ord('p'):
            paused = not paused
        elif key in (ord('+'), ord('=')):
            frame_skip = min(frame_skip + 1, 10)
        elif key == ord('-'):
            frame_skip = max(frame_skip - 1, 1)

        if paused:
            display = last_frame.copy() if last_frame is not None else raw
            cv2.putText(display, "PAUSED", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("HazmatAnalyzer", display)
            continue

        if frame_idx % frame_skip == 0:
            tmp = "/tmp/_hazmat_cam.jpg"
            cv2.imwrite(tmp, raw)
            try:
                last_result = pipe.run(tmp)
            except Exception as e:
                print(f"[inference error] {e}")
            last_frame = draw(raw, last_result)

            fps_count += 1
            elapsed = time.perf_counter() - fps_t0
            if elapsed >= 1.0:
                fps_display = fps_count / elapsed
                fps_count   = 0
                fps_t0      = time.perf_counter()

        display = last_frame.copy() if last_frame is not None else raw
        cv2.putText(display, f"FPS:{fps_display:.1f}  skip:{frame_skip}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("HazmatAnalyzer", display)
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ── entry ───────────────────────────────────────────────────────────────────────

def main():
    ap  = argparse.ArgumentParser(description="HazmatAnalyzer Inference")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_img = sub.add_parser("image")
    p_img.add_argument("source")
    p_img.add_argument("--viz",  action="store_true")
    p_img.add_argument("--conf", type=float, default=0.25)

    p_vid = sub.add_parser("video")
    p_vid.add_argument("source")
    p_vid.add_argument("--out",  type=str,   default=None)
    p_vid.add_argument("--skip", type=int,   default=1)
    p_vid.add_argument("--conf", type=float, default=0.25)

    p_cam = sub.add_parser("camera")
    p_cam.add_argument("--camera", type=int,   default=0)
    p_cam.add_argument("--skip",   type=int,   default=1)
    p_cam.add_argument("--conf",   type=float, default=0.25)

    args = ap.parse_args()
    {"image": run_image, "video": run_video, "camera": run_camera}[args.mode](args)


if __name__ == "__main__":
    main()
