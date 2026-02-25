"""GStreamer camera pipeline for USB fisheye camera on Jetson.

Uses NVMM (NVIDIA Multimedia) for GPU-accelerated capture and resize.
Provides frame stacking for temporal context matching simulation.

Usage:
    cam = CameraCapture(device="/dev/video0")
    cam.start()
    frames = cam.get_stacked_frames()  # (1, 12, 224, 224) tensor
    cam.stop()
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None


@dataclass
class CameraConfig:
    """Camera capture configuration."""
    device: str = "/dev/video0"
    capture_width: int = 640
    capture_height: int = 480
    output_width: int = 224
    output_height: int = 224
    fps: int = 30
    num_frames: int = 4  # Frame stack size
    use_gstreamer: bool = True  # Use GStreamer pipeline (Jetson-optimized)


# GStreamer pipeline for Jetson with NVMM acceleration
GSTREAMER_PIPELINE = (
    "v4l2src device={device} ! "
    "video/x-raw,width={cap_w},height={cap_h},framerate={fps}/1 ! "
    "nvvidconv ! video/x-raw(memory:NVMM),width={out_w},height={out_h},format=RGBA ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=1 max-buffers=1 sync=false"
)

# Fallback pipeline for non-Jetson (development/testing)
FALLBACK_PIPELINE = (
    "v4l2src device={device} ! "
    "video/x-raw,width={cap_w},height={cap_h},framerate={fps}/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "videoscale ! video/x-raw,width={out_w},height={out_h} ! "
    "appsink drop=1 max-buffers=1 sync=false"
)


class CameraCapture:
    """Threaded camera capture with frame stacking."""

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self._cap = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Frame buffer: list of numpy arrays (H, W, 3) float32 [0, 1]
        self._frame_buffer = [
            np.zeros((self.config.output_height, self.config.output_width, 3), dtype=np.float32)
            for _ in range(self.config.num_frames)
        ]
        self._frame_count = 0

    def start(self):
        """Start camera capture thread."""
        if cv2 is None:
            raise RuntimeError("OpenCV not installed")

        pipeline = self._build_pipeline()
        print(f"[Camera] Opening: {pipeline[:80]}...")

        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            # Fallback to direct device
            print("[Camera] GStreamer failed, trying direct device...")
            self._cap = cv2.VideoCapture(self.config.device)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.config.device}")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[Camera] Started ({self.config.output_width}x{self.config.output_height} @ {self.config.fps}fps)")

    def stop(self):
        """Stop camera capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        print("[Camera] Stopped")

    def get_frame(self) -> np.ndarray:
        """Get latest frame as (H, W, 3) float32 [0, 1]."""
        with self._lock:
            return self._frame_buffer[-1].copy()

    def get_stacked_frames(self) -> np.ndarray:
        """Get stacked frames as (num_frames, 3, H, W) float32 [0, 1].

        Returns numpy array ready for torch conversion.
        """
        with self._lock:
            # Stack and rearrange: (num_frames, H, W, 3) → (num_frames, 3, H, W)
            stacked = np.stack(self._frame_buffer, axis=0)
        return stacked.transpose(0, 3, 1, 2)

    def get_stacked_tensor(self) -> "torch.Tensor":
        """Get stacked frames as (1, num_frames*3, H, W) torch tensor."""
        if torch is None:
            raise RuntimeError("PyTorch not installed")
        frames = self.get_stacked_frames()  # (num_frames, 3, H, W)
        # Reshape to (1, num_frames*3, H, W)
        tensor = torch.from_numpy(frames).reshape(1, -1, self.config.output_height, self.config.output_width)
        return tensor

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _build_pipeline(self) -> str:
        """Build GStreamer pipeline string."""
        template = GSTREAMER_PIPELINE if self.config.use_gstreamer else FALLBACK_PIPELINE
        return template.format(
            device=self.config.device,
            cap_w=self.config.capture_width,
            cap_h=self.config.capture_height,
            out_w=self.config.output_width,
            out_h=self.config.output_height,
            fps=self.config.fps,
        )

    def _capture_loop(self):
        """Background thread: continuously capture frames."""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            # Resize if not already done by GStreamer
            h, w = frame.shape[:2]
            if h != self.config.output_height or w != self.config.output_width:
                frame = cv2.resize(
                    frame,
                    (self.config.output_width, self.config.output_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            # BGR → RGB, normalize to [0, 1]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Push into frame buffer (FIFO)
            with self._lock:
                self._frame_buffer.pop(0)
                self._frame_buffer.append(frame_rgb)
                self._frame_count += 1


if __name__ == "__main__":
    # Quick test
    cam = CameraCapture()
    cam.start()
    time.sleep(2)
    frames = cam.get_stacked_frames()
    print(f"Frame shape: {frames.shape}, dtype: {frames.dtype}, range: [{frames.min():.2f}, {frames.max():.2f}]")
    cam.stop()
