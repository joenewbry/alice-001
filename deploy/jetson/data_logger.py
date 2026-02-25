"""HDF5 episode data logger for real-world demonstrations and rollouts.

Records camera frames, joint positions, actions, and metadata per episode.
Data format is compatible with HIL-SERL and Octo fine-tuning pipelines.

Usage:
    logger = DataLogger(LogConfig(log_dir="/ssd/alice/data"))
    logger.new_episode()
    for step in range(100):
        logger.log_step(images=img, joint_pos=pos, actions=act)
    logger.end_episode(success=True)
    logger.close()

Data format (HDF5):
    /episode_0/
        images:    (T, num_frames*3, 224, 224) float32
        joint_pos: (T, 7) float32
        joint_vel: (T, 7) float32
        actions:   (T, 7) float32
        timestamps: (T,) float64
        attrs: {success: bool, duration_s: float, num_steps: int}
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


@dataclass
class LogConfig:
    """Data logging configuration."""
    log_dir: str = "/ssd/alice/data"
    max_episode_steps: int = 2000
    image_shape: tuple = (12, 224, 224)  # 4 frames * 3 channels
    compress: bool = True  # gzip compression for images
    compress_level: int = 4  # 1-9, balance speed vs size


class DataLogger:
    """HDF5 episode data logger."""

    def __init__(self, config: Optional[LogConfig] = None):
        self.config = config or LogConfig()
        self._file = None
        self._episode_group = None
        self._episode_idx = 0
        self._step_idx = 0
        self._episode_start = None

        # Temporary buffers (flush to HDF5 on episode end)
        self._buf_images = []
        self._buf_joint_pos = []
        self._buf_joint_vel = []
        self._buf_actions = []
        self._buf_timestamps = []

    def open(self, filename: Optional[str] = None):
        """Open or create HDF5 file."""
        if h5py is None:
            print("[Logger] h5py not installed, logging disabled")
            return

        os.makedirs(self.config.log_dir, exist_ok=True)
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.config.log_dir, f"episodes_{timestamp}.hdf5")

        self._file = h5py.File(filename, "w")
        self._file.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Logger] Opened {filename}")

    def new_episode(self):
        """Start recording a new episode."""
        if self._file is None:
            self.open()

        # Flush previous episode if needed
        if self._buf_images:
            self._flush_episode()

        self._step_idx = 0
        self._episode_start = time.time()
        self._buf_images = []
        self._buf_joint_pos = []
        self._buf_joint_vel = []
        self._buf_actions = []
        self._buf_timestamps = []

    def log_step(
        self,
        images: Optional[np.ndarray] = None,
        joint_pos: Optional[np.ndarray] = None,
        joint_vel: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
    ):
        """Log a single timestep.

        Args:
            images: Camera frames, shape varies (will be stored as-is)
            joint_pos: (7,) joint positions in radians
            joint_vel: (7,) joint velocities
            actions: (7,) policy actions
        """
        self._buf_timestamps.append(time.time())

        if images is not None:
            self._buf_images.append(images.copy() if isinstance(images, np.ndarray) else images)
        if joint_pos is not None:
            self._buf_joint_pos.append(np.array(joint_pos, dtype=np.float32))
        if joint_vel is not None:
            self._buf_joint_vel.append(np.array(joint_vel, dtype=np.float32))
        if actions is not None:
            self._buf_actions.append(np.array(actions, dtype=np.float32))

        self._step_idx += 1

    def end_episode(self, success: bool = False, metadata: Optional[dict] = None):
        """End current episode and write to HDF5.

        Args:
            success: Whether the episode was successful
            metadata: Additional metadata to store
        """
        self._flush_episode(success=success, metadata=metadata)

    def _flush_episode(self, success: bool = False, metadata: Optional[dict] = None):
        """Write buffered episode data to HDF5."""
        if self._file is None or not self._buf_timestamps:
            return

        ep_name = f"episode_{self._episode_idx}"
        grp = self._file.create_group(ep_name)

        # Timestamps
        grp.create_dataset("timestamps", data=np.array(self._buf_timestamps))

        # Images (with compression)
        if self._buf_images:
            img_data = np.stack(self._buf_images, axis=0)
            if self.config.compress:
                grp.create_dataset(
                    "images", data=img_data,
                    compression="gzip", compression_opts=self.config.compress_level,
                )
            else:
                grp.create_dataset("images", data=img_data)

        # Joint positions
        if self._buf_joint_pos:
            grp.create_dataset("joint_pos", data=np.stack(self._buf_joint_pos))

        # Joint velocities
        if self._buf_joint_vel:
            grp.create_dataset("joint_vel", data=np.stack(self._buf_joint_vel))

        # Actions
        if self._buf_actions:
            grp.create_dataset("actions", data=np.stack(self._buf_actions))

        # Episode metadata
        duration = time.time() - (self._episode_start or time.time())
        grp.attrs["success"] = success
        grp.attrs["num_steps"] = self._step_idx
        grp.attrs["duration_s"] = duration
        grp.attrs["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        if metadata:
            for k, v in metadata.items():
                grp.attrs[k] = v

        self._file.flush()
        self._episode_idx += 1

        status = "SUCCESS" if success else "recorded"
        print(f"[Logger] Episode {self._episode_idx - 1}: {self._step_idx} steps, "
              f"{duration:.1f}s, {status}")

    def close(self):
        """Flush and close HDF5 file."""
        if self._buf_timestamps:
            self._flush_episode()
        if self._file:
            self._file.close()
            print(f"[Logger] Closed ({self._episode_idx} episodes)")

    @property
    def num_episodes(self) -> int:
        return self._episode_idx

    @staticmethod
    def load_episodes(filepath: str) -> dict:
        """Load all episodes from an HDF5 file.

        Returns:
            Dict mapping episode names to dicts of numpy arrays
        """
        episodes = {}
        with h5py.File(filepath, "r") as f:
            for ep_name in f.keys():
                ep = {}
                grp = f[ep_name]
                for key in grp.keys():
                    ep[key] = grp[key][:]
                ep["attrs"] = dict(grp.attrs)
                episodes[ep_name] = ep
        return episodes
