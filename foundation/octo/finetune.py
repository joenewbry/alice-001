"""Fine-tune Octo foundation model on real xArm demonstrations.

Octo is pretrained on 800k robot episodes from Open X-Embodiment,
providing manipulation priors. Fine-tuning on 5-20 real demos adapts it
to the specific robot morphology and task.

Prerequisites:
    pip install octo-model  # or install from github.com/octo-model/octo
    # Download pretrained weights (auto on first use)

Usage:
    # Fine-tune Octo-Small on collected demos
    python -m foundation.octo.finetune \
        --data-dir /ssd/alice/demos \
        --model octo-small \
        --output /ssd/alice/models/octo_finetuned \
        --epochs 100

    # Fine-tune Octo-Base (larger, better generalization)
    python -m foundation.octo.finetune \
        --data-dir /ssd/alice/demos \
        --model octo-base \
        --output /ssd/alice/models/octo_finetuned \
        --epochs 50 --batch-size 16

Data format:
    Expects HDF5 files from deploy/jetson/data_logger.py with:
        /episode_N/images: (T, C, H, W) float32
        /episode_N/joint_pos: (T, 7) float32
        /episode_N/actions: (T, 7) float32
        /episode_N/attrs: {success: bool}
"""

import os
import glob
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None


@dataclass
class FinetuneConfig:
    """Octo fine-tuning configuration."""
    data_dir: str = "/ssd/alice/demos"
    model_name: str = "octo-small"  # "octo-small" (27M) or "octo-base" (93M)
    output_dir: str = "/ssd/alice/models/octo_finetuned"

    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Data
    image_size: tuple = (224, 224)
    action_dim: int = 7
    proprio_dim: int = 7
    sequence_length: int = 10  # Prediction horizon

    # Only use successful episodes
    success_only: bool = True


class DemoDataset(Dataset):
    """Dataset of teleoperation demonstrations from HDF5 files."""

    def __init__(self, data_dir: str, config: FinetuneConfig):
        self.config = config
        self.episodes = []

        # Load all HDF5 files
        h5_files = glob.glob(os.path.join(data_dir, "**/*.hdf5"), recursive=True)
        print(f"[Data] Found {len(h5_files)} HDF5 files in {data_dir}")

        for fpath in h5_files:
            with h5py.File(fpath, "r") as f:
                for ep_name in f.keys():
                    grp = f[ep_name]
                    if config.success_only and not grp.attrs.get("success", False):
                        continue

                    episode = {
                        "joint_pos": grp["joint_pos"][:],
                        "actions": grp["actions"][:],
                    }
                    if "images" in grp:
                        episode["images"] = grp["images"][:]

                    self.episodes.append(episode)

        print(f"[Data] Loaded {len(self.episodes)} episodes")
        if not self.episodes:
            raise ValueError(f"No {'successful ' if config.success_only else ''}episodes found in {data_dir}")

        # Build index: (episode_idx, start_step)
        self._index = []
        for ep_idx, ep in enumerate(self.episodes):
            T = len(ep["actions"])
            for t in range(max(1, T - config.sequence_length)):
                self._index.append((ep_idx, t))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ep_idx, t = self._index[idx]
        ep = self.episodes[ep_idx]
        seq_len = self.config.sequence_length
        T = len(ep["actions"])

        # Get sequence (pad if needed)
        end_t = min(t + seq_len, T)

        # Current observation
        joint_pos = torch.tensor(ep["joint_pos"][t], dtype=torch.float32)

        # Action sequence to predict
        actions = torch.tensor(ep["actions"][t:end_t], dtype=torch.float32)
        if len(actions) < seq_len:
            pad = torch.zeros(seq_len - len(actions), self.config.action_dim)
            actions = torch.cat([actions, pad], dim=0)

        sample = {
            "joint_pos": joint_pos,
            "actions": actions,  # (seq_len, 7)
        }

        # Images if available
        if "images" in ep and len(ep["images"]) > t:
            img = torch.tensor(ep["images"][t], dtype=torch.float32)
            sample["images"] = img

        return sample


class SimplePolicy(nn.Module):
    """Simple action-prediction policy for fine-tuning.

    When Octo is not available, this provides a baseline architecture
    that can still learn from demonstrations via behavior cloning.
    """

    def __init__(self, config: FinetuneConfig):
        super().__init__()
        self.config = config

        # Image encoder (simple CNN if images available)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(12, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )  # Output: 128

        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(config.proprio_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Action decoder (predicts sequence of actions)
        self.decoder = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.sequence_length * config.action_dim),
        )

    def forward(self, images=None, joint_pos=None):
        features = []

        if images is not None:
            img_feat = self.image_encoder(images)
            features.append(img_feat)
        else:
            features.append(torch.zeros(joint_pos.shape[0], 128, device=joint_pos.device))

        if joint_pos is not None:
            proprio_feat = self.proprio_encoder(joint_pos)
            features.append(proprio_feat)

        combined = torch.cat(features, dim=-1)
        actions_flat = self.decoder(combined)
        return actions_flat.reshape(-1, self.config.sequence_length, self.config.action_dim)


def try_load_octo(model_name: str):
    """Attempt to load Octo model. Falls back to SimplePolicy if unavailable."""
    try:
        from octo.model.octo_model import OctoModel
        model = OctoModel.load_pretrained(f"hf://rail-berkeley/{model_name}")
        print(f"[Octo] Loaded pretrained {model_name}")
        return model, True
    except ImportError:
        print("[Octo] octo-model not installed, using SimplePolicy fallback")
        return None, False
    except Exception as e:
        print(f"[Octo] Failed to load {model_name}: {e}")
        print("[Octo] Using SimplePolicy fallback")
        return None, False


def train(config: FinetuneConfig):
    """Run fine-tuning."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # Load data
    dataset = DemoDataset(config.data_dir, config)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )

    # Try Octo, fall back to SimplePolicy
    octo_model, use_octo = try_load_octo(config.model_name)

    if use_octo:
        # Octo fine-tuning path
        print("[Train] Fine-tuning Octo (JAX-based, see Octo docs for details)")
        print("[Train] This path requires the octo library's native training loop")
        print("[Train] See: github.com/octo-model/octo/blob/main/scripts/finetune.py")
        return

    # SimplePolicy (PyTorch) fallback
    model = SimplePolicy(config).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.MSELoss()

    print(f"[Train] Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Train] Dataset: {len(dataset)} samples from {len(dataset.episodes)} episodes")
    print(f"[Train] Training for {config.epochs} epochs...")

    best_loss = float("inf")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            joint_pos = batch["joint_pos"].to(device)
            target_actions = batch["actions"].to(device)
            images = batch.get("images")
            if images is not None:
                images = images.to(device)

            pred_actions = model(images=images, joint_pos=joint_pos)
            loss = criterion(pred_actions, target_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, num_batches)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config.epochs}: loss={avg_loss:.6f}, lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(config.output_dir, exist_ok=True)
            save_path = os.path.join(config.output_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(config),
                "epoch": epoch,
                "loss": best_loss,
            }, save_path)

    # Save final model
    final_path = os.path.join(config.output_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(config),
        "epoch": config.epochs,
        "loss": avg_loss,
    }, final_path)

    print(f"\n[Train] Done. Best loss: {best_loss:.6f}")
    print(f"  Best model: {os.path.join(config.output_dir, 'best_model.pt')}")
    print(f"  Final model: {final_path}")

    # Export to ONNX
    onnx_path = os.path.join(config.output_dir, "policy.onnx")
    model.eval()
    dummy_images = torch.randn(1, 12, 224, 224, device=device)
    dummy_proprio = torch.randn(1, 7, device=device)
    torch.onnx.export(
        model, (dummy_images, dummy_proprio), onnx_path,
        opset_version=17,
        input_names=["images", "proprioception"],
        output_names=["actions"],
    )
    print(f"  ONNX export: {onnx_path}")
    print(f"  Size: {os.path.getsize(onnx_path) / (1024*1024):.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Octo on real demos")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="octo-small", choices=["octo-small", "octo-base"])
    parser.add_argument("--output", type=str, default="/ssd/alice/models/octo_finetuned")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--all-episodes", action="store_true", help="Use all episodes, not just successful")
    args = parser.parse_args()

    config = FinetuneConfig(
        data_dir=args.data_dir,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        success_only=not args.all_episodes,
    )
    train(config)
