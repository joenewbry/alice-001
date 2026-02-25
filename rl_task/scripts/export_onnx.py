"""Export trained policy to ONNX for TensorRT deployment on Jetson.

Supports both state-only (Stage 1) and vision (Stage 2/3) policies.

Usage:
    # Export state-only policy
    python ~/Alice-001/rl_task/scripts/export_onnx.py \
        --checkpoint ~/Alice-001/logs/alice_ball_transfer/model_5000.pt \
        --mode state \
        --output ~/Alice-001/exports/policy_state.onnx

    # Export vision policy (actor only — ResNet + MLP)
    python ~/Alice-001/rl_task/scripts/export_onnx.py \
        --checkpoint ~/Alice-001/logs/alice_ball_transfer_vision/model_10000.pt \
        --mode vision \
        --output ~/Alice-001/exports/policy_vision.onnx

After export, transfer to Jetson and convert to TensorRT:
    scp exports/policy_vision.onnx prometheus:/ssd/alice/models/
    ssh prometheus "trtexec --onnx=/ssd/alice/models/policy_vision.onnx \
        --saveEngine=/ssd/alice/models/policy_vision.engine --fp16"
"""

import argparse
import os

import torch
import torch.nn as nn


class StatePolicy(nn.Module):
    """Wrapper for state-only MLP policy export."""

    def __init__(self, actor_net, num_obs=28, num_actions=7):
        super().__init__()
        self.actor = actor_net
        self.num_obs = num_obs
        self.num_actions = num_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class VisionPolicy(nn.Module):
    """Full vision pipeline: ResNet-18 backbone + MLP head.

    For deployment, the backbone is baked into the ONNX model so TensorRT
    can optimize the entire pipeline end-to-end.
    """

    def __init__(self, backbone, actor_net, num_frames=4):
        super().__init__()
        self.backbone = backbone
        self.actor = actor_net
        self.num_frames = num_frames

    def forward(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (1, num_frames*3, 224, 224) stacked camera frames
            proprio: (1, 14) joint positions + velocities

        Returns:
            actions: (1, 7) joint velocity targets
        """
        visual_feats = self.backbone(images)  # (1, 512)
        obs = torch.cat([visual_feats, proprio], dim=-1)  # (1, 526)
        return self.actor(obs)


def load_rsl_rl_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load RSL-RL checkpoint and extract actor network."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # RSL-RL saves model_state_dict with actor and critic
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Extract actor weights (keys start with "actor.")
    actor_state = {}
    for k, v in state_dict.items():
        if k.startswith("actor."):
            actor_state[k[len("actor."):]] = v

    return actor_state, checkpoint


def build_actor_mlp(hidden_dims: list, input_dim: int, output_dim: int, activation: str = "elu"):
    """Reconstruct actor MLP from config."""
    act_fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

    layers = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(act_fn())
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)


def export_state_policy(checkpoint_path: str, output_path: str):
    """Export state-only policy to ONNX."""
    actor_state, _ = load_rsl_rl_checkpoint(checkpoint_path)

    # Reconstruct actor MLP: [256, 256, 128] hidden dims
    actor = build_actor_mlp([256, 256, 128], input_dim=28, output_dim=7)

    # Load weights
    actor.load_state_dict(actor_state)
    actor.eval()

    policy = StatePolicy(actor)
    dummy_obs = torch.randn(1, 28)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        policy,
        dummy_obs,
        output_path,
        opset_version=17,
        input_names=["observations"],
        output_names=["actions"],
        dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
    )
    print(f"Exported state policy to {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024:.1f} KB")


def export_vision_policy(checkpoint_path: str, output_path: str, num_frames: int = 4):
    """Export vision policy (ResNet + MLP) to ONNX."""
    from alice_ball_transfer.models.vision_backbone import VisionBackbone

    actor_state, _ = load_rsl_rl_checkpoint(checkpoint_path)

    # ResNet-18 backbone
    backbone = VisionBackbone(num_frames=num_frames)

    # Actor MLP: [256, 128] hidden dims, input = 512 + 14 = 526
    actor = build_actor_mlp([256, 128], input_dim=526, output_dim=7)
    actor.load_state_dict(actor_state)
    actor.eval()

    policy = VisionPolicy(backbone, actor, num_frames=num_frames)

    # Dummy inputs
    dummy_images = torch.randn(1, num_frames * 3, 224, 224)
    dummy_proprio = torch.randn(1, 14)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        policy,
        (dummy_images, dummy_proprio),
        output_path,
        opset_version=17,
        input_names=["images", "proprioception"],
        output_names=["actions"],
        dynamic_axes={
            "images": {0: "batch"},
            "proprioception": {0: "batch"},
            "actions": {0: "batch"},
        },
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported vision policy to {output_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Includes: ResNet-18 backbone + MLP actor")
    print(f"\nNext steps:")
    print(f"  scp {output_path} prometheus:/ssd/alice/models/")
    print(f'  ssh prometheus "trtexec --onnx=/ssd/alice/models/{os.path.basename(output_path)} '
          f'--saveEngine=/ssd/alice/models/policy_vision.engine --fp16"')


def main():
    parser = argparse.ArgumentParser(description="Export Alice policy to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--mode", choices=["state", "vision"], required=True)
    parser.add_argument("--output", type=str, required=True, help="Output .onnx path")
    parser.add_argument("--num_frames", type=int, default=4, help="Frame stack size (vision only)")
    args = parser.parse_args()

    if args.mode == "state":
        export_state_policy(args.checkpoint, args.output)
    else:
        export_vision_policy(args.checkpoint, args.output, args.num_frames)


if __name__ == "__main__":
    main()
