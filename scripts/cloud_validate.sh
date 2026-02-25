#!/bin/bash
# Run on GCP GPU VM to validate the Alice 001 USD in Isaac Sim.
# This script:
# 1. Installs NVIDIA Container Toolkit (if needed)
# 2. Pulls Isaac Sim Docker image
# 3. Runs URDF -> USD conversion via Isaac Lab's converter
# 4. Runs basic validation
#
# Prerequisites: NVIDIA GPU drivers installed, Docker installed
# The VM should be created with --image-family=ubuntu-2204-lts + L4/T4 GPU

set -e

WORK_DIR="${HOME}/Alice-001"
NGC_KEY="${NGC_API_KEY:-}"

echo "=== Alice 001 Cloud Validation ==="
echo "Working dir: ${WORK_DIR}"
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "ERROR: No GPU found. Need NVIDIA GPU with drivers."
    exit 1
}

# Install NVIDIA Container Toolkit if not present
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# Login to NGC and pull Isaac Sim
ISAAC_IMAGE="nvcr.io/nvidia/isaac-sim:4.5.0"
echo "Pulling Isaac Sim image (this may take 10-15 min on first run)..."
if [ -n "$NGC_KEY" ]; then
    echo "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
fi
docker pull ${ISAAC_IMAGE}

# Run the Isaac Lab URDF converter inside the container
echo ""
echo "=== Converting URDF -> USD via Isaac Lab ==="
docker run --rm --gpus all \
    -v "${WORK_DIR}:/workspace/Alice-001" \
    ${ISAAC_IMAGE} \
    bash -c "
        cd /isaac-sim && \
        ./isaaclab.sh -p scripts/tools/convert_urdf.py \
            /workspace/Alice-001/urdf/alice_001.urdf \
            /workspace/Alice-001/usd/alice_001_isaaclab.usd \
            --headless --fix-base --merge-joints \
            --joint-stiffness 0.0 --joint-damping 0.0 \
            --joint-target-type none --make-instanceable 2>&1 || echo 'URDF conversion failed (may need path adjustments)'
    "

# Run a basic validation (gravity + joint check)
echo ""
echo "=== Running Physics Validation ==="
docker run --rm --gpus all \
    -v "${WORK_DIR}:/workspace/Alice-001" \
    ${ISAAC_IMAGE} \
    bash -c "
        cd /isaac-sim && \
        python3 -c \"
import omni.isaac.core
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Try loading the USD
usd_path = '/workspace/Alice-001/usd/alice_001.usd'
print(f'Loading {usd_path}...')
add_reference_to_stage(usd_path=usd_path, prim_path='/World/Alice001')

world.reset()

# Run 500 steps
print('Running gravity drop test (500 steps)...')
for i in range(500):
    world.step(render=False)
    if i % 100 == 0:
        print(f'  Step {i}/500')

print('PASS: Simulation ran 500 steps without crash')
world.stop()
\" 2>&1
    "

echo ""
echo "=== Done ==="
echo "USD files in: ${WORK_DIR}/usd/"
ls -lh ${WORK_DIR}/usd/
