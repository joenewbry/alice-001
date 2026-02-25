#!/bin/bash
# Convert Alice 001 URDF to USD using Isaac Lab's converter.
# Run this on the cloud GPU instance after uploading the project.
#
# Usage:
#   cd /path/to/Alice-001
#   bash scripts/convert_urdf_to_usd.sh
#
# Or via isaaclab.sh:
#   ./isaaclab.sh -p scripts/tools/convert_urdf.py \
#     /path/to/Alice-001/urdf/alice_001.urdf \
#     /path/to/Alice-001/usd/alice_001.usd \
#     --headless --fix-base --merge-joints \
#     --joint-stiffness 0.0 --joint-damping 0.0 \
#     --joint-target-type none --make-instanceable

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

URDF_PATH="${PROJECT_DIR}/urdf/alice_001.urdf"
USD_PATH="${PROJECT_DIR}/usd/alice_001.usd"

echo "Converting URDF -> USD"
echo "  Input:  ${URDF_PATH}"
echo "  Output: ${USD_PATH}"

# Find isaaclab.sh
if [ -f "./isaaclab.sh" ]; then
    ISAACLAB="./isaaclab.sh"
elif [ -f "${ISAAC_LAB_PATH}/isaaclab.sh" ]; then
    ISAACLAB="${ISAAC_LAB_PATH}/isaaclab.sh"
else
    echo "ERROR: isaaclab.sh not found. Set ISAAC_LAB_PATH or run from Isaac Lab root."
    exit 1
fi

${ISAACLAB} -p scripts/tools/convert_urdf.py \
    "${URDF_PATH}" \
    "${USD_PATH}" \
    --headless \
    --fix-base \
    --merge-joints \
    --joint-stiffness 0.0 \
    --joint-damping 0.0 \
    --joint-target-type none \
    --make-instanceable

echo ""
echo "Conversion complete!"
echo "  USD file: ${USD_PATH}"
echo ""
echo "Next: run validation"
echo "  ${ISAACLAB} -p ${PROJECT_DIR}/scripts/validate.py --test all --headless"
