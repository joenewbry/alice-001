#!/bin/bash
# Setup NoMachine + xfce4 on GCP VM for interactive Isaac Sim viewing.
#
# Run once on the VM:
#   bash scripts/setup_nomachine.sh
#
# Then from your Mac:
#   1. Install NoMachine client: https://www.nomachine.com/download
#   2. Connect to VM_EXTERNAL_IP:4000
#   3. In the NoMachine desktop, run:
#      python rl_task/scripts/play.py --checkpoint logs/.../model_5749.pt --num_envs 16
#   4. Full 3D viewport — orbit, zoom, watch real-time

set -euo pipefail

echo "=== Installing xfce4 desktop environment ==="
sudo apt-get update
sudo apt-get install -y xfce4 xfce4-goodies dbus-x11

echo ""
echo "=== Installing NoMachine ==="
NOMACHINE_DEB="nomachine_8.14.2_1_amd64.deb"
NOMACHINE_URL="https://download.nomachine.com/download/8.14/Linux/${NOMACHINE_DEB}"

if ! command -v /usr/NX/bin/nxserver &> /dev/null; then
    wget -q "${NOMACHINE_URL}" -O "/tmp/${NOMACHINE_DEB}"
    sudo dpkg -i "/tmp/${NOMACHINE_DEB}"
    rm "/tmp/${NOMACHINE_DEB}"
else
    echo "NoMachine already installed, skipping."
fi

echo ""
echo "=== Creating GCP firewall rule for NoMachine (port 4000) ==="
if gcloud compute firewall-rules describe allow-nomachine &> /dev/null 2>&1; then
    echo "Firewall rule 'allow-nomachine' already exists."
else
    gcloud compute firewall-rules create allow-nomachine \
        --allow tcp:4000 \
        --description "Allow NoMachine remote desktop" \
        --direction INGRESS \
        --priority 1000
    echo "Firewall rule created."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To connect:"
echo "  1. Get VM external IP: gcloud compute instances describe alice-isaaclab-2 --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)'"
echo "  2. Open NoMachine client on Mac"
echo "  3. Connect to <VM_IP>:4000 with your VM username/password"
echo "  4. In the desktop, open a terminal and run Isaac Sim"
