#!/bin/bash
# Run WHAM with LIVE visualization (requires display)
# Use this on an interactive node with X11 forwarding

echo "Starting LIVE real-time WHAM visualization..."
echo "Note: Requires X11 forwarding (ssh -X) or VNC session"
echo ""

conda activate wham

# This will show a live window as it processes
python realtime_wham_online.py \
    examples/IMG_9732_portrait.mov \
    --visualize \
    --frame-skip 2 \
    --duration 20

echo ""
echo "Press 'q' in the window to stop"

