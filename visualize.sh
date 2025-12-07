#!/bin/bash

# Make sure the output directory exists
mkdir -p /home/tcong13/949Final/vis

# Run the visualization script
python3 visualize_stdp.py --frames 50 --neurons 100 --output /home/tcong13/949Final/vis/stdp_test.mp4