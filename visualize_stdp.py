import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import Normalize
from tqdm import tqdm
from pathlib import Path
import imageio.v2 as imageio
import argparse
import pandas as pd
import os

from SNN import STDPReservoir

def generate_patterns():
    """Generate two simple synthetic spike patterns"""
    # Pattern 1: diagonal spikes
    pattern1 = torch.zeros(1, 10)
    for i in range(10):
        if i % 3 == 0:  # Create diagonal-like pattern
            pattern1[0, i] = 1.0
    
    # Pattern 2: alternating spikes
    pattern2 = torch.zeros(1, 10)
    for i in range(10):
        if i % 2 == 0:  # Create alternating pattern
            pattern2[0, i] = 1.0
    
    return pattern1, pattern2

def visualize_stdp_evolution(output_path="stdp_evolution.mp4", num_frames=100, num_neurons=100, dpi=100):
    """
    Create a simple standalone visualization of STDP weight matrix evolution
    
    Args:
        output_path: Path to save the output video
        num_frames: Number of frames to capture
        num_neurons: Number of neurons in the reservoir
        dpi: Resolution of the output video
    """
    # Create STDP reservoir
    print(f"Creating STDP reservoir with {num_neurons} neurons...")
    reservoir = STDPReservoir(n_in=10, n_reservoir=num_neurons)
    
    # Generate synthetic patterns
    pattern1, pattern2 = generate_patterns()
    print(f"Generated synthetic patterns: {pattern1.sum().item()} spikes and {pattern2.sum().item()} spikes")
    
    # Track initial weights
    initial_weights = reservoir.W_rec.clone().detach().cpu().numpy()
    weights_min = initial_weights.min()
    weights_max = initial_weights.max()
    norm = Normalize(weights_min, weights_max)
    
    # Initialize weight change tracking
    max_weight_change = 0
    max_change_indices = (0, 0)
    total_weight_change = 0
    weight_changes_over_time = []
    
    # Create temporary directory for frames
    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    print(f"Saving temporary frames to {temp_dir}")
    
    # Store frames
    frames = []
    print(f"Generating {num_frames} frames for STDP weight evolution...")
    
    # Create figure for animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Run and capture frames
    for i in tqdm(range(num_frames)):
        # Alternate between patterns
        input_pattern = pattern1 if i % 2 == 0 else pattern2
        
        # Forward pass to trigger STDP
        reservoir(input_pattern)
        
        # Capture current weights
        current_weights = reservoir.W_rec.detach().cpu().numpy()
        
        # Compute weight changes from initial state
        weight_changes = np.abs(current_weights - initial_weights)
        
        # Track weight changes
        frame_max_change = weight_changes.max()
        frame_max_idx = np.unravel_index(weight_changes.argmax(), weight_changes.shape)
        frame_total_change = weight_changes.sum()
        
        # Track the largest change across all frames
        if frame_max_change > max_weight_change:
            max_weight_change = frame_max_change
            max_change_indices = frame_max_idx
        
        # Update total weight change
        total_weight_change = frame_total_change
        
        # Add to tracking list
        weight_changes_over_time.append({
            'frame': i,
            'max_change': frame_max_change,
            'max_idx': frame_max_idx,
            'total_change': frame_total_change,
            'value_at_max': current_weights[frame_max_idx]
        })
        
        # Print weight change information every 10 frames
        if i % 10 == 0 or i == num_frames - 1:
            print(f"Frame {i+1}/{num_frames} - Max change: {frame_max_change:.6f} at {frame_max_idx}, Total change: {frame_total_change:.6f}")
            print(f"  Top 3 changed connections:")
            # Find top 3 changes
            flat_indices = np.argsort(weight_changes.flatten())[-3:]
            for rank, idx in enumerate(flat_indices[::-1]):
                row, col = np.unravel_index(idx, weight_changes.shape)
                change_val = weight_changes[row, col]
                current_val = current_weights[row, col]
                print(f"  {rank+1}. ({row},{col}): Change={change_val:.6f}, Current={current_val:.6f}")
            print()
        
        # Plot the weight matrix
        ax1.clear()
        im1 = ax1.imshow(current_weights, cmap='viridis', norm=norm)
        ax1.set_title(f'STDP Weights - Frame {i+1}/{num_frames}')
        ax1.set_xlabel('Neuron (Post-Synaptic)')
        ax1.set_ylabel('Neuron (Pre-Synaptic)')
        
        # Plot weight changes
        ax2.clear()
        im2 = ax2.imshow(weight_changes, cmap='hot', vmin=0)
        ax2.set_title(f'Weight Changes from Initial')
        ax2.set_xlabel('Neuron (Post-Synaptic)')
        
        # Add colorbar if first frame
        if i == 0:
            plt.colorbar(im1, ax=ax1, label='Weight Strength')
            plt.colorbar(im2, ax=ax2, label='Absolute Change')
            
        plt.tight_layout()
        
        # Save the frame
        frame_path = temp_path / f"frame_{i:04d}.png"
        plt.savefig(frame_path, dpi=dpi)
        frames.append(str(frame_path))
    
    plt.close(fig)
    
    print(f"Creating video at {output_path}...")
    # Create video from frames with FFMPEG format
    writer = imageio.get_writer(output_path, format='FFMPEG', fps=10, quality=8)
    for frame_path in frames:
        image = imageio.imread(frame_path)
        writer.append_data(image)
    writer.close()
    
    print(f"STDP weight evolution video saved to {output_path}")
    
    # Print final weight change summary
    print("\n=== STDP Weight Change Summary ===")
    print(f"Initial weights: min={initial_weights.min():.6f}, max={initial_weights.max():.6f}, mean={initial_weights.mean():.6f}")
    
    # Get final weights
    final_weights = reservoir.W_rec.detach().cpu().numpy()
    print(f"Final weights: min={final_weights.min():.6f}, max={final_weights.max():.6f}, mean={final_weights.mean():.6f}")
    
    # Overall changes
    overall_changes = np.abs(final_weights - initial_weights)
    print(f"Maximum weight change: {max_weight_change:.6f} at {max_change_indices}")
    print(f"Total weight change: {total_weight_change:.6f}")
    
    # Find neurons with most change (summing across all connections)
    presynaptic_changes = overall_changes.sum(axis=1)  # Sum across rows
    postsynaptic_changes = overall_changes.sum(axis=0)  # Sum across columns
    
    pre_max_idx = np.argmax(presynaptic_changes)
    post_max_idx = np.argmax(postsynaptic_changes)
    
    print(f"Most changed presynaptic neuron: {pre_max_idx} (total change: {presynaptic_changes[pre_max_idx]:.6f})")
    print(f"Most changed postsynaptic neuron: {post_max_idx} (total change: {postsynaptic_changes[post_max_idx]:.6f})")
    print("==============================\n")
    
    # Export weight changes to CSV
    csv_path = os.path.splitext(output_path)[0] + "_changes.csv"
    print(f"Exporting weight change data to {csv_path}")
    
    # Create a DataFrame with weight change metrics over time
    df_changes = pd.DataFrame(weight_changes_over_time)
    df_changes.to_csv(csv_path, index=False)
    
    # Also export the final top 10 changed weights
    flat_indices = np.argsort(overall_changes.flatten())[-10:]
    top_changes = []
    
    for idx in flat_indices[::-1]:
        row, col = np.unravel_index(idx, overall_changes.shape)
        top_changes.append({
            'pre_neuron': row,
            'post_neuron': col,
            'initial_weight': initial_weights[row, col],
            'final_weight': final_weights[row, col],
            'abs_change': overall_changes[row, col],
            'relative_change': overall_changes[row, col] / (abs(initial_weights[row, col]) + 1e-10)
        })
    
    # Save top changes to CSV
    top_changes_csv = os.path.splitext(output_path)[0] + "_top10_changes.csv"
    pd.DataFrame(top_changes).to_csv(top_changes_csv, index=False)
    print(f"Exported top 10 weight changes to {top_changes_csv}")
    
    # Clean up temporary files
    for frame_path in frames:
        Path(frame_path).unlink()
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="/home/tcong13/949Final/vis/stdp_evolution.mp4", help="Output video file path")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to capture")
    parser.add_argument("--neurons", type=int, default=100, help="Number of neurons in reservoir")
    parser.add_argument("--dpi", type=int, default=100, help="Video resolution (DPI)")
    args = parser.parse_args()
    
    # Ensure the output directory exists
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        output_path = visualize_stdp_evolution(
            output_path=args.output, 
            num_frames=args.frames, 
            num_neurons=args.neurons, 
            dpi=args.dpi
        )
        print(f"Successfully saved video to: {output_path}")
    except Exception as e:
        print(f"Error generating video: {str(e)}")
        import traceback
        traceback.print_exc()
