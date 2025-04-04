#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization script for comparing Betti numbers at a fixed scale
across different layers and positions in a Vision Transformer.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.gridspec import GridSpec

def load_results(json_path):
    """Load JSON results file"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results

def visualize_fixed_scale(results_path, scale="2.5", output_dir=None):
    """
    Create visualizations of Betti numbers at a fixed scale across layers.
    
    Args:
        results_path: Path to the results JSON file
        scale: Scale to visualize (as a string, e.g., "2.5")
        output_dir: Directory to save visualizations
    """
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    
    # Define components, positions and betti dimensions
    components = ['attention', 'mlp']
    positions = ['cls', 'central_patch', 'top_left_patch']
    betti_dims = [0, 1, 2]  # β₀, β₁, β₂
    
    # Process class0 only
    class_name = 'class0'
    
    # Determine number of layers from the data
    num_layers = len(results[class_name]['attention'])
    
    # Set up figure layouts
    for betti_dim in betti_dims:
        # Create figure with a grid of subplots
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'Betti-{betti_dim} Values at Scale {scale} Across Layers', fontsize=16)
        
        gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
        
        # Plot for component comparison (attention vs mlp) across positions
        ax1 = fig.add_subplot(gs[0])
        ax1.set_title(f'Component Comparison by Position', fontsize=14)
        
        # Colors for positions
        position_colors = {
            'cls': 'blue',
            'central_patch': 'green',
            'top_left_patch': 'red'
        }
        
        # For each position
        for position in positions:
            # Extract attention betti values across layers
            attention_betti = []
            for layer_idx in range(num_layers):
                if (position in results[class_name]['attention'][layer_idx] and 
                    scale in results[class_name]['attention'][layer_idx][position] and
                    len(results[class_name]['attention'][layer_idx][position][scale]) > betti_dim):
                    attention_betti.append(results[class_name]['attention'][layer_idx][position][scale][betti_dim])
                else:
                    attention_betti.append(0)
            
            # Extract mlp betti values across layers
            mlp_betti = []
            for layer_idx in range(num_layers):
                if (position in results[class_name]['mlp'][layer_idx] and 
                    scale in results[class_name]['mlp'][layer_idx][position] and
                    len(results[class_name]['mlp'][layer_idx][position][scale]) > betti_dim):
                    mlp_betti.append(results[class_name]['mlp'][layer_idx][position][scale][betti_dim])
                else:
                    mlp_betti.append(0)
            
            # Plot attention with solid lines
            ax1.plot(range(num_layers), attention_betti, color=position_colors[position], 
                    linestyle='-', marker='o', linewidth=2, 
                    label=f'{position} (Attention)')
            
            # Plot mlp with dashed lines
            ax1.plot(range(num_layers), mlp_betti, color=position_colors[position], 
                    linestyle='--', marker='s', linewidth=2, 
                    label=f'{position} (MLP)')
        
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel(f'Betti-{betti_dim} Value', fontsize=12)
        ax1.grid(True)
        ax1.legend(loc='upper right', fontsize=10)
        
        # Plot for position comparison (cls vs central_patch vs top_left_patch) across components
        ax2 = fig.add_subplot(gs[1])
        ax2.set_title(f'Position Comparison by Component', fontsize=14)
        
        # Colors for components
        component_colors = {
            'attention': 'purple',
            'mlp': 'orange'
        }
        
        # For each component
        for component in components:
            # For each position
            for position in positions:
                # Extract betti values across layers
                position_betti = []
                for layer_idx in range(num_layers):
                    if (position in results[class_name][component][layer_idx] and 
                        scale in results[class_name][component][layer_idx][position] and
                        len(results[class_name][component][layer_idx][position][scale]) > betti_dim):
                        position_betti.append(results[class_name][component][layer_idx][position][scale][betti_dim])
                    else:
                        position_betti.append(0)
                
                # Use different line styles for positions
                linestyle = '-' if position == 'cls' else '--' if position == 'central_patch' else ':'
                
                # Plot with component color and position-specific line style
                ax2.plot(range(num_layers), position_betti, color=component_colors[component], 
                        linestyle=linestyle, marker='o' if component == 'attention' else 's', 
                        linewidth=2, label=f'{component} ({position})')
        
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel(f'Betti-{betti_dim} Value', fontsize=12)
        ax2.grid(True)
        ax2.legend(loc='upper right', fontsize=10)
        
        # Save or show the figure
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f'betti{betti_dim}_scale{scale}.png'), dpi=300, bbox_inches='tight')
            print(f"Saved visualization to: {os.path.join(output_dir, f'betti{betti_dim}_scale{scale}.png')}")
        else:
            plt.show()
        
        plt.close(fig)
    
    print("Visualization complete!")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate visualizations of Betti numbers at a fixed scale')
    parser.add_argument('--results_path', type=str, required=True, help='Path to the results JSON file')
    parser.add_argument('--scale', type=str, default="2.5", help='Scale to visualize (default: 2.5)')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save visualizations')
    args = parser.parse_args()
    
    # Call the visualization function
    visualize_fixed_scale(args.results_path, args.scale, args.output_dir) 