#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D visualization for topology analysis results.
Plots the Betti numbers with:
- X-axis: Layers
- Y-axis: Scales
- Z-axis: Betti number values
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import datetime
import argparse

def load_results(json_path):
    """Load JSON results file"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results

def visualize_betti_3d(results_dir=None, output_dir=None):
    """
    Create 3D visualizations of Betti numbers.
    
    Args:
        results_dir: Directory containing the results
        output_dir: Directory to save visualizations
    """
    # Create timestamped output directory if none provided
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'results_3d_{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading results from: {results_dir}")
    print(f"Saving visualizations to: {output_dir}")
    
    # Define which positions to include
    positions = ['cls', 'central_patch', 'top_left_patch']
    
    # Process each activation function
    for activation in ['GELU', 'ReLU']:
        # Load results for this activation
        results_path = os.path.join(results_dir, f'{activation}_results.json')
        results = load_results(results_path)
        
        # Create directory for this activation
        act_dir = os.path.join(output_dir, activation)
        os.makedirs(act_dir, exist_ok=True)
        
        # Process only class0
        class_name = 'class0'
        
        # Create directory for this class
        class_dir = os.path.join(act_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Process each component (attention and mlp)
        for component in ['attention', 'mlp']:
            # Process each Betti dimension (0, 1, 2)
            for betti_dim in range(3):
                # Create 3D figure
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # Extract layer count and scales
                num_layers = len(results[class_name][component])
                
                # Get all scales (convert string keys to float)
                all_scales = set()
                for layer_idx in range(num_layers):
                    for position in positions:
                        if position in results[class_name][component][layer_idx]:
                            all_scales.update(float(s) for s in results[class_name][component][layer_idx][position].keys())
                scales = sorted(all_scales)
                
                # Set up meshgrid for 3D surface
                X, Y = np.meshgrid(range(num_layers), range(len(scales)))
                
                # Process each position
                for position_idx, position in enumerate(positions):
                    # Create Z matrix (betti values)
                    Z = np.zeros((len(scales), num_layers))
                    
                    # Fill Z with betti values
                    for layer_idx in range(num_layers):
                        layer_data = results[class_name][component][layer_idx]
                        if position in layer_data:
                            for scale_idx, scale in enumerate(scales):
                                scale_str = str(float(scale))
                                if scale_str in layer_data[position]:
                                    betti_values = layer_data[position][scale_str]
                                    if betti_dim < len(betti_values):
                                        Z[scale_idx, layer_idx] = betti_values[betti_dim]
                    
                    # Plot surface
                    surf = ax.plot_surface(X, Y, Z, alpha=0.7, label=position,
                                          cmap=cm.viridis if position_idx == 0 else 
                                               cm.plasma if position_idx == 1 else cm.inferno)
                    surf._facecolors2d = surf._facecolor3d
                    surf._edgecolors2d = surf._edgecolor3d
                
                # Custom legend for positions
                ax.plot([0], [0], [0], color=cm.viridis(0.5), label='cls')
                ax.plot([0], [0], [0], color=cm.plasma(0.5), label='central_patch')
                ax.plot([0], [0], [0], color=cm.inferno(0.5), label='top_left_patch')
                
                # Set labels and title
                ax.set_xlabel('Layers')
                ax.set_ylabel('Scale Index')
                ax.set_zlabel(f'Betti Number β{betti_dim}')
                ax.set_title(f'{activation} - {class_name} - {component} - β{betti_dim}')
                
                # Set y-ticks to actual scale values (show a subset for readability)
                if len(scales) > 10:
                    tick_indices = np.linspace(0, len(scales)-1, 5, dtype=int)
                    ax.set_yticks(tick_indices)
                    ax.set_yticklabels([f"{scales[i]:.2f}" for i in tick_indices])
                else:
                    ax.set_yticks(range(len(scales)))
                    ax.set_yticklabels([f"{s:.2f}" for s in scales])
                
                ax.legend()
                
                # Save figure
                plt.savefig(os.path.join(class_dir, f"{component}_betti{betti_dim}_3d.png"), dpi=300)
                plt.close()
                
                print(f"Created 3D visualization for {activation} - {class_name} - {component} - β{betti_dim}")
                
                # Also create 2D heatmap view
                for position in positions:
                    plt.figure(figsize=(12, 8))
                    
                    # Create Z matrix (betti values)
                    Z = np.zeros((len(scales), num_layers))
                    
                    # Fill Z with betti values
                    for layer_idx in range(num_layers):
                        layer_data = results[class_name][component][layer_idx]
                        if position in layer_data:
                            for scale_idx, scale in enumerate(scales):
                                scale_str = str(float(scale))
                                if scale_str in layer_data[position]:
                                    betti_values = layer_data[position][scale_str]
                                    if betti_dim < len(betti_values):
                                        Z[scale_idx, layer_idx] = betti_values[betti_dim]
                    
                    # Create heatmap
                    plt.imshow(Z, aspect='auto', cmap='viridis', interpolation='nearest')
                    plt.colorbar(label=f'Betti Number β{betti_dim}')
                    
                    # Set labels and title
                    plt.xlabel('Layers')
                    plt.ylabel('Scale Index')
                    plt.title(f'{activation} - {class_name} - {component} - {position} - β{betti_dim}')
                    
                    # Set y-ticks to actual scale values (show a subset for readability)
                    if len(scales) > 10:
                        tick_indices = np.linspace(0, len(scales)-1, 5, dtype=int)
                        plt.yticks(tick_indices, [f"{scales[i]:.2f}" for i in tick_indices])
                    else:
                        plt.yticks(range(len(scales)), [f"{s:.2f}" for s in scales])
                    
                    # Save figure
                    plt.savefig(os.path.join(class_dir, f"{component}_{position}_betti{betti_dim}_heatmap.png"), dpi=300)
                    plt.close()
                    
                    print(f"Created heatmap for {activation} - {class_name} - {component} - {position} - β{betti_dim}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate 3D visualizations of topology analysis results')
    parser.add_argument('--results_dir', type=str, help='Directory containing the results')
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'results_3d_{timestamp}'
    
    # Determine results directory
    results_dir = args.results_dir if args.results_dir else 'results'
    
    # Visualize results
    visualize_betti_3d(results_dir, output_dir)
    
    print(f"Visualization complete. Results saved in '{output_dir}' directory.") 