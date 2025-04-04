#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D visualization for topology analysis results.
Plots the attention and MLP Betti numbers on the same axis with:
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

def visualize_betti_3d_combined(results_dir=None, output_dir=None):
    """
    Create 3D visualizations of Betti numbers with attention and MLP on the same axis.
    
    Args:
        results_dir: Directory containing the results
        output_dir: Directory to save visualizations
    """
    # Create timestamped output directory if none provided
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'results_3d_combined_{timestamp}'
    
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
        
        # Process each position
        for position in positions:
            # Process each Betti dimension (0, 1, 2)
            for betti_dim in range(3):
                # Create 3D figure
                fig = plt.figure(figsize=(14, 12))
                ax = fig.add_subplot(111, projection='3d')
                
                # Define colors for components
                component_colors = {
                    'attention': cm.cool,
                    'mlp': cm.autumn
                }
                
                # Get max layer count across components
                num_layers = max(len(results[class_name]['attention']), len(results[class_name]['mlp']))
                
                # Get all scales (convert string keys to float) across both components
                all_scales = set()
                for component in ['attention', 'mlp']:
                    for layer_idx in range(len(results[class_name][component])):
                        if position in results[class_name][component][layer_idx]:
                            all_scales.update(float(s) for s in results[class_name][component][layer_idx][position].keys())
                scales = sorted(all_scales)
                
                # Plot each component on the same axis
                for component_idx, component in enumerate(['attention', 'mlp']):
                    # Set up meshgrid for 3D surface
                    component_layers = len(results[class_name][component])
                    X, Y = np.meshgrid(range(component_layers), range(len(scales)))
                    
                    # Create Z matrix (betti values)
                    Z = np.zeros((len(scales), component_layers))
                    
                    # Fill Z with betti values
                    for layer_idx in range(component_layers):
                        layer_data = results[class_name][component][layer_idx]
                        if position in layer_data:
                            for scale_idx, scale in enumerate(scales):
                                scale_str = str(float(scale))
                                if scale_str in layer_data[position]:
                                    betti_values = layer_data[position][scale_str]
                                    if betti_dim < len(betti_values):
                                        Z[scale_idx, layer_idx] = betti_values[betti_dim]
                    
                    # Plot surface with component-specific color map
                    color_map = component_colors[component]
                    surf = ax.plot_surface(X, Y, Z, alpha=0.7, label=component,
                                          cmap=color_map)
                    surf._facecolors2d = surf._facecolor3d
                    surf._edgecolors2d = surf._edgecolor3d
                
                # Custom legend for components
                ax.plot([0], [0], [0], color=component_colors['attention'](0.5), label='Attention')
                ax.plot([0], [0], [0], color=component_colors['mlp'](0.5), label='MLP')
                
                # Set labels and title
                ax.set_xlabel('Layers')
                ax.set_ylabel('Scale Index')
                ax.set_zlabel(f'Betti Number β{betti_dim}')
                ax.set_title(f'{activation} - {class_name} - {position} - β{betti_dim} (Attention & MLP Combined)')
                
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
                plt.savefig(os.path.join(class_dir, f"combined_{position}_betti{betti_dim}_3d.png"), dpi=300)
                plt.close()
                
                print(f"Created combined 3D visualization for {activation} - {class_name} - {position} - β{betti_dim}")
                
                # Also create 2D heatmap comparison view (side by side)
                fig, axes = plt.subplots(1, 2, figsize=(18, 8))
                
                for comp_idx, component in enumerate(['attention', 'mlp']):
                    component_layers = len(results[class_name][component])
                    Z = np.zeros((len(scales), component_layers))
                    
                    # Fill Z with betti values
                    for layer_idx in range(component_layers):
                        layer_data = results[class_name][component][layer_idx]
                        if position in layer_data:
                            for scale_idx, scale in enumerate(scales):
                                scale_str = str(float(scale))
                                if scale_str in layer_data[position]:
                                    betti_values = layer_data[position][scale_str]
                                    if betti_dim < len(betti_values):
                                        Z[scale_idx, layer_idx] = betti_values[betti_dim]
                    
                    # Create heatmap
                    im = axes[comp_idx].imshow(Z, aspect='auto', 
                                              cmap=component_colors[component], 
                                              interpolation='nearest')
                    plt.colorbar(im, ax=axes[comp_idx], label=f'Betti Number β{betti_dim}')
                    
                    # Set labels and title
                    axes[comp_idx].set_xlabel('Layers')
                    axes[comp_idx].set_ylabel('Scale Index')
                    axes[comp_idx].set_title(f'{component.upper()} - {position} - β{betti_dim}')
                    
                    # Set y-ticks to actual scale values (show a subset for readability)
                    if len(scales) > 10:
                        tick_indices = np.linspace(0, len(scales)-1, 5, dtype=int)
                        axes[comp_idx].set_yticks(tick_indices)
                        axes[comp_idx].set_yticklabels([f"{scales[i]:.2f}" for i in tick_indices])
                    else:
                        axes[comp_idx].set_yticks(range(len(scales)))
                        axes[comp_idx].set_yticklabels([f"{s:.2f}" for s in scales])
                
                plt.suptitle(f'{activation} - {class_name} - {position} - β{betti_dim} Comparison')
                plt.tight_layout()
                
                # Save figure
                plt.savefig(os.path.join(class_dir, f"comparison_{position}_betti{betti_dim}_heatmap.png"), dpi=300)
                plt.close()
                
                print(f"Created comparison heatmap for {activation} - {class_name} - {position} - β{betti_dim}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate 3D visualizations of topology analysis results with attention and MLP on the same axis')
    parser.add_argument('--results_dir', type=str, help='Directory containing the results')
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'results_3d_combined_{timestamp}'
    
    # Determine results directory
    results_dir = args.results_dir if args.results_dir else 'results'
    
    # Visualize results
    visualize_betti_3d_combined(results_dir, output_dir)
    
    print(f"Visualization complete. Results saved in '{output_dir}' directory.") 