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

def load_results(json_path):
    """Load JSON results file"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results

def visualize_betti_3d(output_dir='results_3d'):
    """
    Create 3D visualizations of Betti numbers.
    
    Args:
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define which positions to include
    positions = ['cls', 'central_patch', 'total_patches']
    
    # Process each activation function
    for activation in ['GELU', 'ReLU']:
        # Load results for this activation
        results_path = os.path.join('results', f'{activation}_results.json')
        
        # Check if file exists
        if not os.path.exists(results_path):
            print(f"Results file not found: {results_path}")
            continue
            
        results = load_results(results_path)
        
        # Create directory for this activation
        act_dir = os.path.join(output_dir, activation)
        os.makedirs(act_dir, exist_ok=True)
        
        # Process each class
        for class_name in ['class0', 'class1']:
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
                    ax.plot([0], [0], [0], color=cm.inferno(0.5), label='total_patches')
                    
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

def visualize_betti_curves(output_dir='results_betti_curves'):
    """
    Create plots showing Betti numbers across layers for each scale.
    
    Args:
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define which positions to include
    positions = ['cls', 'central_patch', 'total_patches']
    
    # Process each activation function
    for activation in ['GELU', 'ReLU']:
        # Load results for this activation
        results_path = os.path.join('results', f'{activation}_results.json')
        
        # Check if file exists
        if not os.path.exists(results_path):
            print(f"Results file not found: {results_path}")
            continue
            
        results = load_results(results_path)
        
        # Create directory for this activation
        act_dir = os.path.join(output_dir, activation)
        os.makedirs(act_dir, exist_ok=True)
        
        # Process each class
        for class_name in ['class0', 'class1']:
            # Create directory for this class
            class_dir = os.path.join(act_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Get all scales (convert string keys to float)
            all_scales = set()
            for component in ['attention', 'mlp']:
                num_layers = len(results[class_name][component])
                for layer_idx in range(num_layers):
                    for position in positions:
                        if position in results[class_name][component][layer_idx]:
                            all_scales.update(float(s) for s in results[class_name][component][layer_idx][position].keys())
            scales = sorted(all_scales)
            
            # Process each position
            for position in positions:
                # Create a directory for this position
                pos_dir = os.path.join(class_dir, position)
                os.makedirs(pos_dir, exist_ok=True)
                
                # For each Betti dimension (0, 1, 2)
                for betti_dim in range(3):
                    # Create figure
                    plt.figure(figsize=(15, 10))
                    
                    # Process each component (attention and mlp)
                    for component in ['attention', 'mlp']:
                        num_layers = len(results[class_name][component])
                        
                        # For each scale, plot its Betti numbers across layers
                        for scale in scales:
                            scale_str = str(float(scale))
                            
                            # Extract Betti numbers across layers for this scale
                            betti_values = []
                            for layer_idx in range(num_layers):
                                if (position in results[class_name][component][layer_idx] and 
                                    scale_str in results[class_name][component][layer_idx][position] and
                                    len(results[class_name][component][layer_idx][position][scale_str]) > betti_dim):
                                    betti_values.append(results[class_name][component][layer_idx][position][scale_str][betti_dim])
                                else:
                                    betti_values.append(0)
                            
                            # Plot this scale's values across layers
                            plt.plot(range(len(betti_values)), betti_values, 
                                     label=f"{component}, scale={float(scale):.2f}", alpha=0.7)
                    
                    plt.xlabel('Layer')
                    plt.ylabel(f'Betti Number β{betti_dim}')
                    plt.title(f'{activation} - {class_name} - {position} - β{betti_dim}')
                    plt.grid(True)
                    
                    # Only add legend if there are 10 or fewer scales, otherwise too cluttered
                    if len(scales) <= 10:
                        plt.legend()
                    
                    # Save figure
                    plt.savefig(os.path.join(pos_dir, f"betti{betti_dim}_curve.png"), dpi=300)
                    plt.close()
                    
                    print(f"Created Betti curve for {activation} - {class_name} - {position} - β{betti_dim}")

if __name__ == "__main__":
    print("Generating 3D visualizations...")
    visualize_betti_3d()
    
    print("\nGenerating Betti curves...")
    visualize_betti_curves()
    
    print("\nVisualization complete!") 