#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D visualization for topology analysis results integrated with the vit_topology_combined_main.py workflow.
Plots the Betti numbers with:
- X-axis: Layers
- Y-axis: Scales
- Z-axis: Betti number values

This version is designed to be imported and called directly from the main workflow.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import datetime

def load_results(json_path):
    """Load JSON results file"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results

def create_3d_visualization(results, scales, activation_name, method='euclidean', output_dir=None):
    """
    Create 3D visualizations of Betti numbers for a specific activation function.
    
    Args:
        results: Dictionary containing the topology analysis results
        scales: List of scales used for the analysis
        activation_name: Name of the activation function (e.g., 'GELU', 'ReLU')
        method: Distance method used ('euclidean' or 'geodesic')
        output_dir: Directory to save visualizations
    
    Returns:
        Directory path where visualizations were saved
    """
    # Create timestamped output directory if none provided
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'3d_vis_{activation_name}_{method}_{timestamp}'
    
    # Create directory for this activation
    vis_dir = os.path.join(output_dir, f'{activation_name}_{method}_3d')
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Creating 3D visualizations for {activation_name} with {method} distance...")
    print(f"Saving to: {vis_dir}")
    
    # Define which positions to include
    positions = ['cls', 'central_patch', 'top_left_patch']
    
    # Process only class0
    class_name = 'class0'
    
    # Process each component (attention and mlp)
    for component in ['attention', 'mlp']:
        # Process each Betti dimension (0, 1, 2)
        for betti_dim in range(3):
            # Create 3D figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract layer count
            num_layers = len(results[class_name][component])
            
            # Use the provided scales
            # Convert to a list if it's not already
            if not isinstance(scales, list):
                scales = list(scales)
            
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
                            if scale in layer_data[position]:
                                betti_values = layer_data[position][scale]
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
            ax.set_title(f'{activation_name} - {class_name} - {component} - β{betti_dim}')
            
            # Set y-ticks to actual scale values (show a subset for readability)
            if len(scales) > 10:
                tick_indices = np.linspace(0, len(scales)-1, 5, dtype=int)
                ax.set_yticks(tick_indices)
                ax.set_yticklabels([f"{scales[int(i)]:.2f}" for i in tick_indices])
            else:
                ax.set_yticks(range(len(scales)))
                ax.set_yticklabels([f"{s:.2f}" for s in scales])
            
            ax.legend()
            
            # Save figure
            plt.savefig(os.path.join(vis_dir, f"{component}_betti{betti_dim}_3d.png"), dpi=300)
            plt.close()
            
            print(f"Created 3D visualization for {activation_name} - {class_name} - {component} - β{betti_dim}")
            
            # Also create 2D heatmap view for each position
            for position in positions:
                plt.figure(figsize=(12, 8))
                
                # Create Z matrix (betti values)
                Z = np.zeros((len(scales), num_layers))
                
                # Fill Z with betti values
                for layer_idx in range(num_layers):
                    layer_data = results[class_name][component][layer_idx]
                    if position in layer_data:
                        for scale_idx, scale in enumerate(scales):
                            if scale in layer_data[position]:
                                betti_values = layer_data[position][scale]
                                if betti_dim < len(betti_values):
                                    Z[scale_idx, layer_idx] = betti_values[betti_dim]
                
                # Create heatmap
                plt.imshow(Z, aspect='auto', cmap='viridis', interpolation='nearest')
                plt.colorbar(label=f'Betti Number β{betti_dim}')
                
                # Set labels and title
                plt.xlabel('Layers')
                plt.ylabel('Scale Index')
                plt.title(f'{activation_name} - {class_name} - {component} - {position} - β{betti_dim}')
                
                # Set y-ticks to actual scale values (show a subset for readability)
                if len(scales) > 10:
                    tick_indices = np.linspace(0, len(scales)-1, 5, dtype=int)
                    plt.yticks(tick_indices, [f"{scales[int(i)]:.2f}" for i in tick_indices])
                else:
                    plt.yticks(range(len(scales)), [f"{s:.2f}" for s in scales])
                
                # Save figure
                plt.savefig(os.path.join(vis_dir, f"{component}_{position}_betti{betti_dim}_heatmap.png"), dpi=300)
                plt.close()
                
                print(f"Created heatmap for {activation_name} - {class_name} - {component} - {position} - β{betti_dim}")
    
    return vis_dir

def visualize_all_results(results_dir, all_results, all_scales):
    """
    Create 3D visualizations for all activation functions and methods in the results.
    
    Args:
        results_dir: Base directory where results are stored
        all_results: Dictionary mapping activation names to results
        all_scales: Dictionary mapping activation names to scales
    
    Returns:
        Dictionary mapping activation names to visualization directories
    """
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_base_dir = os.path.join(results_dir, f'3d_visualizations_{timestamp}')
    os.makedirs(vis_base_dir, exist_ok=True)
    
    vis_dirs = {}
    
    for activation_name, results in all_results.items():
        scales = all_scales[activation_name]
        
        # Extract method from results_dir
        if 'euclidean' in results_dir:
            method = 'euclidean'
        elif 'geodesic' in results_dir:
            method = 'geodesic'
        else:
            method = 'unknown'
        
        # Create visualizations
        vis_dir = create_3d_visualization(
            results, 
            scales, 
            activation_name, 
            method=method,
            output_dir=vis_base_dir
        )
        
        vis_dirs[activation_name] = vis_dir
    
    print(f"All 3D visualizations created and saved to {vis_base_dir}")
    return vis_dirs

def load_and_visualize_from_dir(results_dir):
    """
    Load results from a directory and create 3D visualizations.
    
    Args:
        results_dir: Directory containing the results
    
    Returns:
        Dictionary mapping activation names to visualization directories
    """
    print(f"Loading results from {results_dir}")
    
    # Find all result JSON files
    json_files = []
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith('_results.json'):
                json_files.append(os.path.join(root, file))
    
    if not json_files:
        print(f"No result files found in {results_dir}")
        return {}
    
    # Extract method from results_dir
    if 'euclidean' in results_dir:
        method = 'euclidean'
    elif 'geodesic' in results_dir:
        method = 'geodesic'
    else:
        method = 'unknown'
    
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_base_dir = os.path.join(results_dir, f'3d_visualizations_{timestamp}')
    os.makedirs(vis_base_dir, exist_ok=True)
    
    vis_dirs = {}
    
    # Process each result file
    for json_file in json_files:
        # Extract activation name from filename
        filename = os.path.basename(json_file)
        activation_name = filename.split('_')[0]  # Assumes format like "GELU_euclidean_results.json"
        
        # Load results
        results = load_results(json_file)
        
        # Extract scales from the results
        scales = []
        try:
            # Find first available scale
            for component in ['attention', 'mlp']:
                if component in results['class0']:
                    for layer_data in results['class0'][component]:
                        for position, pos_data in layer_data.items():
                            scales = [float(s) for s in pos_data.keys()]
                            if scales:
                                break
                        if scales:
                            break
                if scales:
                    break
            
            scales = sorted(scales)
        except (KeyError, IndexError):
            print(f"Warning: Could not extract scales from {json_file}")
            scales = list(range(10))  # Default scales if extraction fails
        
        # Create visualizations
        vis_dir = create_3d_visualization(
            results, 
            scales, 
            activation_name, 
            method=method,
            output_dir=vis_base_dir
        )
        
        vis_dirs[activation_name] = vis_dir
    
    print(f"All 3D visualizations created and saved to {vis_base_dir}")
    return vis_dirs

if __name__ == "__main__":
    # This script is meant to be imported and used from the main workflow
    # But it can also be run directly to visualize existing results
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate 3D visualizations of topology analysis results')
    parser.add_argument('--results_dir', type=str, default='results_latest', 
                        help='Directory containing the results (default: results_latest)')
    args = parser.parse_args()
    
    # Visualize results
    load_and_visualize_from_dir(args.results_dir) 