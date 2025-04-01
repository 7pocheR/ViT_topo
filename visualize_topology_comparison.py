#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comparison visualizations for topology analysis between GELU and ReLU activations.
Compares how the two activation functions affect topology in the ViT model.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def load_results(json_path):
    """Load JSON results file"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results

def visualize_activation_comparison(output_dir='results_comparison'):
    """
    Create visualizations comparing GELU and ReLU activation functions.
    
    Args:
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results files
    gelu_path = os.path.join('results', 'GELU_results.json')
    relu_path = os.path.join('results', 'ReLU_results.json')
    
    # Check if files exist
    missing_files = []
    if not os.path.exists(gelu_path):
        missing_files.append(gelu_path)
    if not os.path.exists(relu_path):
        missing_files.append(relu_path)
        
    if missing_files:
        print(f"Error: Could not find result files: {', '.join(missing_files)}")
        return
        
    gelu_results = load_results(gelu_path)
    relu_results = load_results(relu_path)
    
    # Define positions and components to analyze
    positions = ['cls', 'central_patch', 'total_patches']
    components = ['attention', 'mlp']
    
    # Process each class
    for class_name in ['class0', 'class1']:
        # Create directory for this class
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Process each position
        for position in positions:
            # Create directory for this position
            pos_dir = os.path.join(class_dir, position)
            os.makedirs(pos_dir, exist_ok=True)
            
            # Process each component
            for component in components:
                # Get scales (from GELU, assuming both have same scales)
                all_scales = set()
                num_layers = len(gelu_results[class_name][component])
                
                for layer_idx in range(num_layers):
                    if position in gelu_results[class_name][component][layer_idx]:
                        all_scales.update(float(s) for s in gelu_results[class_name][component][layer_idx][position].keys())
                
                scales = sorted(all_scales)
                
                # Skip if no scales found
                if not scales:
                    print(f"No scales found for {class_name}-{position}-{component}")
                    continue
                
                # Select a representative scale for comparison (middle of the range)
                middle_scale_idx = len(scales) // 2
                middle_scale = scales[middle_scale_idx]
                middle_scale_str = str(float(middle_scale))
                
                # Compare each Betti dimension across activations
                for betti_dim in range(3):
                    plt.figure(figsize=(12, 8))
                    
                    # Extract values for GELU
                    gelu_values = []
                    for layer_idx in range(num_layers):
                        if (position in gelu_results[class_name][component][layer_idx] and
                            middle_scale_str in gelu_results[class_name][component][layer_idx][position] and
                            len(gelu_results[class_name][component][layer_idx][position][middle_scale_str]) > betti_dim):
                            gelu_values.append(gelu_results[class_name][component][layer_idx][position][middle_scale_str][betti_dim])
                        else:
                            gelu_values.append(0)
                    
                    # Extract values for ReLU
                    relu_values = []
                    for layer_idx in range(num_layers):
                        if (position in relu_results[class_name][component][layer_idx] and
                            middle_scale_str in relu_results[class_name][component][layer_idx][position] and
                            len(relu_results[class_name][component][layer_idx][position][middle_scale_str]) > betti_dim):
                            relu_values.append(relu_results[class_name][component][layer_idx][position][middle_scale_str][betti_dim])
                        else:
                            relu_values.append(0)
                    
                    # Plot comparison
                    plt.plot(range(num_layers), gelu_values, 'b-', marker='o', linewidth=2, label='GELU')
                    plt.plot(range(num_layers), relu_values, 'r-', marker='s', linewidth=2, label='ReLU')
                    
                    plt.xlabel('Layer')
                    plt.ylabel(f'Betti Number β{betti_dim}')
                    plt.title(f'GELU vs ReLU - {class_name} - {position} - {component} - β{betti_dim}\nScale: {float(middle_scale):.2f}')
                    plt.legend()
                    plt.grid(True)
                    
                    # Save figure
                    plt.savefig(os.path.join(pos_dir, f"{component}_betti{betti_dim}_comparison.png"), dpi=300)
                    plt.close()
                    
                    print(f"Created activation comparison for {class_name} - {position} - {component} - β{betti_dim}")
                
                # Create heatmap of GELU-ReLU difference for all scales
                for betti_dim in range(3):
                    plt.figure(figsize=(14, 8))
                    
                    # Create matrices for both activations
                    gelu_matrix = np.zeros((len(scales), num_layers))
                    relu_matrix = np.zeros((len(scales), num_layers))
                    
                    # Fill matrices
                    for scale_idx, scale in enumerate(scales):
                        scale_str = str(float(scale))
                        
                        for layer_idx in range(num_layers):
                            # GELU values
                            if (position in gelu_results[class_name][component][layer_idx] and
                                scale_str in gelu_results[class_name][component][layer_idx][position] and
                                len(gelu_results[class_name][component][layer_idx][position][scale_str]) > betti_dim):
                                gelu_matrix[scale_idx, layer_idx] = gelu_results[class_name][component][layer_idx][position][scale_str][betti_dim]
                            
                            # ReLU values
                            if (position in relu_results[class_name][component][layer_idx] and
                                scale_str in relu_results[class_name][component][layer_idx][position] and
                                len(relu_results[class_name][component][layer_idx][position][scale_str]) > betti_dim):
                                relu_matrix[scale_idx, layer_idx] = relu_results[class_name][component][layer_idx][position][scale_str][betti_dim]
                    
                    # Calculate difference (GELU - ReLU)
                    diff_matrix = gelu_matrix - relu_matrix
                    
                    # Create heatmap
                    plt.figure(figsize=(12, 8))
                    plt.imshow(diff_matrix, aspect='auto', cmap='coolwarm', interpolation='nearest')
                    plt.colorbar(label=f'Difference in Betti Number β{betti_dim} (GELU - ReLU)')
                    
                    # Set labels and title
                    plt.xlabel('Layers')
                    plt.ylabel('Scale Index')
                    plt.title(f'GELU-ReLU Difference - {class_name} - {position} - {component} - β{betti_dim}')
                    
                    # Set y-ticks to actual scale values (show a subset for readability)
                    if len(scales) > 10:
                        tick_indices = np.linspace(0, len(scales)-1, 5, dtype=int)
                        plt.yticks(tick_indices, [f"{scales[i]:.2f}" for i in tick_indices])
                    else:
                        plt.yticks(range(len(scales)), [f"{s:.2f}" for s in scales])
                    
                    # Save figure
                    plt.savefig(os.path.join(pos_dir, f"{component}_betti{betti_dim}_diff_heatmap.png"), dpi=300)
                    plt.close()
                    
                    print(f"Created difference heatmap for {class_name} - {position} - {component} - β{betti_dim}")

def visualize_class_comparison(output_dir='results_class_comparison'):
    """
    Create visualizations comparing class 0 and class 1 patterns.
    
    Args:
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results files for GELU and ReLU
    models = ['GELU', 'ReLU']
    results = {}
    
    for model in models:
        model_path = os.path.join('results', f'{model}_results.json')
        if not os.path.exists(model_path):
            print(f"Error: Could not find result file: {model_path}")
            return
        results[model] = load_results(model_path)
    
    # Define positions and components to analyze
    positions = ['cls', 'central_patch', 'total_patches']
    components = ['attention', 'mlp']
    
    # Process each activation model
    for model in models:
        # Create directory for this model
        model_dir = os.path.join(output_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        
        # Process each position
        for position in positions:
            # Create directory for this position
            pos_dir = os.path.join(model_dir, position)
            os.makedirs(pos_dir, exist_ok=True)
            
            # Process each component
            for component in components:
                # Get scales (from class0, assuming both classes have same scales)
                all_scales = set()
                num_layers = len(results[model]['class0'][component])
                
                for layer_idx in range(num_layers):
                    if position in results[model]['class0'][component][layer_idx]:
                        all_scales.update(float(s) for s in results[model]['class0'][component][layer_idx][position].keys())
                
                scales = sorted(all_scales)
                
                # Skip if no scales found
                if not scales:
                    print(f"No scales found for {model}-{position}-{component}")
                    continue
                
                # Select a representative scale for comparison (middle of the range)
                middle_scale_idx = len(scales) // 2
                middle_scale = scales[middle_scale_idx]
                middle_scale_str = str(float(middle_scale))
                
                # Compare each Betti dimension across classes
                for betti_dim in range(3):
                    plt.figure(figsize=(12, 8))
                    
                    # Extract values for class0
                    class0_values = []
                    for layer_idx in range(num_layers):
                        if (position in results[model]['class0'][component][layer_idx] and
                            middle_scale_str in results[model]['class0'][component][layer_idx][position] and
                            len(results[model]['class0'][component][layer_idx][position][middle_scale_str]) > betti_dim):
                            class0_values.append(results[model]['class0'][component][layer_idx][position][middle_scale_str][betti_dim])
                        else:
                            class0_values.append(0)
                    
                    # Extract values for class1
                    class1_values = []
                    for layer_idx in range(num_layers):
                        if (position in results[model]['class1'][component][layer_idx] and
                            middle_scale_str in results[model]['class1'][component][layer_idx][position] and
                            len(results[model]['class1'][component][layer_idx][position][middle_scale_str]) > betti_dim):
                            class1_values.append(results[model]['class1'][component][layer_idx][position][middle_scale_str][betti_dim])
                        else:
                            class1_values.append(0)
                    
                    # Plot comparison
                    plt.plot(range(num_layers), class0_values, 'g-', marker='o', linewidth=2, label='Class 0 (digit 0)')
                    plt.plot(range(num_layers), class1_values, 'm-', marker='s', linewidth=2, label='Class 1 (non-0)')
                    
                    plt.xlabel('Layer')
                    plt.ylabel(f'Betti Number β{betti_dim}')
                    plt.title(f'{model} - Class Comparison - {position} - {component} - β{betti_dim}\nScale: {float(middle_scale):.2f}')
                    plt.legend()
                    plt.grid(True)
                    
                    # Save figure
                    plt.savefig(os.path.join(pos_dir, f"{component}_betti{betti_dim}_class_comparison.png"), dpi=300)
                    plt.close()
                    
                    print(f"Created class comparison for {model} - {position} - {component} - β{betti_dim}")

if __name__ == "__main__":
    print("Generating activation comparison visualizations (GELU vs ReLU)...")
    visualize_activation_comparison()
    
    print("\nGenerating class comparison visualizations (class 0 vs class 1)...")
    visualize_class_comparison()
    
    print("\nComparison visualization complete!") 