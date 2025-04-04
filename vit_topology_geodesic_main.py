#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for analyzing topology in Vision Transformers using graph geodesic distance
Based on "Topology of Deep Neural Networks" (Naitzat et al., 2020)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import time
import json
import datetime

# Project imports
from vit_model import SimpleViT
from topology_analysis_geodesic import analyze_topology_geodesic, visualize_betti_numbers

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def prepare_topology_data():
    """
    Load MNIST datasets and create a data loader specifically for topological analysis
    focusing only on class 0 (digit 0) samples.
    
    Prioritizes samples from the test set and supplements with training set samples
    if needed to reach 1000 total samples.
    """
    print("Preparing data for topology analysis...")
    
    # Use the same transform as for training/testing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load test dataset first
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Find all digit 0 indices in test set
    test_class0_indices = torch.where(test_dataset.targets == 0)[0]
    num_test_class0 = len(test_class0_indices)
    print(f"Found {num_test_class0} samples of digit 0 in the MNIST test set")
    
    # Target number of samples
    target_samples = 1000
    
    # Create binary labels for consistency with model training
    test_dataset.targets = (test_dataset.targets == 0).float()
    
    # If we don't have enough test samples, supplement with training samples
    if num_test_class0 < target_samples:
        additional_needed = target_samples - num_test_class0
        print(f"Need {additional_needed} more samples from training set")
        
        # Load training dataset
        train_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        # Find all digit 0 indices in training set
        train_class0_indices = torch.where(train_dataset.targets == 0)[0]
        num_train_class0 = len(train_class0_indices)
        print(f"Found {num_train_class0} samples of digit 0 in the MNIST training set")
        
        # Randomly sample the needed additional indices
        train_indices_to_use = train_class0_indices[torch.randperm(num_train_class0)[:additional_needed]]
        
        # Create binary labels for training set
        train_dataset.targets = (train_dataset.targets == 0).float()
        
        # Create a concatenated dataset
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([
            torch.utils.data.Subset(test_dataset, test_class0_indices),
            torch.utils.data.Subset(train_dataset, train_indices_to_use)
        ])
        
        print(f"Created combined dataset with {len(combined_dataset)} samples of digit 0")
        
        # Create a data loader for the combined dataset
        topology_loader = torch.utils.data.DataLoader(
            combined_dataset, batch_size=256, shuffle=False
        )
    else:
        # Just use the test dataset
        print(f"Using {num_test_class0} samples from test set only")
        
        # Create a data loader that only includes class 0 samples
        topology_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(test_dataset, test_class0_indices),
            batch_size=256, shuffle=False
        )
    
    return topology_loader

def visualize_topology_results(results, scales, activation_name, output_dir='results'):
    """
    Visualize topology analysis results.
    
    Args:
        results: Dictionary of Betti numbers
        scales: List of scales
        activation_name: Name of activation function
        output_dir: Output directory for plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Betti numbers for each position and component for class 0
    class_name = "class0"
    class_dir = os.path.join(output_dir, f"{activation_name}_geodesic_{class_name}")
    os.makedirs(class_dir, exist_ok=True)
    
    for component in ['attention', 'mlp']:
        # For each Betti dimension (0, 1, 2)
        for betti_dim in range(3):  # β₀, β₁, β₂
            # Create a figure for this component and Betti dimension
            plt.figure(figsize=(15, 10))
            
            # Positions to plot
            positions_to_plot = ['cls', 'central_patch', 'top_left_patch']
            
            # For each position
            for position in positions_to_plot:
                # Extract Betti numbers across layers
                layer_betti = []
                
                # For each layer
                for layer_idx, layer_results in enumerate(results[class_name][component]):
                    if position in layer_results and scales[0] in layer_results[position] and len(layer_results[position][scales[0]]) > betti_dim:
                        # Use the first scale for visualization
                        layer_betti.append(layer_results[position][scales[0]][betti_dim])
                    else:
                        # Position not available or Betti dimension not available
                        layer_betti.append(0)
                
                # Plot only if there are non-zero values for dimensions > 0
                if len(layer_betti) > 0 and (betti_dim == 0 or sum(layer_betti) > 0):
                    plt.plot(range(len(layer_betti)), layer_betti, marker='o', label=f"{position} β{betti_dim}")
            
            plt.xlabel('Layer')
            plt.ylabel(f'Betti Number β{betti_dim}')
            plt.title(f'{activation_name} (Geodesic) - {class_name} - {component} - β{betti_dim}')
            plt.grid(True)
            plt.legend()
            
            # Save figure
            plt.savefig(os.path.join(class_dir, f"{component}_betti{betti_dim}.png"))
            plt.close()
    
    # Save the numerical results
    with open(os.path.join(output_dir, f"{activation_name}_geodesic_results.json"), 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for class_name, class_results in results.items():
            serializable_results[class_name] = {}
            for component, component_results in class_results.items():
                serializable_results[class_name][component] = []
                for layer_results in component_results:
                    serializable_layer = {}
                    for position, position_results in layer_results.items():
                        serializable_layer[position] = {}
                        for scale, betti_numbers in position_results.items():
                            # Convert NumPy values to native Python types
                            serializable_layer[position][str(float(scale))] = [int(b) for b in betti_numbers]
                    serializable_results[class_name][component].append(serializable_layer)
        
        # Use a serialization helper function to handle NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        json.dump(serializable_results, f, indent=2, cls=NumpyEncoder)

def main():
    """Main function to execute the geodesic distance-based topology analysis"""
    # Configure model parameters for the pre-trained model
    model_params = {
        "img_size": 28,
        "patch_size": 4,
        "in_channels": 1,
        "embedding_dim": 24,
        "num_heads": 4,
        "num_transformer_blocks": 8,
        "mlp_ratio": 1.5,
    }
    
    # Create GELU model
    model_gelu = SimpleViT(
        activation='gelu',
        **model_params
    ).to(device)
    
    # Load the pretrained GELU model
    model_path_gelu = 'model_gelu_emb24_blocks8.pth'
    
    if os.path.exists(model_path_gelu):
        print("Loading pretrained GELU model...")
        model_gelu.load_state_dict(torch.load(model_path_gelu))
    else:
        raise FileNotFoundError(f"Pre-trained GELU model not found at {model_path_gelu}. "
                               f"Please run vit_topology_main.py first to train the models.")
    
    # Create a separate data loader specifically for topology analysis
    topology_loader = prepare_topology_data()
    
    # Create timestamp for results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results_geodesic_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Analyze topology with geodesic distance for GELU only
    print(f"\nAnalyzing topology with geodesic distance (saving to {results_dir})...")
    
    # GELU model
    gelu_results, gelu_scales = analyze_topology_geodesic(
        model_gelu, topology_loader, device, "GELU", pca_components=24, output_dir=results_dir
    )
    
    # Visualize results
    print("\nVisualizing results...")
    visualize_topology_results(gelu_results, gelu_scales, "GELU", output_dir=results_dir)
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds") 