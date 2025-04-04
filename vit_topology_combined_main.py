#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined main script for analyzing topology in Vision Transformers
Supports both Euclidean and graph geodesic distance methods.
Based on "Topology of Deep Neural Networks" (Naitzat et al., 2020)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
import datetime
import argparse

# Project imports
from vit_model import SimpleViT
from topology_analysis_combined import analyze_topology, visualize_betti_numbers
from visualize_topology_3d_for_integration import visualize_all_results, load_and_visualize_from_dir

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_mnist_data():
    """
    Load MNIST dataset and create balanced binary classification (0 vs. non-0)
    """
    print("Preparing MNIST data...")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create binary labels (0 vs. non-0)
    train_dataset.targets = (train_dataset.targets == 0).float()
    test_dataset.targets = (test_dataset.targets == 0).float()
    
    # Balance the dataset for training
    # Separate class 0 and class 1 samples
    train_class0_indices = torch.where(train_dataset.targets == 0)[0]
    train_class1_indices = torch.where(train_dataset.targets == 1)[0]
    
    # Sample an equal number from each class
    num_samples_per_class = min(len(train_class0_indices), len(train_class1_indices))
    
    # Randomly sample from the larger class to balance
    if len(train_class0_indices) > num_samples_per_class:
        train_class0_indices = train_class0_indices[torch.randperm(len(train_class0_indices))[:num_samples_per_class]]
    if len(train_class1_indices) > num_samples_per_class:
        train_class1_indices = train_class1_indices[torch.randperm(len(train_class1_indices))[:num_samples_per_class]]
    
    # Combine the indices
    balanced_train_indices = torch.cat([train_class0_indices, train_class1_indices])
    
    # Create balanced samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(balanced_train_indices)
    
    # Apply the same balancing to test dataset
    test_class0_indices = torch.where(test_dataset.targets == 0)[0]
    test_class1_indices = torch.where(test_dataset.targets == 1)[0]
    
    num_test_samples_per_class = min(len(test_class0_indices), len(test_class1_indices))
    
    if len(test_class0_indices) > num_test_samples_per_class:
        test_class0_indices = test_class0_indices[torch.randperm(len(test_class0_indices))[:num_test_samples_per_class]]
    if len(test_class1_indices) > num_test_samples_per_class:
        test_class1_indices = test_class1_indices[torch.randperm(len(test_class1_indices))[:num_test_samples_per_class]]
    
    balanced_test_indices = torch.cat([test_class0_indices, test_class1_indices])
    test_sampler = torch.utils.data.SubsetRandomSampler(balanced_test_indices)
    
    # Verify the type
    print(f"Target tensor type: {train_dataset.targets.dtype}")
    print(f"Class balance - Class 0: {num_samples_per_class}, Class 1: {num_samples_per_class}")
    
    # Create data loaders with the samplers
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, sampler=train_sampler
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, sampler=test_sampler
    )
    
    return train_loader, test_loader

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

def train_model(model, train_loader, activation_name, depth, width, epochs=30):
    """Train the ViT model"""
    print(f"Training {activation_name} model (depth={depth}, width={width})...")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Ensure float type
            
            # Debug: print types if first batch of first epoch
            if i == 0 and epoch == 0:
                print(f"Image type: {images.dtype}, Label type: {labels.dtype}")
                with torch.no_grad():
                    outputs = model(images)
                    print(f"Model output type: {outputs.dtype}, shape: {outputs.shape}")
                
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            predicted = (outputs > 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        # Print statistics
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, '
              f'Accuracy: {accuracy:.2f}%')
        
        # Early stopping if accuracy is high enough (keeping at 99%)
        if accuracy > 99:
            print(f"Reached {accuracy:.2f}% accuracy. Early stopping.")
            break
    
    # Save the model
    save_path = f'model_{activation_name.lower()}_emb{width}_blocks{depth}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model

def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            predicted = (outputs > 0).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def visualize_topology_results(results, scales, activation_name, method='euclidean', output_dir='results'):
    """
    Visualize topology analysis results.
    
    Args:
        results: Dictionary of Betti numbers
        scales: List of scales
        activation_name: Name of activation function
        method: 'euclidean' or 'geodesic'
        output_dir: Output directory for plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Betti numbers for each position and component for class 0
    class_name = "class0"
    method_suffix = "_" + method
    class_dir = os.path.join(output_dir, f"{activation_name}{method_suffix}_{class_name}")
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
            plt.title(f'{activation_name} ({method.capitalize()}) - {class_name} - {component} - β{betti_dim}')
            plt.grid(True)
            plt.legend()
            
            # Save figure
            plt.savefig(os.path.join(class_dir, f"{component}_betti{betti_dim}.png"))
            plt.close()
    
    # Save the numerical results
    with open(os.path.join(output_dir, f"{activation_name}_{method}_results.json"), 'w') as f:
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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vision Transformer Topology Analysis')
    
    parser.add_argument('--method', type=str, default='euclidean', choices=['euclidean', 'geodesic'],
                        help='Distance method to use (euclidean or geodesic)')
    
    parser.add_argument('--train', action='store_true',
                        help='Train models if not already trained')
    
    parser.add_argument('--model', type=str, default='both', choices=['gelu', 'relu', 'both'],
                        help='Which activation model to analyze (gelu, relu, or both)')
    
    parser.add_argument('--auto-scales', action='store_true',
                        help='Use automatic scale selection algorithms')
    
    parser.add_argument('--k', type=int, default=14,
                        help='Number of nearest neighbors for geodesic method (default: 14)')
    
    # Add new arguments for model architecture
    parser.add_argument('--depth', type=int, default=8,
                        help='Number of transformer blocks/layers (default: 8)')
    
    parser.add_argument('--width', type=int, default=24,
                        help='Embedding dimension for the transformer (default: 24)')
    
    parser.add_argument('--heads', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    
    parser.add_argument('--mlp-ratio', type=float, default=1.5,
                        help='MLP ratio for transformer blocks (default: 1.5)')
    
    # Add argument for controlling PCA
    parser.add_argument('--use-pca', action='store_true',
                        help='Apply PCA dimensionality reduction to features (only valid with Euclidean distance)')
    
    parser.add_argument('--pca-components', type=int, default=20,
                        help='Number of PCA components to use if PCA is enabled (default: 20)')
    
    # Add argument for 3D visualization
    parser.add_argument('--visualize-3d', action='store_true',
                        help='Create 3D visualizations of topology results')
    
    parser.add_argument('--no-visualization', action='store_true',
                        help='Skip all visualizations (useful for batch processing)')
    
    return parser.parse_args()

def main():
    """Main function to execute the entire pipeline"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Validate arguments - PCA only makes sense with Euclidean distance
    if args.use_pca and args.method == 'geodesic':
        print("ERROR: PCA can only be used with Euclidean distance method.")
        print("PCA distorts the neighborhood structure that geodesic distance is trying to capture.")
        print("Please use either:")
        print("  1. Euclidean distance with PCA: --method euclidean --use-pca")
        print("  2. Geodesic distance without PCA: --method geodesic")
        return
    
    print(f"Using device: {device}")
    print(f"Analysis method: {args.method}")
    print(f"Automatic scale selection: {args.auto_scales}")
    print(f"Model architecture - Depth: {args.depth}, Width: {args.width}, Heads: {args.heads}")
    if args.method == 'euclidean':
        print(f"Using PCA: {args.use_pca}")
        if args.use_pca:
            print(f"PCA components: {args.pca_components}")
    if args.method == 'geodesic':
        print(f"Initial k value for geodesic method: {args.k}")
    
    # Configure model parameters
    model_params = {
        "img_size": 28,
        "patch_size": 4,
        "in_channels": 1,
        "embedding_dim": args.width,
        "num_heads": args.heads,
        "num_transformer_blocks": args.depth,
        "mlp_ratio": args.mlp_ratio,
    }
    
    # Determine model file names based on architecture
    model_suffix = f"_emb{args.width}_blocks{args.depth}"
    
    # Create models with different activations
    models_to_analyze = []
    if args.model in ['gelu', 'both']:
        model_gelu = SimpleViT(
            activation='gelu',
            **model_params
        ).to(device)
        models_to_analyze.append(('GELU', model_gelu, f'model_gelu{model_suffix}.pth'))
    
    if args.model in ['relu', 'both']:
        model_relu = SimpleViT(
            activation='relu',
            **model_params
        ).to(device)
        models_to_analyze.append(('ReLU', model_relu, f'model_relu{model_suffix}.pth'))
    
    # Check if we need to train models
    models_need_training = False
    for _, model, model_path in models_to_analyze:
        if not os.path.exists(model_path):
            models_need_training = True
            break
    
    # Train models if needed
    if models_need_training and args.train:
        # Prepare data for training
        train_loader, test_loader = prepare_mnist_data()
        
        for activation_name, model, model_path in models_to_analyze:
            if not os.path.exists(model_path):
                print(f"Training {activation_name} model...")
                model = train_model(model, train_loader, activation_name, args.depth, args.width)
                # Evaluate model
                evaluate_model(model, test_loader)
            else:
                print(f"Loading pre-trained {activation_name} model from {model_path}")
                model.load_state_dict(torch.load(model_path, map_location=device))
    elif models_need_training and not args.train:
        print("Some models need training but --train flag is not set.")
        print("Pre-trained models are required for analysis.")
        print("Re-run with the --train flag to train missing models first.")
        return
    else:
        # Load pre-trained models
        for activation_name, model, model_path in models_to_analyze:
            print(f"Loading pre-trained {activation_name} model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Create a separate data loader specifically for topology analysis
    topology_loader = prepare_topology_data()
    
    # Create timestamp for results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    method_suffix = "_" + args.method
    model_arch_suffix = f"_d{args.depth}w{args.width}"
    pca_suffix = "_pca" if args.use_pca else "_nopca"
    results_dir = f'results{method_suffix}{model_arch_suffix}{pca_suffix}_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Analyze topology for each model
    print(f"\nAnalyzing topology (saving to {results_dir})...")
    
    all_results = {}
    all_scales = {}
    
    for activation_name, model, _ in models_to_analyze:
        print(f"\nAnalyzing {activation_name} model...")
        
        # Determine PCA components if PCA is enabled
        pca_components = args.pca_components
        if args.use_pca:
            # Adjust PCA components based on embedding dimension if not explicitly set
            if args.pca_components == 20:  # Using the default value
                pca_components = max(20, args.width // 2)
                print(f"Automatically adjusted PCA components to: {pca_components}")
        
        # Run topology analysis with the specified method
        results, scales = analyze_topology(
            model, 
            topology_loader, 
            device, 
            activation_name, 
            method=args.method, 
            k=args.k, 
            pca_components=pca_components, 
            auto_select_scales=args.auto_scales, 
            output_dir=results_dir,
            use_pca=args.use_pca
        )
        
        all_results[activation_name] = results
        all_scales[activation_name] = scales
        
        # Visualize results unless skipped
        if not args.no_visualization:
            visualize_topology_results(
                results, 
                scales, 
                activation_name, 
                method=args.method, 
                output_dir=results_dir
            )
    
    # Save configuration details for this run with timestamp
    config = vars(args)
    config['timestamp'] = timestamp
    config['device'] = str(device)
    
    config_file = os.path.join(results_dir, f'config_{timestamp}.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create 3D visualizations if requested
    if args.visualize_3d and not args.no_visualization:
        print("\nCreating 3D visualizations of topology results...")
        vis_dirs = visualize_all_results(results_dir, all_results, all_scales)
        print(f"3D visualizations created in: {', '.join(vis_dirs.values())}")
    
    print(f"\nAll topology analyses complete. Results saved to {results_dir}.")
    # Create a symbolic link to the latest results for easy access
    try:
        latest_link = 'results_latest'
        if os.path.exists(latest_link) or os.path.islink(latest_link):
            os.remove(latest_link)
        os.symlink(results_dir, latest_link)
        print(f"Created symbolic link '{latest_link}' pointing to the latest results.")
    except Exception as e:
        print(f"Could not create symbolic link to latest results: {e}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds") 