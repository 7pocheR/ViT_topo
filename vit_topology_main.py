#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for analyzing topology in Vision Transformers
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
from topology_analysis import compute_persistent_homology, calculate_betti_numbers, analyze_topology, visualize_betti_numbers

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

def train_model(model, train_loader, activation_name, epochs=30):
    """Train the ViT model"""
    print(f"Training {activation_name} model...")
    
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
    save_path = f'model_{activation_name.lower()}.pth'
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
    class_dir = os.path.join(output_dir, f"{activation_name}_{class_name}")
    os.makedirs(class_dir, exist_ok=True)
    
    for component in ['attention', 'mlp']:
        # For each Betti dimension (0, 1, 2, 3)
        for betti_dim in range(4):  # β₀, β₁, β₂, β₃
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
            plt.title(f'{activation_name} - {class_name} - {component} - β{betti_dim}')
            plt.grid(True)
            plt.legend()
            
            # Save figure
            plt.savefig(os.path.join(class_dir, f"{component}_betti{betti_dim}.png"))
            plt.close()
    
    # Save the numerical results
    with open(os.path.join(output_dir, f"{activation_name}_results.json"), 'w') as f:
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
    """Main function to execute the entire pipeline"""
    # Step 1: Prepare data
    train_loader, test_loader = prepare_mnist_data()
    
    # Step 2: Define model architectures (with GELU and ReLU)
    print("Creating models...")
    
    # Configure model parameters
    model_params = {
        "img_size": 28,
        "patch_size": 4,
        "in_channels": 1,
        "embedding_dim": 24,           # Increased from 16 to 24
        "num_heads": 4,
        "num_transformer_blocks": 8,   # Reduced from 12 to 8
        "mlp_ratio": 1.5,
    }
    
    # Create models with different activations
    model_gelu = SimpleViT(
        activation='gelu',
        **model_params
    ).to(device)
    
    model_relu = SimpleViT(
        activation='relu',
        **model_params
    ).to(device)
    
    # Check if models exist, otherwise train them
    model_path_gelu = 'model_gelu_emb24_blocks8.pth'
    model_path_relu = 'model_relu_emb24_blocks8.pth'
    
    if os.path.exists(model_path_gelu):
        print("Loading pretrained GELU model...")
        model_gelu.load_state_dict(torch.load(model_path_gelu))
    else:
        # Step 3: Train models
        model_gelu = train_model(model_gelu, train_loader, activation_name="GELU")
        # Save with new name
        torch.save(model_gelu.state_dict(), model_path_gelu)
    
    if os.path.exists(model_path_relu):
        print("Loading pretrained ReLU model...")
        model_relu.load_state_dict(torch.load(model_path_relu))
    else:
        model_relu = train_model(model_relu, train_loader, activation_name="ReLU")
        # Save with new name
        torch.save(model_relu.state_dict(), model_path_relu)
    
    # Step 4: Evaluate models
    print("\nEvaluating models:")
    gelu_accuracy = evaluate_model(model_gelu, test_loader)
    relu_accuracy = evaluate_model(model_relu, test_loader)
    
    print(f"\nGELU Model Accuracy: {gelu_accuracy:.2f}%")
    print(f"ReLU Model Accuracy: {relu_accuracy:.2f}%")
    
    # Create a separate data loader specifically for topology analysis
    topology_loader = prepare_topology_data()
    
    # Create timestamp for results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results_{timestamp}_emb24_blocks8'
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 5: Analyze topology
    print(f"\nAnalyzing topology (saving to {results_dir})...")
    
    # GELU model
    gelu_results, gelu_scales = analyze_topology(
        model_gelu, topology_loader, device, "GELU", pca_components=24, output_dir=results_dir
    )
    
    # ReLU model
    relu_results, relu_scales = analyze_topology(
        model_relu, topology_loader, device, "ReLU", pca_components=24, output_dir=results_dir
    )
    
    # Step 6: Visualize results
    print("\nVisualizing results...")
    visualize_topology_results(gelu_results, gelu_scales, "GELU", output_dir=results_dir)
    visualize_topology_results(relu_results, relu_scales, "ReLU", output_dir=results_dir)
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds") 