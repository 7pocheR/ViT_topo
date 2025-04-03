#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Topology analysis module for computing persistent homology and Betti numbers.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from ripser import ripser
from persim import plot_diagrams
import torch

def create_pointclouds(features, labels, class_label=0):
    """
    Create pointclouds from extracted features.
    
    Args:
        features: Dictionary containing feature vectors
        labels: Labels for the feature vectors
        class_label: Class label to filter (0 for digit 0) - not needed when using topology_loader
        
    Returns:
        Dictionary of pointclouds for different positions
    """
    # When using topology_loader, all samples are already class 0, no need to filter
    # Just use all samples
    
    # Prepare pointclouds dictionary
    pointclouds = {}
    
    # Define the target size for pointclouds
    target_size = 1000
    
    # Report the number of samples
    num_samples = len(labels)
    print(f"Creating pointclouds with {num_samples} samples of digit 0")
    
    # Ensure we have sensible target size
    if num_samples < target_size:
        print(f"WARNING: Only {num_samples} samples available. "
              f"Reducing target size to {num_samples} to avoid artificial upsampling.")
        target_size = num_samples
    
    # For each key in features (attention/mlp)
    for key, feature_list in features.items():
        pointclouds[key] = []
        
        # Process each layer
        for layer_idx, feature_tensor in enumerate(feature_list):
            # Calculate the central patch position
            # For a 7x7 grid (49 patches), the central patch is at position 24 (0-indexed)
            # Add 1 to account for CLS token at position 0
            central_patch_idx = 24 + 1
            
            # Top-left corner patch is the first patch after CLS token (index 1)
            top_left_patch_idx = 1
            
            # Create exactly 3 pointclouds
            layer_pointclouds = {
                # CLS token pointcloud
                'cls': feature_tensor[:, 0, :].cpu().numpy(),
                
                # Central patch pointcloud
                'central_patch': feature_tensor[:, central_patch_idx, :].cpu().numpy(),
                
                # Top-left corner patch
                'top_left_patch': feature_tensor[:, top_left_patch_idx, :].cpu().numpy()
            }
            
            # Sample exactly target_size points from each position
            for position, cloud in layer_pointclouds.items():
                num_available = len(cloud)
                
                if num_available > target_size:
                    # Downsample to exactly target_size points using evenly spaced indices
                    indices = np.linspace(0, num_available-1, target_size, dtype=int)
                    layer_pointclouds[position] = cloud[indices]
            
            pointclouds[key].append(layer_pointclouds)
    
    return pointclouds

def apply_pca(pointclouds, n_components=50):
    """
    Apply PCA to reduce dimension of pointclouds.
    
    Args:
        pointclouds: Dictionary of pointclouds
        n_components: Number of PCA components
        
    Returns:
        Dictionary of dimension-reduced pointclouds
    """
    from sklearn.decomposition import PCA
    
    reduced_pointclouds = {}
    
    # Process each feature type (attention/mlp)
    for key, layer_pointclouds in pointclouds.items():
        reduced_pointclouds[key] = []
        
        # Process each layer
        for layer_idx, position_pointclouds in enumerate(layer_pointclouds):
            reduced_positions = {}
            
            # Process each position
            for position, cloud in position_pointclouds.items():
                # Skip if empty
                if len(cloud) == 0:
                    continue
                
                # Apply PCA if dimension > n_components
                if cloud.shape[1] > n_components:
                    pca = PCA(n_components=n_components)
                    reduced_positions[position] = pca.fit_transform(cloud)
                else:
                    reduced_positions[position] = cloud
            
            reduced_pointclouds[key].append(reduced_positions)
    
    return reduced_pointclouds

def compute_persistent_homology(pointcloud, k=14, max_dim=2, batch_size=1000):
    """
    Compute persistent homology for a pointcloud.
    
    Args:
        pointcloud: Numpy array of shape (n_points, n_dimensions)
        k: Number of nearest neighbors for distance computation
        max_dim: Maximum homology dimension (keeping at 3 to compute β₀, β₁, β₂, β₃)
        batch_size: Batch size for processing large pointclouds (reduced to 1000 for memory efficiency)
        
    Returns:
        Persistence diagrams
    """
    # For large pointclouds, use batching
    if len(pointcloud) > batch_size:
        # Randomly sample points
        indices = np.random.choice(len(pointcloud), batch_size, replace=False)
        pointcloud = pointcloud[indices]
    
    print(f"Computing persistent homology on pointcloud of shape {pointcloud.shape}")
    
    # Compute persistent homology using ripser
    try:
        diagrams = ripser(pointcloud, maxdim=max_dim)['dgms']
        return diagrams
    except Exception as e:
        print(f"Error computing persistent homology: {e}")
        # Return empty diagrams if computation fails
        empty_diagrams = [np.empty((0, 2)) for _ in range(max_dim + 1)]
        return empty_diagrams

def calculate_betti_numbers(diagrams, scales):
    """
    Calculate Betti numbers from persistence diagrams at specified scales.
    
    Args:
        diagrams: List of persistence diagrams for different dimensions
        scales: List of scales at which to calculate Betti numbers
        
    Returns:
        Dictionary mapping scales to Betti numbers
    """
    betti_numbers = {}
    
    for scale in scales:
        betti_at_scale = []
        
        for dim, diagram in enumerate(diagrams):
            # Count points born before and dying after the scale
            born_before = diagram[:, 0] <= scale
            dying_after = diagram[:, 1] > scale
            betti_at_scale.append(np.sum(born_before & dying_after))
        
        betti_numbers[scale] = betti_at_scale
    
    return betti_numbers

def visualize_betti_numbers(betti_numbers, title="Betti Numbers"):
    """
    Visualize Betti numbers.
    
    Args:
        betti_numbers: Dictionary mapping scales to Betti numbers
        title: Plot title
    """
    scales = sorted(betti_numbers.keys())
    betti_0 = [betti_numbers[scale][0] for scale in scales]
    betti_1 = [betti_numbers[scale][1] for scale in scales if len(betti_numbers[scale]) > 1]
    betti_2 = [betti_numbers[scale][2] for scale in scales if len(betti_numbers[scale]) > 2]
    
    plt.figure(figsize=(10, 6))
    plt.plot(scales, betti_0, 'b-', marker='o', label='β₀')
    
    if betti_1:
        plt.plot(scales, betti_1, 'r-', marker='s', label='β₁')
    
    if betti_2:
        plt.plot(scales, betti_2, 'g-', marker='^', label='β₂')
    
    plt.xlabel('Scale (ε)')
    plt.ylabel('Betti Number')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()

def calibrate_scales(pointclouds, k=14, max_dim=2, n_samples=3):
    """
    Calibrate scales for persistent homology.
    
    Args:
        pointclouds: Dictionary of pointclouds
        k: Number of nearest neighbors
        max_dim: Maximum homology dimension (increased to 3)
        n_samples: Number of pointclouds to sample for calibration
        
    Returns:
        List of recommended scales
    """
    print("Using fixed scale range as specified (0.2 to 1.0 with 0.1 increments)")
    
    # Use fixed scales as specified
    scales = np.arange(0.2, 1.1, 0.1)
    return scales

def analyze_topology(model, data_loader, device, activation_name, pca_components=50):
    """
    Analyze topology of model representations.
    
    Args:
        model: Trained model
        data_loader: Data loader for test set
        device: Device to run model on
        activation_name: Name of activation function
        pca_components: Number of PCA components
        
    Returns:
        Dictionary of Betti numbers
    """
    model.eval()
    
    # Collect features and labels
    all_features = []
    all_labels = []
    
    # Target number of class 0 samples
    target_class0_samples = 1000
    
    # Maximum number of batches to process to avoid excessive computation
    max_batches = 50
    
    print(f"Extracting features for {activation_name} model...")
    current_class0_count = 0
    processed_batches = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            if i >= max_batches:
                print(f"Reached maximum batch limit ({max_batches}). Stopping feature extraction.")
                break
            
            images = images.to(device)
            
            # Forward pass
            _ = model(images)
            
            # Extract features
            features = model.extract_features()
            all_features.append(features)
            all_labels.append(labels)
            
            current_class0_count += len(labels)
            processed_batches += 1
            
            print(f"Processed batch {processed_batches}, Total class 0 samples: {current_class0_count}/{target_class0_samples}", end='\r')
            
            # Stop if we've collected enough class 0 samples
            if current_class0_count >= target_class0_samples:
                print(f"\nReached target of {target_class0_samples} class 0 samples. Stopping feature extraction.")
                break
    
    # Check if any features were collected
    if not all_features:
        raise ValueError("No features were collected. Please check the data loader.")
    
    # Combine features from batches
    combined_features = {
        'attention': [torch.cat([f['attention'][i] for f in all_features], dim=0) for i in range(len(all_features[0]['attention']))],
        'mlp': [torch.cat([f['mlp'][i] for f in all_features], dim=0) for i in range(len(all_features[0]['mlp']))]
    }
    combined_labels = torch.cat(all_labels, dim=0)
    
    print(f"\nExtracted features from {len(combined_labels)} images with class 0")
    
    # Create pointclouds for class 0 only
    print("Creating pointclouds...")
    pointclouds_class0 = create_pointclouds(combined_features, combined_labels, class_label=0)
    
    # Apply PCA
    print("Applying PCA...")
    reduced_pointclouds_class0 = apply_pca(pointclouds_class0, n_components=pca_components)
    
    # Get fixed scales
    scales = calibrate_scales(reduced_pointclouds_class0)
    print(f"Using scales: {scales}")
    
    # Compute topology
    results = {
        'class0': {'attention': [], 'mlp': []}
    }
    
    # Total number of computations for progress tracking
    total_computations = 0
    for key in ['attention', 'mlp']:
        for position_clouds in reduced_pointclouds_class0[key]:
            total_computations += len(position_clouds)
    
    # Initialize progress counter
    current_computation = 0
    
    print(f"Beginning topology analysis ({total_computations} total computations)...")
    
    # Analyze each position for class 0
    for key in ['attention', 'mlp']:
        for layer_idx, position_clouds in enumerate(reduced_pointclouds_class0[key]):
            layer_results = {}
            
            for position, cloud in position_clouds.items():
                current_computation += 1
                if len(cloud) > 0:
                    print(f"Progress: {current_computation}/{total_computations} - Class 0, {key}, Layer {layer_idx}, {position}")
                    try:
                        # Compute persistent homology
                        diagrams = compute_persistent_homology(cloud)
                        
                        # Calculate Betti numbers
                        betti_numbers = calculate_betti_numbers(diagrams, scales)
                        
                        layer_results[position] = betti_numbers
                    except Exception as e:
                        print(f"\nError processing {key} layer {layer_idx}, position {position}: {e}")
            
            results['class0'][key].append(layer_results)
    
    print("\nTopology analysis complete!")
    return results, scales 