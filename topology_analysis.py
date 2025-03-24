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
        class_label: Class label to filter (0 or 1)
        
    Returns:
        Dictionary of pointclouds for different positions
    """
    # Filter to get only the specified class
    class_mask = (labels == class_label)
    
    # Prepare pointclouds dictionary
    pointclouds = {}
    
    # Define the target size for pointclouds (reduced to 1000 for memory efficiency)
    target_size = 1000
    
    # Report the number of samples in this class
    class_sample_count = class_mask.sum().item()
    print(f"Creating pointclouds for class {class_label}: {class_sample_count} samples available")
    
    # For each key in features (attention/mlp)
    for key, feature_list in features.items():
        pointclouds[key] = []
        
        # Process each layer
        for layer_idx, feature_tensor in enumerate(feature_list):
            # Get features for the specified class
            class_features = feature_tensor[class_mask]
            
            # Calculate the central patch position
            # For a 7x7 grid (49 patches), the central patch is at position 24 (0-indexed)
            # Add 1 to account for CLS token at position 0
            central_patch_idx = 24 + 1
            
            # Create exactly 3 pointclouds
            layer_pointclouds = {
                # CLS token pointcloud
                'cls': class_features[:, 0, :].cpu().numpy(),
                
                # Central patch pointcloud (at position 4,4)
                'central_patch': class_features[:, central_patch_idx, :].cpu().numpy(),
                
                # Total patches pointcloud - sample more patches to reach the target size
                'total_patches': class_features[:min(500, len(class_features)), 1:, :].reshape(-1, class_features.shape[-1]).cpu().numpy()
            }
            
            # Ensure consistent size by sampling if needed
            for position, cloud in layer_pointclouds.items():
                # For total_patches, we might need to downsample
                if len(cloud) > target_size:
                    indices = np.random.choice(len(cloud), target_size, replace=False)
                    layer_pointclouds[position] = cloud[indices]
                # For cls and central_patch, we might need to upsample (duplicate points)
                elif len(cloud) < target_size and position != 'total_patches':
                    # Duplicate points with replacement to reach target size
                    indices = np.random.choice(len(cloud), target_size, replace=True)
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

def compute_persistent_homology(pointcloud, k=14, max_dim=3, batch_size=1000):
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

def calibrate_scales(pointclouds, k=14, max_dim=3, n_samples=3):
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
    print("Calibrating scales for persistent homology...")
    
    # Sample pointclouds for calibration
    sampled_clouds = []
    
    # Try to get one of each type of pointcloud
    positions_to_try = ['cls', 'central_patch', 'total_patches']
    
    for key in pointclouds:
        if len(sampled_clouds) < n_samples:
            for layer_idx, position_clouds in enumerate(pointclouds[key]):
                for position in positions_to_try:
                    if position in position_clouds and len(position_clouds[position]) > 0 and len(sampled_clouds) < n_samples:
                        sampled_clouds.append((key, layer_idx, position, position_clouds[position]))
                        break
    
    # Determine max persistence
    max_persistence = 0
    
    for key, layer_idx, position, cloud in sampled_clouds:
        try:
            # Compute persistent homology
            diagrams = compute_persistent_homology(cloud, k=k, max_dim=max_dim)
            
            # Find maximum finite death time
            for dim, diagram in enumerate(diagrams):
                finite_deaths = diagram[diagram[:, 1] < np.inf, 1]
                if len(finite_deaths) > 0:
                    max_persistence = max(max_persistence, np.max(finite_deaths))
        except Exception as e:
            print(f"Error during calibration: {e}")
    
    # Define scales based on persistence range
    if max_persistence > 0:
        scales = np.linspace(0.5, max_persistence * 1.2, 5)
        return scales
    else:
        return [0.5, 1.0, 1.5, 2.0, 2.5]  # Default scales

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
    
    # Process a moderate number of batches (8 batches is a good balance)
    max_batches = 8
    
    print(f"Extracting features for {activation_name} model...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            if i >= max_batches:
                break
                
            images = images.to(device)
            
            # Forward pass
            _ = model(images)
            
            # Extract features
            features = model.extract_features()
            all_features.append(features)
            all_labels.append(labels)
            
            print(f"Processed batch {i+1}/{max_batches}", end='\r')
    
    # Combine features from batches
    combined_features = {
        'attention': [torch.cat([f['attention'][i] for f in all_features], dim=0) for i in range(len(all_features[0]['attention']))],
        'mlp': [torch.cat([f['mlp'][i] for f in all_features], dim=0) for i in range(len(all_features[0]['mlp']))]
    }
    combined_labels = torch.cat(all_labels, dim=0)
    
    print(f"\nExtracted features from {len(combined_labels)} images")
    
    # Create pointclouds
    print("Creating pointclouds...")
    pointclouds_class0 = create_pointclouds(combined_features, combined_labels, class_label=0)
    pointclouds_class1 = create_pointclouds(combined_features, combined_labels, class_label=1)
    
    # Apply PCA
    print("Applying PCA...")
    reduced_pointclouds_class0 = apply_pca(pointclouds_class0, n_components=pca_components)
    reduced_pointclouds_class1 = apply_pca(pointclouds_class1, n_components=pca_components)
    
    # Calibrate scales
    scales = calibrate_scales(reduced_pointclouds_class0)
    print(f"Calibrated scales: {scales}")
    
    # Compute topology
    results = {
        'class0': {'attention': [], 'mlp': []},
        'class1': {'attention': [], 'mlp': []}
    }
    
    # Total number of computations for progress tracking
    total_computations = 0
    for cls_data in [reduced_pointclouds_class0, reduced_pointclouds_class1]:
        for key in ['attention', 'mlp']:
            for position_clouds in cls_data[key]:
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
    
    # Analyze each position for class 1
    for key in ['attention', 'mlp']:
        for layer_idx, position_clouds in enumerate(reduced_pointclouds_class1[key]):
            layer_results = {}
            
            for position, cloud in position_clouds.items():
                current_computation += 1
                if len(cloud) > 0:
                    print(f"Progress: {current_computation}/{total_computations} - Class 1, {key}, Layer {layer_idx}, {position}")
                    try:
                        # Compute persistent homology
                        diagrams = compute_persistent_homology(cloud)
                        
                        # Calculate Betti numbers
                        betti_numbers = calculate_betti_numbers(diagrams, scales)
                        
                        layer_results[position] = betti_numbers
                    except Exception as e:
                        print(f"\nError processing {key} layer {layer_idx}, position {position}: {e}")
            
            results['class1'][key].append(layer_results)
    
    print("\nTopology analysis complete!")
    return results, scales 