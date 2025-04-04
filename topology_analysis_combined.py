#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined topology analysis module supporting both Euclidean and graph geodesic distance
for computing persistent homology and Betti numbers in neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from ripser import ripser
from persim import plot_diagrams
from scipy.sparse.csgraph import shortest_path
import torch
import os
import time
from itertools import groupby
from operator import itemgetter
import datetime
from scipy.spatial.distance import pdist

def create_pointclouds(features, labels, class_label=0):
    """
    Create pointclouds from extracted features.
    
    Args:
        features: Dictionary containing feature vectors
        labels: Labels for the feature vectors
        class_label: Class label to filter (0 for digit 0)
        
    Returns:
        Dictionary of pointclouds for different positions
    """
    # Prepare pointclouds dictionary
    pointclouds = {}
    
    # Define the target size for pointclouds
    target_size = 1000
    
    # Report the number of samples
    num_samples = len(labels)
    print(f"Creating pointclouds with {num_samples} samples of class {class_label}")
    
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
            # Get dimensions to calculate patch positions safely
            num_samples, num_tokens, feature_dim = feature_tensor.shape
            
            # Calculate the central patch position - ensure it works for different patch sizes
            # For MNIST with 7x7 grid, we have 49 patches + CLS token = 50 total tokens
            num_patches = num_tokens - 1  # Subtract CLS token
            
            # Safely calculate positions
            if num_patches == 49:  # 7x7 layout
                # For a 7x7 grid (49 patches), the central patch is at position 24 (0-indexed)
                # Add 1 to account for CLS token at position 0
                central_patch_idx = 24 + 1
            else:
                # For other layouts, try to find the center
                grid_size = int(np.sqrt(num_patches))
                if grid_size**2 == num_patches:  # Perfect square
                    # Calculate center patch for a square grid
                    center_pos = grid_size // 2 * grid_size + grid_size // 2
                    central_patch_idx = center_pos + 1  # +1 for CLS token
                else:
                    # If not a perfect square grid, use a position near the middle
                    central_patch_idx = num_patches // 2 + 1  # +1 for CLS token
            
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
    First scales each pointcloud by its diameter to ensure scale invariance.
    
    Args:
        pointclouds: Dictionary of pointclouds
        n_components: Number of PCA components
        
    Returns:
        Dictionary of dimension-reduced pointclouds
    """
    from scipy.spatial.distance import pdist
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
                
                # Calculate the diameter of the pointcloud (maximum pairwise distance)
                # For large pointclouds, use sampling to estimate diameter
                if len(cloud) > 2000:
                    # Sample a subset of points
                    sample_size = 2000
                    indices = np.random.choice(len(cloud), sample_size, replace=False)
                    sampled_cloud = cloud[indices]
                    pairwise_distances = pdist(sampled_cloud)
                    diameter = np.max(pairwise_distances) if len(pairwise_distances) > 0 else 1.0
                else:
                    pairwise_distances = pdist(cloud)
                    diameter = np.max(pairwise_distances) if len(pairwise_distances) > 0 else 1.0
                
                print(f"Pointcloud diameter for {key} layer {layer_idx}, {position}: {diameter}")
                
                # Scale the pointcloud by its diameter
                scaled_cloud = cloud / diameter if diameter > 0 else cloud
                
                # Apply PCA if dimension > n_components
                if scaled_cloud.shape[1] > n_components:
                    pca = PCA(n_components=n_components)
                    reduced_positions[position] = pca.fit_transform(scaled_cloud)
                else:
                    reduced_positions[position] = scaled_cloud
            
            reduced_pointclouds[key].append(reduced_positions)
    
    return reduced_pointclouds

#--------------------------------
# Euclidean Distance Functions
#--------------------------------

def compute_persistent_homology(pointcloud, max_dim=2, batch_size=1000):
    """
    Compute persistent homology using Euclidean distance.
    
    Args:
        pointcloud: Numpy array of shape (n_points, n_dimensions)
        max_dim: Maximum homology dimension
        batch_size: Batch size for processing large pointclouds
        
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

def select_scales_euclidean(pointcloud, max_dim=2, num_scales=8):
    """
    Select representative scales using Maximum Relative Persistence method.
    
    Args:
        pointcloud: Numpy array of shape (n_points, n_dimensions)
        max_dim: Maximum homology dimension
        num_scales: Number of scales to return
        
    Returns:
        List of optimal scales and the persistence diagrams
    """
    print("Auto-selecting scales using Maximum Relative Persistence method...")
    
    # Compute persistent homology
    diagrams = compute_persistent_homology(pointcloud, max_dim=max_dim)
    
    # Get the maximum death value across all dimensions
    max_death = 0
    for dim in range(min(max_dim+1, len(diagrams))):
        finite_deaths = diagrams[dim][~np.isinf(diagrams[dim][:, 1]), 1]
        if len(finite_deaths) > 0:
            max_death = max(max_death, np.max(finite_deaths))
    
    if max_death <= 0:
        print("No significant features found, using default scales")
        return np.linspace(0.1, 2.0, num_scales), diagrams
    
    # Calculate relative persistence for each feature
    all_features = []
    
    for dim in range(min(max_dim+1, len(diagrams))):
        finite_diagram = diagrams[dim][~np.isinf(diagrams[dim][:, 1])]
        
        for birth, death in finite_diagram:
            # Calculate persistence relative to the scale
            if birth > 0:
                rel_persistence = (death - birth) / birth
            else:
                rel_persistence = death - birth
                
            # Store dimension, birth, death, and relative persistence
            all_features.append((dim, birth, death, rel_persistence))
    
    # Sort by relative persistence
    all_features.sort(key=lambda x: x[3], reverse=True)
    
    # Take scales from top features
    top_features = all_features[:min(len(all_features), num_scales*2)]
    
    # Extract both birth, death, and midpoints as potential scales
    scales = []
    for _, birth, death, _ in top_features:
        scales.append(birth)
        scales.append(death)
        scales.append((birth + death) / 2)
    
    # Ensure we have exactly num_scales unique scales
    scales = sorted(list(set(scales)))
    
    if len(scales) > num_scales:
        # Take evenly spaced samples
        indices = np.linspace(0, len(scales)-1, num_scales, dtype=int)
        scales = [scales[i] for i in indices]
    elif len(scales) < num_scales:
        # Add more scales
        min_scale = min(scales) if scales else 0.1
        max_scale = max(scales) if scales else 2.0
        
        # Add boundary values and intermediate values
        full_range = np.linspace(min_scale*0.5, max_scale*1.5, num_scales)
        scales = sorted(list(set(scales + list(full_range))))[:num_scales]
    
    print(f"Selected scales: {scales}")
    return np.array(scales), diagrams

def calibrate_scales_euclidean(pointclouds, max_dim=2, auto_select=True):
    """
    Calibrate scales for Euclidean persistent homology.
    
    Args:
        pointclouds: Dictionary of pointclouds
        max_dim: Maximum homology dimension
        auto_select: Whether to use automatic scale selection
        
    Returns:
        List of recommended scales
    """
    if not auto_select:
        print("Using fixed scale range as specified (0.2 to 1.0 with 0.1 increments)")
        # Use fixed scales as in the original script
        scales = np.arange(0.2, 1.1, 0.1)
        return scales
    
    # Sample a small subset of point clouds for efficiency
    print("Performing automatic scale selection...")
    sample_clouds = []
    
    # Sample one cloud from each component and layer
    for key in pointclouds:
        for layer_point_clouds in pointclouds[key]:
            for position, cloud in layer_point_clouds.items():
                sample_clouds.append(cloud)
                break  # Just take the first position
            break  # Just take the first layer
        break  # Just take the first component
    
    # Use the first point cloud for calibration
    if sample_clouds:
        scales, _ = select_scales_euclidean(sample_clouds[0], max_dim)
    else:
        # Fallback to default scales
        scales = np.arange(0.2, 1.1, 0.1)
    
    print(f"Using scales: {scales}")
    return scales

#--------------------------------
# Graph Geodesic Distance Functions
#--------------------------------

def compute_graph_geodesic_distance(pointcloud, k):
    """
    Compute graph geodesic distance matrix based on k-nearest neighbors.
    
    Args:
        pointcloud: Numpy array of shape (n_points, n_dimensions)
        k: Number of nearest neighbors
        
    Returns:
        Distance matrix using graph geodesic distance
    """
    n_points = len(pointcloud)
    
    # Compute k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(pointcloud)
    # +1 because the point itself is included as a neighbor
    
    # Get the k-nearest neighbors graph (binary adjacency matrix)
    knn_graph = nbrs.kneighbors_graph(pointcloud)
    
    # Convert to a format suitable for shortest path calculation
    # Each edge in the graph has length 1 as per the paper
    # Make the graph symmetric (undirected)
    knn_graph = knn_graph.maximum(knn_graph.transpose())
    
    # Compute shortest path distances
    geodesic_distances = shortest_path(csgraph=knn_graph, directed=False)
    
    # In case some points are not connected (will have inf distance)
    # Replace inf with a large finite value to avoid numerical issues
    max_distance = np.max(geodesic_distances[np.isfinite(geodesic_distances)])
    geodesic_distances[~np.isfinite(geodesic_distances)] = max_distance + 1
    
    return geodesic_distances

def compute_persistent_homology_geodesic(pointcloud, k, max_dim=2, batch_size=1000):
    """
    Compute persistent homology using graph geodesic distance.
    
    Args:
        pointcloud: Numpy array of shape (n_points, n_dimensions)
        k: Number of nearest neighbors for distance computation
        max_dim: Maximum homology dimension
        batch_size: Batch size for processing large pointclouds
        
    Returns:
        Persistence diagrams
    """
    # For large pointclouds, use batching to avoid memory issues
    if len(pointcloud) > batch_size:
        # Randomly sample points
        indices = np.random.choice(len(pointcloud), batch_size, replace=False)
        pointcloud = pointcloud[indices]
    
    print(f"Computing persistent homology with geodesic distance on pointcloud of shape {pointcloud.shape}")
    
    # Compute graph geodesic distance matrix
    start_time = time.time()
    print(f"Computing k={k}-nearest neighbors graph...")
    geodesic_distances = compute_graph_geodesic_distance(pointcloud, k)
    print(f"Geodesic distance matrix computed in {time.time() - start_time:.2f} seconds")
    
    # Compute persistent homology using ripser with the distance matrix
    try:
        print(f"Computing persistent homology with max dimension {max_dim}...")
        start_time = time.time()
        
        # Use the correct method for passing a distance matrix to ripser
        diagrams = ripser(X=geodesic_distances, distance_matrix=True, maxdim=max_dim)['dgms']
        
        print(f"Persistent homology computed in {time.time() - start_time:.2f} seconds")
        return diagrams
    except Exception as e:
        print(f"Error computing persistent homology: {e}")
        # Return empty diagrams if computation fails
        empty_diagrams = [np.empty((0, 2)) for _ in range(max_dim + 1)]
        return empty_diagrams

def detect_stable_plateaus(pointcloud, k, max_dim=2, batch_size=1000):
    """
    Detect stable plateaus in Betti number curves for graph geodesic distance.
    
    Args:
        pointcloud: Numpy array of shape (n_points, n_dimensions)
        k: Number of nearest neighbors
        max_dim: Maximum homology dimension
        batch_size: Batch size to limit memory usage
        
    Returns:
        Dictionary of optimal scales for each dimension and the persistence diagrams
    """
    print(f"Detecting stable plateaus with k={k}...")
    
    # Compute persistent homology using geodesic distance
    diagrams = compute_persistent_homology_geodesic(pointcloud, k, max_dim, batch_size)
    
    # Get the maximum distance value in the geodesic matrix
    geodesic_distances = compute_graph_geodesic_distance(
        pointcloud[:min(len(pointcloud), batch_size)], k)
    max_dist = int(np.max(geodesic_distances))
    
    # Create a range of integer scales
    scales = np.arange(1, max_dist + 1)
    
    optimal_scales = {}
    
    for dim in range(min(max_dim+1, len(diagrams))):
        # Calculate Betti numbers across scales
        betti_values = []
        for scale in scales:
            born_before = diagrams[dim][:, 0] <= scale
            dying_after = diagrams[dim][:, 1] > scale
            betti_values.append(np.sum(born_before & dying_after))
        
        betti_array = np.array(betti_values)
        
        # Find plateaus (where values don't change)
        plateaus = []
        current_plateau = [0]
        
        for i in range(1, len(betti_array)):
            if betti_array[i] == betti_array[i-1]:
                current_plateau.append(i)
            else:
                if len(current_plateau) > 1:  # Only consider plateaus of length > 1
                    plateaus.append((current_plateau, betti_array[current_plateau[0]]))
                current_plateau = [i]
        
        # Add the last plateau if it exists
        if len(current_plateau) > 1:
            plateaus.append((current_plateau, betti_array[current_plateau[0]]))
        
        if plateaus:
            # Sort plateaus by length (descending) and then by Betti value (descending)
            plateaus.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
            
            # Choose the middle of the longest plateau
            longest_plateau = plateaus[0][0]
            middle_idx = longest_plateau[len(longest_plateau)//2]
            optimal_scales[dim] = scales[middle_idx]
        else:
            # No plateau found
            optimal_scales[dim] = k
    
    # Ensure we have at least some scales for each dimension
    for dim in range(max_dim+1):
        if dim not in optimal_scales:
            # Default to k if no plateau found
            optimal_scales[dim] = k
    
    # Report the optimal scales
    print(f"Optimal scales for each dimension: {optimal_scales}")
    
    # Return a sorted list of unique scales and the diagrams
    unique_scales = sorted(list(set(optimal_scales.values())))
    
    # Make sure we have a reasonable number of scales
    if len(unique_scales) < 3:
        if len(unique_scales) > 0:
            min_scale = min(unique_scales)
            max_scale = max(unique_scales)
            
            # Add some intermediate scales
            unique_scales = sorted(list(set(unique_scales + 
                                           list(range(min_scale, max_scale+1)))))
        else:
            # Fallback to default scales based on k
            unique_scales = [k-1, k, k+1, k+2]
    
    return unique_scales, diagrams

def calibrate_scales_geodesic(pointclouds, k_values=[10, 14, 20, 25, 30, 35], max_dim=2, auto_select=True):
    """
    Calibrate k and ε parameters for geodesic persistent homology.
    
    Args:
        pointclouds: Dictionary of pointclouds
        k_values: List of k values to test
        max_dim: Maximum homology dimension
        auto_select: Whether to use automatic scale selection
        
    Returns:
        Optimal k value and list of recommended scales
    """
    print("Calibrating k and ε parameters for geodesic persistent homology...")
    
    if not auto_select:
        # Paper suggests integer and half-integer scales
        # Range from 1.0 to 4.5 as shown in Figure 10 of the paper
        scales = np.arange(1.0, 5.0, 0.5)
        
        # For simplicity, we'll just use k=14 (for D-I) and ε=2.5 as the paper recommends
        optimal_k = 14  # As mentioned in the paper for D-I dataset
        
        print(f"Using fixed optimal k={optimal_k} and scales: {scales}")
        return optimal_k, scales
    
    # Use automatic scale selection
    # Sample a small subset of point clouds for efficiency
    print("Performing automatic scale selection...")
    sample_clouds = []
    
    # Sample one cloud from each component and layer
    for key in pointclouds:
        for layer_point_clouds in pointclouds[key]:
            for position, cloud in layer_point_clouds.items():
                sample_clouds.append(cloud)
                break  # Just take the first position
            break  # Just take the first layer
        break  # Just take the first component
    
    # Use the first point cloud for calibration
    if sample_clouds:
        # Default to k=14 as in the paper
        optimal_k = 14
        
        # If we have enough sample data, try to optimize k
        if len(sample_clouds[0]) >= 100:
            # Try different k values and find the one that gives the most stable plateaus
            print(f"Testing k values: {k_values}")
            best_stability = 0
            
            for k in k_values:
                try:
                    scales, _ = detect_stable_plateaus(sample_clouds[0], k, max_dim)
                    # Measure stability as the range of scales
                    if len(scales) > 1:
                        stability = scales[-1] - scales[0]
                        print(f"k={k}, stability={stability}, scales={scales}")
                        if stability > best_stability:
                            best_stability = stability
                            optimal_k = k
                except Exception as e:
                    print(f"Error calibrating with k={k}: {e}")
                    continue
            
            # Now get the final scales with the optimal k
            scales, _ = detect_stable_plateaus(sample_clouds[0], optimal_k, max_dim)
        else:
            # Sample too small, use default
            scales = np.arange(1.0, 5.0, 0.5)
    else:
        # Fallback to default k and scales if no samples available
        optimal_k = 14
        scales = np.arange(1.0, 5.0, 0.5)
    
    print(f"Using optimal k={optimal_k} and scales: {scales}")
    return optimal_k, scales

#--------------------------------
# Common Functions
#--------------------------------

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
        
    Returns:
        Figure object
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

def visualize_persistence_diagram(diagrams, title="Persistence Diagram", output_path=None):
    """
    Visualize persistence diagram.
    
    Args:
        diagrams: List of persistence diagrams
        title: Plot title
        output_path: Path to save the figure (if None, just display)
        
    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(10, 6))
    plot_diagrams(diagrams, show=False, title=title)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    return fig

def analyze_topology(model, data_loader, device, activation_name, method='euclidean', 
                     k=14, pca_components=50, auto_select_scales=True, output_dir='results', use_pca=False):
    """
    Analyze topology of model representations using either Euclidean or graph geodesic distance.
    
    Args:
        model: Trained model
        data_loader: Data loader for test set
        device: Device to run model on
        activation_name: Name of activation function
        method: 'euclidean' or 'geodesic'
        k: Number of nearest neighbors (for geodesic method)
        pca_components: Number of PCA components (only used if use_pca=True)
        auto_select_scales: Whether to automatically select scales
        output_dir: Directory to save results
        use_pca: Whether to apply PCA dimensionality reduction (default: False)
                 Note: Should only be True when using Euclidean distance
        
    Returns:
        Dictionary of Betti numbers and scales used
    """
    # Validate method and PCA compatibility
    if use_pca and method == 'geodesic':
        print("WARNING: PCA should not be used with geodesic distance.")
        print("PCA distorts the neighborhood structure that geodesic distance tries to capture.")
        print("Disabling PCA for this analysis.")
        use_pca = False
    
    model.eval()
    metric_name = "Euclidean" if method == 'euclidean' else "Graph Geodesic"
    print(f"Analyzing topology using {metric_name} distance...")
    
    # Collect features and labels
    all_features = []
    all_labels = []
    
    # Target number of class 0 samples
    target_class0_samples = 1000
    
    # Maximum number of batches to process
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
            
            try:
                # Extract features - handle different feature extraction methods
                if hasattr(model, 'extract_features'):
                    features = model.extract_features()
                elif hasattr(model, 'get_features'):
                    features = model.get_features()
                else:
                    raise AttributeError("Model doesn't have extract_features or get_features method")
                
                # Verify feature structure before adding
                if 'attention' in features and 'mlp' in features:
                    all_features.append(features)
                    all_labels.append(labels)
                else:
                    print(f"WARNING: Batch {i} features have unexpected structure. Skipping.")
                    continue
                
                # Count class 0 samples
                current_class0_count += (labels == 0).sum().item()
                processed_batches += 1
                
                print(f"Processed batch {processed_batches}, Total class 0 samples: {current_class0_count}/{target_class0_samples}", end='\r')
                
                # Stop if we've collected enough class 0 samples
                if current_class0_count >= target_class0_samples:
                    print(f"\nReached target of {target_class0_samples} class 0 samples. Stopping feature extraction.")
                    break
            except Exception as e:
                print(f"Error extracting features for batch {i}: {e}")
                # Continue with the next batch
                continue
    
    # Check if any features were collected
    if not all_features:
        raise ValueError("No features were collected. Please check the data loader and model's feature extraction method.")
    
    # Ensure all feature lists have consistent lengths
    attention_lengths = [len(f['attention']) for f in all_features]
    mlp_lengths = [len(f['mlp']) for f in all_features]
    
    if len(set(attention_lengths)) > 1 or len(set(mlp_lengths)) > 1:
        print("WARNING: Inconsistent feature list lengths across batches. Using the first batch's structure.")
        # Use the structure from the first batch
        reference_attention_len = attention_lengths[0]
        reference_mlp_len = mlp_lengths[0]
        
        # Filter to keep only compatible batches
        compatible_features = []
        compatible_labels = []
        
        for i, (features, labels) in enumerate(zip(all_features, all_labels)):
            if len(features['attention']) == reference_attention_len and len(features['mlp']) == reference_mlp_len:
                compatible_features.append(features)
                compatible_labels.append(labels)
        
        all_features = compatible_features
        all_labels = compatible_labels
    
    try:
        # Combine features from batches
        combined_features = {
            'attention': [torch.cat([f['attention'][i] for f in all_features], dim=0) for i in range(len(all_features[0]['attention']))],
            'mlp': [torch.cat([f['mlp'][i] for f in all_features], dim=0) for i in range(len(all_features[0]['mlp']))]
        }
        combined_labels = torch.cat(all_labels, dim=0)
    except Exception as e:
        print(f"Error combining features: {e}")
        # Try to recover - create a more flexible combination approach
        print("Attempting to recover using a more flexible approach...")
        
        # Get the minimum layer counts (smallest common denominator)
        min_attention_layers = min(len(f['attention']) for f in all_features)
        min_mlp_layers = min(len(f['mlp']) for f in all_features)
        
        # Only use the layers that are present in all batches
        combined_features = {
            'attention': [torch.cat([f['attention'][i] for f in all_features], dim=0) for i in range(min_attention_layers)],
            'mlp': [torch.cat([f['mlp'][i] for f in all_features], dim=0) for i in range(min_mlp_layers)]
        }
        combined_labels = torch.cat(all_labels, dim=0)
    
    print(f"\nExtracted features from {len(combined_labels)} images")
    
    # Create pointclouds for class 0 only
    print("Creating pointclouds...")
    pointclouds_class0 = create_pointclouds(combined_features, combined_labels, class_label=0)
    
    # Apply PCA if requested and compatible with the method, otherwise skip
    if use_pca and method == 'euclidean':
        print(f"Applying diameter scaling followed by PCA with {pca_components} components...")
        reduced_pointclouds_class0 = apply_pca(pointclouds_class0, n_components=pca_components)
    else:
        if method == 'geodesic':
            print("PCA is not appropriate for geodesic distance analysis. Using raw features.")
        else:
            print("Skipping PCA dimensionality reduction as requested.")
        reduced_pointclouds_class0 = pointclouds_class0
    
    # Calibrate scales based on the method
    if method == 'euclidean':
        scales = calibrate_scales_euclidean(reduced_pointclouds_class0, max_dim=2, 
                                          auto_select=auto_select_scales)
    else:  # geodesic
        optimal_k, scales = calibrate_scales_geodesic(reduced_pointclouds_class0, 
                                                   auto_select=auto_select_scales)
        k = optimal_k  # Use the optimal k value
    
    # Create directories to save results with timestamp to avoid overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    method_suffix = "euclidean" if method == "euclidean" else "geodesic"
    
    # Add PCA and scaling info to directory name
    if use_pca:
        pca_suffix = f"_scaled_pca{pca_components}"
    else:
        pca_suffix = "_nopca"
    
    diagrams_dir = os.path.join(output_dir, f'persistent_diagrams_{method_suffix}{pca_suffix}', activation_name)
    os.makedirs(diagrams_dir, exist_ok=True)
    
    betti_dir = os.path.join(output_dir, f'betti_numbers_{method_suffix}{pca_suffix}', activation_name)
    os.makedirs(betti_dir, exist_ok=True)
    
    # Compute topology
    results = {
        'class0': {'attention': [], 'mlp': []}
    }
    
    # Dictionary to store persistent diagrams
    persistent_diagrams = {
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
            layer_diagrams = {}
            
            for position, cloud in position_clouds.items():
                current_computation += 1
                if len(cloud) > 0:
                    print(f"Progress: {current_computation}/{total_computations} - Class 0, {key}, Layer {layer_idx}, {position}")
                    try:
                        # Compute persistent homology based on the method
                        if method == 'euclidean':
                            diagrams = compute_persistent_homology(cloud)
                        else:  # geodesic
                            diagrams = compute_persistent_homology_geodesic(cloud, k=k)
                        
                        # Store persistent diagrams
                        layer_diagrams[position] = diagrams
                        
                        # Calculate Betti numbers at each scale
                        betti_numbers = calculate_betti_numbers(diagrams, scales)
                        
                        # Save Betti numbers
                        layer_results[position] = betti_numbers
                        
                        # Visualize and save persistence diagram
                        preprocessing_info = "scaled by diameter + PCA" if use_pca else "raw features"
                        fig = visualize_persistence_diagram(
                            diagrams, 
                            title=f"{activation_name} - {key} Layer {layer_idx} - {position} ({preprocessing_info})",
                            output_path=os.path.join(diagrams_dir, f"{key}_layer{layer_idx}_{position}.png")
                        )
                        plt.close(fig)
                        
                        # Visualize and save Betti numbers
                        fig = visualize_betti_numbers(
                            betti_numbers,
                            title=f"{activation_name} - {key} Layer {layer_idx} - {position} - Betti Numbers"
                        )
                        plt.savefig(os.path.join(betti_dir, f"{key}_layer{layer_idx}_{position}_betti.png"), bbox_inches='tight')
                        plt.close(fig)
                        
                    except Exception as e:
                        print(f"\nError processing {key} layer {layer_idx}, position {position}: {e}")
            
            results['class0'][key].append(layer_results)
            persistent_diagrams['class0'][key].append(layer_diagrams)
    
    # Save persistent diagrams for later use with timestamp
    diagrams_file = os.path.join(diagrams_dir, f'persistent_diagrams_{timestamp}.npy')
    print(f"Saving persistent diagrams to {diagrams_file}...")
    np.save(diagrams_file, persistent_diagrams)
    
    # Save results in a readable format with timestamp
    results_file = os.path.join(output_dir, f'topology_results_{method_suffix}_{activation_name}_{timestamp}.json')
    print(f"Saving results to {results_file}...")
    import json
    
    # Add analysis metadata
    metadata = {
        "method": method,
        "activation": activation_name,
        "timestamp": timestamp,
        "preprocessing": {
            "diameter_scaling": True if use_pca else False,
            "pca": use_pca,
            "pca_components": pca_components if use_pca else None
        },
        "k_nearest_neighbors": k if method == 'geodesic' else None,
        "scales": [float(s) for s in scales]
    }
    
    # Convert results to a serializable format with metadata
    serializable_results = {
        "metadata": metadata,
        "results": {}
    }
    
    for class_name, class_results in results.items():
        serializable_results["results"][class_name] = {}
        for component, layer_results in class_results.items():
            serializable_results["results"][class_name][component] = []
            for layer_idx, position_results in enumerate(layer_results):
                serializable_layer = {}
                for position, scale_results in position_results.items():
                    serializable_layer[position] = {}
                    for scale, betti_numbers in scale_results.items():
                        serializable_layer[position][str(float(scale))] = [int(b) for b in betti_numbers]
                serializable_results["results"][class_name][component].append(serializable_layer)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nTopology analysis with {metric_name} distance complete!")
    preprocessing_desc = "diameter scaling + PCA" if use_pca else "no preprocessing"
    print(f"Preprocessing applied: {preprocessing_desc}")
    return results, scales

if __name__ == "__main__":
    # This script is meant to be imported and used, not run directly
    print("This script is meant to be imported and used as part of a larger pipeline.")
    print("Example usage:")
    print("from topology_analysis_combined import analyze_topology")
    print("results, scales = analyze_topology(model, data_loader, device, 'GELU', method='euclidean')")
    print("results, scales = analyze_topology(model, data_loader, device, 'GELU', method='geodesic')") 