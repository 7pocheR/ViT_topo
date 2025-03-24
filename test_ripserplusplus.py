import numpy as np

# Try importing ripserplusplus
try:
    import ripserplusplus as rpp_py
    print("Successfully imported ripserplusplus")
except ImportError as e:
    print(f"Failed to import ripserplusplus: {e}")
    exit(1)

# Create a simple point cloud (3D circle)
n_points = 100
theta = np.linspace(0, 2*np.pi, n_points)
point_cloud = np.zeros((n_points, 3))
point_cloud[:, 0] = np.cos(theta)
point_cloud[:, 1] = np.sin(theta)
point_cloud[:, 2] = 0

print(f"Created point cloud with shape: {point_cloud.shape}")

# Try running ripser++
try:
    print("Computing persistence diagrams...")
    diagrams = rpp_py.run_ripserplusplus(point_cloud, max_dim=2)
    
    # Print results
    print("\nPersistence diagrams:")
    for dim, diagram in enumerate(diagrams):
        print(f"Dimension {dim}: {len(diagram)} persistence pairs")
        if len(diagram) > 0:
            print(f"Sample pairs: {diagram[:min(3, len(diagram))]}")
    
    print("\nRipserplusplus test completed successfully!")
except Exception as e:
    print(f"Error running ripserplusplus: {e}")
    exit(1) 