import numpy as np

# Try importing ripserplusplus
try:
    import ripserplusplus as rpp_py
    print("Successfully imported ripserplusplus")
except ImportError as e:
    print(f"Failed to import ripserplusplus: {e}")
    exit(1)

# Print available functions in the module
print("Available functions in ripserplusplus:")
for attr in dir(rpp_py):
    if not attr.startswith('__'):
        print(f"  - {attr}")

# Create a simple point cloud (3D circle)
n_points = 100
theta = np.linspace(0, 2*np.pi, n_points)
point_cloud = np.zeros((n_points, 3))
point_cloud[:, 0] = np.cos(theta)
point_cloud[:, 1] = np.sin(theta)
point_cloud[:, 2] = 0

print(f"Created point cloud with shape: {point_cloud.shape}")

# Try running ripser++ with the correct API
try:
    print("Computing persistence diagrams...")
    # Try different API calls based on what's available
    if hasattr(rpp_py, 'ripser'):
        diagrams = rpp_py.ripser(point_cloud, maxdim=2)
        print("Used rpp_py.ripser() function")
    elif hasattr(rpp_py, 'run'):
        # Original API uses run function with command string
        cmd = f"--format point-cloud --dim 2"
        diagrams = rpp_py.run(cmd, point_cloud)
        print("Used rpp_py.run() function")
    elif hasattr(rpp_py, 'rips_dm'):
        # Direct distance matrix API
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(point_cloud))
        diagrams = rpp_py.rips_dm(distances, maxdim=2)
        print("Used rpp_py.rips_dm() function")
    else:
        print("Could not find appropriate function in ripserplusplus module")
        print("Please check the documentation for the correct API")
        exit(1)
    
    # Print results
    if isinstance(diagrams, dict):
        print("\nPersistence diagrams (dictionary format):")
        for key, value in diagrams.items():
            print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
    elif isinstance(diagrams, list):
        print("\nPersistence diagrams (list format):")
        for dim, diagram in enumerate(diagrams):
            print(f"Dimension {dim}: {len(diagram)} persistence pairs")
            if len(diagram) > 0:
                print(f"Sample pairs: {diagram[:min(3, len(diagram))]}")
    else:
        print(f"\nUnknown format: {type(diagrams)}")
        print(diagrams)
    
    print("\nRipserplusplus test completed successfully!")
except Exception as e:
    print(f"Error running ripserplusplus: {e}")
    exit(1) 