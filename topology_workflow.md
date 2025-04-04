# Vision Transformer Topology Analysis Workflow

The following flowchart explains the workflow of `topology_analysis.py`, which analyzes the topological features of Vision Transformer representations.

```mermaid
flowchart TD
    Start([Start Analysis]) --> LoadModel[Load Trained ViT Model]
    LoadModel --> ExtractFeatures[Extract Features from Test Data]
    
    subgraph FeatureCollection[Feature Collection]
        ExtractFeatures --> A{Enough\nClass 0\nSamples?}
        A -->|No| ProcessBatch[Process Next Batch]
        ProcessBatch --> CountClass0[Count Class 0 in Batch]
        CountClass0 --> ForwardPass[Forward Pass through Model]
        ForwardPass --> StoreFeatures[Store Attention & MLP Features]
        StoreFeatures --> UpdateCount[Update Class 0 Count]
        UpdateCount --> A
    end
    
    A -->|Yes| CombineFeatures[Combine Features from All Batches]
    
    CombineFeatures --> FilterClass0[Filter for Class 0 Samples Only]
    FilterClass0 --> B{Have 1000\nClass 0\nSamples?}
    
    B -->|Yes| CreatePointclouds[Create Pointclouds as-is]
    B -->|No| SystematicSampling[Apply Systematic Sampling]
    
    subgraph SamplingProcess[Sampling Process]
        SystematicSampling --> C{Samples < 1000?}
        C -->|Yes| Upsample[Repeat Points Systematically to Reach 1000]
        C -->|No| Downsample[Use Evenly Spaced Indices]
    end
    
    CreatePointclouds --> PositionExtraction[Extract 3 Token Positions:
    1. CLS Token
    2. Central Patch
    3. Top-Left Patch]
    
    SystematicSampling --> PositionExtraction
    
    PositionExtraction --> ApplyPCA[Apply PCA to Reduce Dimensions]
    
    ApplyPCA --> CalibrateScales[Calibrate Scales for Persistent Homology]
    
    CalibrateScales --> ComputeTopology[Compute Persistent Homology]
    
    ComputeTopology --> CalculateBetti[Calculate Betti Numbers]
    
    CalculateBetti --> VisualizeResults[Visualize Results]
    
    VisualizeResults --> End([End Analysis])
```

## Process Details

1. **Feature Extraction**: 
   - Process batches of test data through the model
   - Extract attention and MLP features from each layer
   - Continue until we have at least 1000 class 0 samples or reach batch limit

2. **Pointcloud Creation**:
   - Filter features to keep only class 0 samples
   - For each layer and feature type (attention/MLP), create pointclouds for:
     - CLS token
     - Central patch (middle of image)
     - Top-left corner patch

3. **Systematic Sampling**:
   - If we have fewer than 1000 class 0 samples, systematically repeat points to reach 1000
   - If we have more than 1000 class 0 samples, use evenly spaced sampling
   - This ensures consistent pointcloud size for all analyses

4. **Topology Analysis**:
   - Apply PCA to reduce dimensionality
   - Calibrate appropriate scales for persistent homology
   - Compute persistent homology for each pointcloud
   - Calculate Betti numbers at different scales

5. **Result Visualization**:
   - Generate plots of Betti numbers across layers
   - Save numerical results as JSON

This explains the seemingly paradoxical message "Using systematic sampling to reach 1000 points from 980 samples" - the code uses systematic upsampling (repeating some points) to reach exactly 1000 points when we have fewer than 1000 samples available. 