# Vision Transformers (ViT) Explained

This document explains how Vision Transformers work for image classification, particularly for readers who understand neural networks and language transformers but are new to Vision Transformers.

## From Language to Vision: The Transformer Journey

```
┌────────────────────┐     ┌────────────────────┐
│ Language           │     │ Vision             │
│ Transformer        │     │ Transformer        │
│                    │     │                    │
│ [WORDS] → [OUTPUT] │     │ [PATCHES] → [CLASS]│
└────────────────────┘     └────────────────────┘
```

### Key Insight

The fundamental insight of Vision Transformers (ViT) is treating **image patches like words**. Just as language transformers process sequences of word tokens, vision transformers process sequences of image patches.

## ViT Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────┐    ┌─────────┐    ┌───────────────┐    ┌─────────┐│
│  │Image│ → │Patchify │ → │Transformer     │ → │MLP Head ││
│  │     │    │+ Embed  │    │Encoder Blocks │    │        ││
│  └─────┘    └─────────┘    └───────────────┘    └─────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 1. Image Patching and Embedding

```
Input Image (28×28)        Image Patches            Patch Embeddings
┌──────────────┐          ┌───┬───┬───┬───┐         ┌───┬───┬───┬───┐
│              │          │P₁ │P₂ │P₃ │P₄ │         │E₁ │E₂ │E₃ │E₄ │
│              │          ├───┼───┼───┼───┤         ├───┼───┼───┼───┤
│              │    →     │P₅ │P₆ │P₇ │P₈ │    →    │E₅ │E₆ │E₇ │E₈ │
│              │          ├───┼───┼───┼───┤         ├───┼───┼───┼───┤
│              │          │...|...|...|...│         │...|...|...|...│
└──────────────┘          └───┴───┴───┴───┘         └───┴───┴───┴───┘
                             7×7 grid of              Embedded as
                             4×4 patches             vectors (dim=16)
```

### Process:
1. Split 28×28 MNIST images into 49 patches (each 4×4 pixels)
2. Flatten each patch into a vector
3. Project each vector to embedding dimension (d=16)
4. Add positional embeddings to retain spatial information

## 2. The CLS Token: Classification Specialist

```
Patch Sequence with CLS Token
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ CLS │ P₁  │ P₂  │ ... │ P₄₈ │ P₄₉ │
└─────┴─────┴─────┴─────┴─────┴─────┘
   ↑
Special token that will
collect classification
information
```

### The CLS Token Is Special:
- Added as the first token in the sequence
- Initialized with learnable parameters
- Through self-attention, it aggregates information from all patches
- The final state of this token is used for classification

## 3. Transformer Encoder Blocks

```
                   ┌───────────────────────┐
                   │  Transformer Block    │
                   │                       │
Input              │   ┌───────────────┐   │               Output
Tokens    ┌───┐    │   │Multi-Head     │   │    ┌───┐     Tokens
[CLS,P₁...P₄₉] →   │ → │Self-Attention │ → │  → [CLS',P₁'...P₄₉']
          └───┘    │   └───────────────┘   │    └───┘
                   │          ↓            │
                   │   ┌───────────────┐   │
                   │   │   MLP Block   │   │
                   │   └───────────────┘   │
                   │                       │
                   └───────────────────────┘
                          × 12 blocks
```

### Self-Attention Mechanism:
- Each patch can "look at" all other patches
- Calculates attention weights between every pair of patches
- Crucial for understanding global image relationships

```
┌─────┬─────┬─────┬─────┬─────┐
│     │ CLS │ P₁  │ P₂  │ P₃  │
├─────┼─────┼─────┼─────┼─────┤
│ CLS │  •  │  •  │  •  │  •  │
├─────┼─────┼─────┼─────┼─────┤
│ P₁  │  •  │  •  │  •  │  •  │
├─────┼─────┼─────┼─────┼─────┤
│ P₂  │  •  │  •  │  •  │  •  │
├─────┼─────┼─────┼─────┼─────┤
│ P₃  │  •  │  •  │  •  │  •  │
└─────┴─────┴─────┴─────┴─────┘
  Attention weights matrix
```

### MLP Block:
- Processes each token independently 
- Applies non-linearities (GELU or ReLU in our case)
- Expands and contracts dimensions (using expansion factor of 1.5)

## 4. Classification Head

```
Final CLS Token      MLP Classification Head      Output
┌─────────┐          ┌───────────────────┐       ┌─────┐
│         │          │                   │       │     │
│  CLS'   │    →     │    Linear → GELU  │   →   │  0  │
│         │          │    → Linear       │       │     │
└─────────┘          └───────────────────┘       └─────┘
  dim=16                                         Binary
                                              Classification
```

### Binary Classification:
- Only the final CLS token is used for classification
- A simple MLP projects it to a scalar (logit)
- Binary classification (0 vs. non-0) using sigmoid activation

## Our Specific Architecture

```
SimpleViT Parameters:
- Image size: 28×28 (MNIST)
- Patch size: 4×4
- Embedding dimension: 16
- Transformer blocks: 12
- Attention heads: 4
- MLP ratio: 1.5
- Activation: GELU or ReLU
```

## How It Works in Our Task

### 1. Preprocessing
- MNIST images are normalized
- Binary classification: digit 0 vs. all others (1-9)
- Dataset is balanced (equal number of 0 and non-0 examples)

### 2. Forward Pass
- Image → Patches → Embeddings + CLS token
- 12 transformer blocks process the tokens
- CLS token gathers information through self-attention
- Classification head makes binary prediction

### 3. Training
- BCEWithLogitsLoss for binary classification
- Adam optimizer
- Early stopping at 99% accuracy
- Two models trained: one with GELU, one with ReLU activation

## The Topology Analysis Connection

For our topology analysis, we:
1. Extract features from the attention layers and MLP blocks
2. Create point clouds from these features
3. Apply persistent homology to understand topological structures
4. Track how these structures change across network layers

The CLS token is particularly important as it contains the most classification-relevant information.

## Advantages over CNNs

```
CNN                           ViT
┌───────────────────┐        ┌───────────────────┐
│ Local Receptive   │        │ Global Receptive  │
│ Field             │        │ Field Immediately │
└───────────────────┘        └───────────────────┘
┌───────────────────┐        ┌───────────────────┐
│ Hierarchical      │        │ Flat, Non-        │
│ Feature Learning  │        │ Hierarchical      │
└───────────────────┘        └───────────────────┘
┌───────────────────┐        ┌───────────────────┐
│ Translation       │        │ Position-aware    │
│ Invariance        │        │ Processing        │
└───────────────────┘        └───────────────────┘
```

- ViTs capture long-range dependencies from the beginning
- Less inductive bias, more flexible feature learning
- Particularly strong with sufficient training data 