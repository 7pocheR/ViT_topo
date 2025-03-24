#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Vision Transformer (ViT) implementation with hooks for feature extraction.
Supports both GELU and ReLU activation functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and embeds them.
    """
    def __init__(self, img_size, patch_size, in_channels, embedding_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection
        self.projection = nn.Conv2d(
            in_channels, embedding_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        x = self.projection(x)  # [batch_size, embedding_dim, grid_size, grid_size]
        x = x.flatten(2)        # [batch_size, embedding_dim, n_patches]
        x = x.transpose(1, 2)   # [batch_size, n_patches, embedding_dim]
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        
        # For feature extraction
        self.attention_output = None
    
    def forward(self, x):
        batch_size, n_tokens, _ = x.shape
        
        # qkv: [batch_size, n_tokens, 3*embedding_dim]
        qkv = self.qkv(x)
        
        # Reshape: [batch_size, n_tokens, 3, num_heads, head_dim]
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.num_heads, self.head_dim)
        
        # Permute: [3, batch_size, num_heads, n_tokens, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Unpack Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        # [batch_size, num_heads, n_tokens, n_tokens]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to V
        # [batch_size, num_heads, n_tokens, head_dim]
        x = (attn @ v)
        
        # Reshape: [batch_size, n_tokens, embedding_dim]
        x = x.transpose(1, 2).reshape(batch_size, n_tokens, self.embedding_dim)
        
        # Project back to embedding_dim
        x = self.proj(x)
        
        # Store output for hook
        self.attention_output = x
        
        return x

class MLP(nn.Module):
    """
    MLP block with configurable activation function.
    """
    def __init__(self, embedding_dim, mlp_ratio=1.5, activation='gelu'):
        super().__init__()
        hidden_dim = int(embedding_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        
        # Set activation function
        if activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # For feature extraction
        self.mlp_output = None
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        # Store output for hook
        self.mlp_output = x
        
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block with attention and MLP.
    """
    def __init__(self, embedding_dim, num_heads, mlp_ratio=1.5, activation='gelu'):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads)
        
        # MLP block
        self.mlp = MLP(embedding_dim, mlp_ratio, activation)
        
        # For feature extraction
        self.attention_output = None
        self.mlp_output = None
    
    def forward(self, x):
        # First residual block - attention
        attn_output = self.attention(self.norm1(x))
        x = x + attn_output
        self.attention_output = x.clone()  # Store for feature extraction
        
        # Second residual block - MLP
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        self.mlp_output = x.clone()  # Store for feature extraction
        
        return x

class SimpleViT(nn.Module):
    """
    Simple Vision Transformer for MNIST classification.
    """
    def __init__(
        self,
        img_size=28,
        patch_size=4,
        in_channels=1,
        num_classes=1,  # Binary classification
        embedding_dim=48,
        num_heads=4,
        num_transformer_blocks=4,
        mlp_ratio=1.5,
        activation='gelu'
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim
        )
        
        # Number of patches
        self.n_patches = self.patch_embed.n_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, embedding_dim)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                activation=activation
            )
            for _ in range(num_transformer_blocks)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embedding_dim)
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize patch embedding weights
        nn.init.normal_(self.patch_embed.projection.weight, std=0.02)
        
        # Initialize cls token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize classification head
        nn.init.zeros_(self.classifier.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
    
    def extract_features(self):
        """
        Extract features from all extraction points.
        Returns a dictionary with features from both attention and MLP outputs.
        """
        features = {
            'attention': [],  # Features after attention blocks
            'mlp': []         # Features after MLP blocks
        }
        
        for block in self.transformer_blocks:
            if hasattr(block, 'attention_output') and block.attention_output is not None:
                features['attention'].append(block.attention_output)
            
            if hasattr(block, 'mlp_output') and block.mlp_output is not None:
                features['mlp'].append(block.mlp_output)
        
        return features
    
    def forward(self, x):
        # Get batch size
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Extract class token for classification
        cls_token_final = x[:, 0]
        
        # Classification
        x = self.classifier(cls_token_final)
        
        return x 