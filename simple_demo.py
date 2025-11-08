#!/usr/bin/env python3
"""
Simple End-to-End Demo for GraphLAMA

This script demonstrates a minimal end-to-end pipeline for GraphLAMA
without requiring large model downloads or extensive dependencies.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
import json
import os

print("=" * 60)
print("GraphLAMA - Simple End-to-End Demo")
print("=" * 60)

# Check if dependencies are available
print("\n[1/5] Checking dependencies...")
try:
    import torch_geometric
    print(f"  ✓ PyTorch: {torch.__version__}")
    print(f"  ✓ PyG: {torch_geometric.__version__}")
except ImportError as e:
    print(f"  ✗ Missing dependency: {e}")
    print("\n  Please install dependencies:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    print("  pip install torch_geometric")
    exit(1)

# Create synthetic graph data for demonstration
print("\n[2/5] Creating synthetic graph data...")
num_nodes = 10
num_edges = 20
num_features = 8
num_classes = 3

# Random node features
x = torch.randn(num_nodes, num_features)

# Random edges (for simplicity, create a simple graph)
edge_index = torch.randint(0, num_nodes, (2, num_edges))

# Random labels
y = torch.randint(0, num_classes, (num_nodes,))

graph_data = Data(x=x, edge_index=edge_index, y=y)
print(f"  ✓ Created graph with {num_nodes} nodes and {num_edges} edges")
print(f"  ✓ Node features: {num_features}D, Classes: {num_classes}")

# Simple GNN model for demonstration
print("\n[3/5] Creating simple GNN model...")
class SimpleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGNN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x

model = SimpleGNN(num_features, 16, num_classes)
print(f"  ✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Demonstrate forward pass
print("\n[4/5] Running forward pass...")
with torch.no_grad():
    output = model(graph_data.x)
    predictions = output.argmax(dim=1)
print(f"  ✓ Forward pass completed")
print(f"  ✓ Output shape: {output.shape}")
print(f"  ✓ Predictions: {predictions.tolist()}")

# Create a sample conversation template (GraphLAMA style)
print("\n[5/5] Creating sample conversation format...")
sample_conversation = {
    "id": "demo_001",
    "graph": {
        "node_idx": 0,
        "num_nodes": num_nodes,
        "num_edges": num_edges
    },
    "conversations": [
        {
            "from": "human",
            "value": "What is the category of this graph node?"
        },
        {
            "from": "gpt",
            "value": f"Based on the graph structure, this node belongs to class {predictions[0].item()}."
        }
    ]
}

output_file = "demo_output.json"
with open(output_file, 'w') as f:
    json.dump(sample_conversation, f, indent=2)

print(f"  ✓ Sample conversation saved to {output_file}")
print("\n" + "=" * 60)
print("Demo completed successfully!")
print("=" * 60)
print("\nThis demonstrates the basic structure of GraphLAMA:")
print("1. Graph data representation (PyG Data objects)")
print("2. GNN model for node feature processing")
print("3. Conversation format for graph-based QA")
print("\nFor full functionality, please refer to README.md")
print("and ensure all dependencies from requirements.txt are installed.")
