#!/usr/bin/env python3
"""
End-to-End Example with Cora Dataset

This script demonstrates the GraphLAMA pipeline using the Cora dataset:
1. Load Cora dataset
2. Prepare graph data
3. Create training/test splits
4. Generate conversation format data
"""

import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import json
import os
from pathlib import Path
import random

print("=" * 70)
print("GraphLAMA - Cora Dataset End-to-End Example")
print("=" * 70)

# Categories for Cora dataset
CORA_CATEGORIES = [
    "Case_Based",
    "Genetic_Algorithms", 
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory"
]

# Step 1: Load Cora dataset
print("\n[1/6] Loading Cora dataset...")
data_dir = Path("./data/cora_raw")
data_dir.mkdir(parents=True, exist_ok=True)

try:
    dataset = Planetoid(root=str(data_dir), name='Cora')
    data = dataset[0]
    print(f"  ✓ Loaded Cora dataset")
    print(f"  ✓ Nodes: {data.num_nodes}")
    print(f"  ✓ Edges: {data.num_edges}")
    print(f"  ✓ Features: {data.num_features}")
    print(f"  ✓ Classes: {dataset.num_classes}")
except Exception as e:
    print(f"  ✗ Error loading dataset: {e}")
    print("  Creating synthetic Cora-like data instead...")
    
    # Create synthetic data if download fails
    num_nodes = 2708
    num_features = 1433
    num_classes = 7
    num_edges = 5429 * 2  # Undirected edges
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, num_classes, (num_nodes,))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:140] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[1000:1500] = True
    
    data = Data(x=x, edge_index=edge_index, y=y, 
                train_mask=train_mask, test_mask=test_mask)
    dataset = type('Dataset', (), {'num_classes': num_classes})()
    print(f"  ✓ Created synthetic Cora-like dataset")

# Step 2: Create train/test splits
print("\n[2/6] Creating train/test splits...")
train_indices = data.train_mask.nonzero(as_tuple=True)[0].tolist()
test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()

# Sample a subset for demonstration
num_train_samples = min(20, len(train_indices))
num_test_samples = min(50, len(test_indices))

sampled_train = random.sample(train_indices, num_train_samples)
sampled_test = random.sample(test_indices, num_test_samples)

print(f"  ✓ Train samples: {num_train_samples}")
print(f"  ✓ Test samples: {num_test_samples}")

# Step 3: Generate conversation format data
print("\n[3/6] Generating conversation format data...")

def create_conversation_item(node_idx, label, is_train=True):
    """Create a conversation item for a graph node."""
    category = CORA_CATEGORIES[label % len(CORA_CATEGORIES)]
    
    # Question prompt
    question_prompt = (
        "Given a citation graph where nodes represent papers and "
        "edges represent citations between papers. Each paper has word features. "
        "What is the research category of the target paper?\n\n"
        "Available categories:\n" + 
        "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(CORA_CATEGORIES)])
    )
    
    # Answer
    answer = f"The paper belongs to the category: {category}"
    
    return {
        "id": f"cora_{'train' if is_train else 'test'}_{node_idx}",
        "graph": {
            "node_idx": int(node_idx),
            "num_nodes": data.num_nodes,
            "num_edges": data.num_edges
        },
        "conversations": [
            {
                "from": "human",
                "value": question_prompt
            },
            {
                "from": "gpt",
                "value": answer
            }
        ],
        "label": category
    }

train_items = [create_conversation_item(idx, data.y[idx].item(), is_train=True) 
               for idx in sampled_train]
test_items = [create_conversation_item(idx, data.y[idx].item(), is_train=False) 
              for idx in sampled_test]

print(f"  ✓ Created {len(train_items)} training conversations")
print(f"  ✓ Created {len(test_items)} test conversations")

# Step 4: Save formatted data
print("\n[4/6] Saving formatted data...")
output_dir = Path("./data/cora_processed")
output_dir.mkdir(parents=True, exist_ok=True)

train_file = output_dir / "train_items.json"
test_file = output_dir / "test_items.json"

with open(train_file, 'w') as f:
    json.dump(train_items, f, indent=2)

with open(test_file, 'w') as f:
    json.dump(test_items, f, indent=2)

print(f"  ✓ Saved train data to {train_file}")
print(f"  ✓ Saved test data to {test_file}")

# Step 5: Save graph data
print("\n[5/6] Saving graph data...")
graph_data_file = output_dir / "graph_data.pt"
torch.save({
    'cora': data,
    'num_nodes': data.num_nodes,
    'num_edges': data.num_edges,
    'num_features': data.num_features,
    'num_classes': dataset.num_classes
}, graph_data_file)

print(f"  ✓ Saved graph data to {graph_data_file}")

# Step 6: Display sample
print("\n[6/6] Sample conversation:")
print("-" * 70)
sample = train_items[0]
print(f"ID: {sample['id']}")
print(f"Node Index: {sample['graph']['node_idx']}")
print(f"\nHuman: {sample['conversations'][0]['value'][:100]}...")
print(f"\nGPT: {sample['conversations'][1]['value']}")
print("-" * 70)

print("\n" + "=" * 70)
print("✓ End-to-end data preparation completed successfully!")
print("=" * 70)

print("\nGenerated files:")
print(f"  1. {train_file}")
print(f"  2. {test_file}")
print(f"  3. {graph_data_file}")

print("\nNext steps:")
print("  - Review the generated conversation format data")
print("  - Use these files for training with grapht3/train/train_graph.py")
print("  - Evaluate with grapht3/eval scripts")
print("\nFor full training, please ensure you have:")
print("  - A pretrained LLM (e.g., Vicuna)")
print("  - Sufficient GPU resources")
print("  - All dependencies from requirements.txt installed")
