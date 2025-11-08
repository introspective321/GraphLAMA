# Running GraphLAMA End-to-End

This guide provides step-by-step instructions to run GraphLAMA code from start to finish.

## Quick Start (Minimal Dependencies)

For a quick demonstration without heavy dependencies:

```bash
# Install minimal dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric transformers

# Run simple demo
python3 simple_demo.py
```

This will create a synthetic graph and demonstrate the basic GraphLAMA structure.

## Cora Dataset Example

To run the full Cora dataset example:

```bash
# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric transformers sentencepiece

# Run Cora data preparation
python3 run_cora_example.py
```

This will:
1. Download the Cora citation network dataset
2. Create train/test splits
3. Generate conversation format data
4. Save processed data to `data/cora_processed/`

## Complete End-to-End Pipeline

For the full GraphLAMA pipeline with training and evaluation:

### 1. Environment Setup

```bash
# Create conda environment
conda create -n graphlama python=3.8
conda activate graphlama

# Install PyTorch with CUDA (for GPU support)
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install PyG and extensions
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# Install project dependencies
pip install -r requirements.txt

# Optional: Install FastChat for Vicuna support
pip3 install "fschat[model_worker,webui]"
```

### 2. Data Preparation

```bash
# Run Cora example to generate formatted data
python3 run_cora_example.py

# Or use the existing data preparation script
cd data/cora
python reshape_cora.py
```

### 3. Pre-training (Stage 1 & 2)

```bash
cd grapht3

# Stage 1: Train graph projector
bash ../scripts/tune_script/stage1.sh

# Stage 2: Fine-tune with graph-text alignment
bash ../scripts/tune_script/stage2.sh
```

**Note:** You'll need to:
- Update paths in the scripts to point to your model and data locations
- Have a pretrained LLM (e.g., Vicuna-7B) downloaded
- Have sufficient GPU memory (recommended: >24GB)

### 4. Test-time Tuning

```bash
# Fine-tune on specific tasks
bash scripts/tune_script/SFTonGFM_train.sh
```

### 5. Evaluation

```bash
# Evaluate on test set
bash scripts/tune_script/SFTonGFM_eval.sh
```

## File Structure

After running the examples, you'll have:

```
GraphLAMA/
├── data/
│   ├── cora_processed/          # Generated from run_cora_example.py
│   │   ├── train_items.json     # Training conversations
│   │   ├── test_items.json      # Test conversations
│   │   └── graph_data.pt        # Graph structure
│   └── cora_raw/                # Downloaded Cora dataset
├── demo_output.json             # Output from simple_demo.py
├── simple_demo.py               # Minimal demo
└── run_cora_example.py          # Cora data preparation
```

## Troubleshooting

### Missing Dependencies

If you encounter `ModuleNotFoundError`, install the missing package:

```bash
pip install <package-name>
```

Common missing packages:
- `torch-scatter`, `torch-sparse`: PyG extensions (use appropriate wheel for your PyTorch/CUDA version)
- `transformers`: Hugging Face transformers
- `sentencepiece`: Tokenization
- `fschat`: FastChat for Vicuna support

### GPU/CUDA Issues

For CPU-only mode:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Path Configuration

Update hardcoded paths in scripts:
- Replace `/home/cjz/...` with your actual paths
- Set `PYTHONPATH` to your GraphLAMA directory
- Update model paths in shell scripts

## Example Outputs

### simple_demo.py
Creates `demo_output.json` with a sample conversation format.

### run_cora_example.py
Generates:
- `data/cora_processed/train_items.json`: 20 training samples
- `data/cora_processed/test_items.json`: 50 test samples
- `data/cora_processed/graph_data.pt`: Full Cora graph

## Next Steps

1. **Quick Test**: Run `simple_demo.py` to verify installation
2. **Data Prep**: Run `run_cora_example.py` to prepare Cora dataset
3. **Review Data**: Check generated JSON files to understand format
4. **Full Training**: Configure paths in shell scripts and run training pipeline
5. **Evaluation**: Use evaluation scripts to test model performance

## Resources

- Paper: [GraphLAMA on arXiv](https://arxiv.org/pdf/2506.21559)
- Main README: See `README.md` for detailed project information
- GraphGPT: https://github.com/HKUDS/GraphGPT (predecessor project)
