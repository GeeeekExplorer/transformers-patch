# Transformers-Patch 🛠️

Memory optimization patches for HuggingFace Transformers.

## Features ✨

* **Memory Reduction** - Significantly lowers memory usage in Transformers models

* **Zero Configuration** - Works automatically after import

## Installation ⚡

```bash
pip install git+https://github.com/GeeeekExplorer/transformers-patch.git
```

## Quick Start 🚀

Just import the patch **before** loading any Transformers models:

```python
import transformers_patch
from transformers import AutoModel
```

## Benchmark 📊

**Test Configuration**:
* 8x GPU machine
* Micro batch size: 1
* Sequence length: 4096
* Gradient checkpointing: Disabled
* Model: Qwen3-8B

| Memory Component           | Fixed Allocation | Before Patch | After Patch |
|----------------------------|------------------|--------------|-------------|
| Model + Gradients          | 30.5 GB          | -            | -           |
| ZeRO Optimizer States      | 11.4 GB          | -            | -           |
| **Activations**            | -                | 37.7 GB      | 16.7 GB     |
| **Total Memory Allocated** | -                | 79.6 GB      | 58.6 GB     |

✅ **55%** reduction in activation memory

✅ **26%** reduction in total memory usage

## Example Usage 📋

See complete example in `train.py`.

## Acknowledgements 🙏

* [unsloth](https://github.com/unslothai/unsloth)