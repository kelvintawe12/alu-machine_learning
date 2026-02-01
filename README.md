# alu-machine_learning

This repository contains machine learning and mathematics projects, exercises, and solutions for ALU coursework.

## Structure
- `math/plotting/`: Python scripts for various plotting and data visualization tasks.
- `supervised_learning/classification/`: Python implementations of single neurons, shallow and deep neural networks, and related utilities (one-hot encoding, model save/load, etc.).
- `supervised_learning/tensorflow/`: TensorFlow 1.x scripts for building, training, and evaluating neural networks from scratch, including:
  - Placeholder creation
  - Layer creation with He initialization
  - Forward propagation
  - Accuracy and loss calculation
  - Training operation and full training loop
  - Model saving and evaluation
- `math/`: Datasets and supporting files for experiments.

## Requirements
- Python 3.5
- numpy 1.15
- tensorflow 1.12
- All code is PEP8 (pycodestyle 2.4) compliant
- All files are executable and have full documentation

## Usage
1. **Install dependencies:**
   ```sh
   pip install numpy==1.15 tensorflow==1.12
   ```
2. **Run classification or TensorFlow scripts** as described in each subfolder's README or by following the main scripts provided.

## Project Standards
- Each module, class, and function is fully documented.
- All code is compatible with Ubuntu 16.04 LTS and allowed editors (vi, vim, emacs).
- No Keras or unsupported imports are used.

## Author
kelvintawe12

---
For more details, see the README files in each subdirectory.
