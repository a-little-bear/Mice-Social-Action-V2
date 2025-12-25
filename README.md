# MABe Mouse Behavior Detection - Project Structure

This project implements a **Multi-View Hierarchical Spatio-Temporal Fusion Framework** designed for the MABe Mouse Behavior Detection competition. The structure is modular, scalable, and supports the ablation studies described in the competition analysis.

## Directory Structure

```
mice_social_action_new/
├── configs/                # Configuration files for ablation studies
│   └── base_config.yaml    # Main config controlling all modules
├── data/                   # Raw competition data
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                # Entry points
│   └── train.py            # Training script
├── src/                    # Source code
│   ├── data/               # Module A: Preprocessing & Augmentation
│   │   ├── dataset.py      # Dataset loader
│   │   └── transforms.py   # Coordinate transforms & Augmentation
│   ├── models/             # Module B & C: Architecture
│   │   ├── encoders/       # Temporal & Topology Encoders
│   │   ├── heads/          # Classification Heads
│   │   └── fusion_model.py # Dual-Stream Fusion Model
│   ├── postprocessing/     # Module D: Optimization
│   │   └── optimization.py # Thresholding & Smoothing
│   ├── training/           # Training loops & utilities
│   └── utils/              # Helper functions
└── README.md
```

## Installation

1.  **Environment Setup**:
    ```bash
    conda create -n mabe python=3.10
    conda activate mabe
    ```

2.  **Dependencies**:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pandas numpy pyarrow fastparquet tqdm pyyaml scikit-learn
    ```

## Usage

### Training
To start training with the default configuration:
```bash
python scripts/train.py
```

### Configuration
Modify `configs/base_config.yaml` to change model architecture, data paths, or training hyperparameters.

## Performance Tuning & Hardware Recommendations

### Recommended Hardware
*   **GPU**: NVIDIA RTX 6000 Ada (96GB) or RTX 4090/5090 (24GB+).
    *   *Note*: For 96GB VRAM, you can significantly increase `batch_size` (e.g., 512 or 1024).
*   **RAM**: 64GB+ (128GB+ recommended for full dataset preloading).
*   **Storage**: NVMe SSD is critical if not using RAM preloading.

### Optimization Settings (in `configs/base_config.yaml`)

For high-end workstations (e.g., 22-core CPU, 96GB VRAM), use the following settings to maximize utilization:

```yaml
data:
  # ...
  batch_size: 512       # Increase to fill VRAM (96GB can handle 1024+)
  num_workers: 16       # Set to ~70% of CPU cores for fast data augmentation
  preload: true         # CRITICAL: Loads all data into RAM to eliminate disk I/O bottleneck
  cache_size: 5000      # Keep high when preload is true
```

**Performance Note**: Enabling `preload: true` on a system with sufficient RAM can speed up training by **10-50x** by avoiding repeated disk reads and Parquet parsing.

## Module Mapping

### Module A: Preprocessing & Augmentation
- **Files**: `src/data/transforms.py`, `src/data/dataset.py`
- **Features**:
    - **Coordinate Transform**: Implemented in `CoordinateTransform` class. Handles Agent-centric conversion.
    - **Keypoint Unification**: Implemented in `KeypointUnification` class.
    - **Augmentation**: Implemented in `Augmentation` class (Flip, Rotate, Time Stretch).
- **Config**: Controlled by `data.preprocessing` and `data.augmentation` in `base_config.yaml`.

### Module B: Dual-Stream Encoder
- **Files**: `src/models/encoders/`
- **Features**:
    - **Stream 1 (Topology)**: `TopologyEncoder` (Supports ST-GCN, GAT).
    - **Stream 2 (Temporal)**: `TemporalEncoder` (Supports 1D-CNN, Transformer, WaveNet).
- **Config**: Controlled by `model.stream1_topology` and `model.stream2_temporal`.

### Module C: Fusion & Classification Head
- **Files**: `src/models/fusion_model.py`
- **Features**:
    - Combines outputs from both streams.
    - Supports Concatenation or Stacking.
- **Config**: Controlled by `model.fusion`.

### Module D: Post-Processing Optimization
- **Files**: `src/postprocessing/optimization.py`
- **Features**:
    - **Lab-Specific Thresholding**: `optimize_thresholds` method.
    - **Smoothing**: `apply_smoothing` (Median Filter).
    - **Gap Filling**: `fill_gaps`.
- **Config**: Controlled by `post_processing`.
