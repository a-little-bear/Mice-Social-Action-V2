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

## Usage

1.  **Configure**: Edit `configs/base_config.yaml` to select model types and toggle features for ablation.
2.  **Train**: Run `python scripts/train.py`.
