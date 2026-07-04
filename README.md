# Symmetry-Aware Diffusion Models for 3D Point Clouds

> **Thesis** — *Incorporación de Simetría Geométrica en Modelos de Difusión para Nubes de Puntos 3D*
>
> Department of Electrical Engineering, Universidad de Chile, 2026.

This repository implements and evaluates methods for integrating geometric symmetry as an inductive bias in diffusion models for 3D point cloud generation, covering both **physical-space** and **latent-space** approaches.

---

## Overview

Standard diffusion models treat each point coordinate independently, forcing the network to learn global structural properties — such as reflection symmetry — purely from data. This work introduces explicit symmetry priors into the diffusion process through two complementary strategies:

### Physical Space: Joint Diffusion (DiT)

A **co-diffusion** scheme that jointly estimates the point cloud's fundamental domain and its symmetry planes. The model processes only half (or a fraction) of the shape and reconstructs the full object via reflection operators, reducing Transformer self-attention cost by up to **4×** for a single symmetry plane.

| Variant | Planes | Description |
|---------|--------|-------------|
| **Joint (Static X)** | 1 fixed | Single canonical plane `[1,0,0]` — control experiment |
| **Orthogonal** | 3 canonical | Planes aligned to principal axes (`X`, `Y`, `Z`) |
| **Dihedral** | 6 canonical | Three primary + three diagonal planes |
| **Sparse** | 3 or 6 optimized | Per-object planes discovered via Chamfer minimization |

Additionally, three **baseline** strategies are compared:
- **Vanilla DiT** — standard diffusion without symmetry
- **Symmetry Noise** — symmetric noise injection (masked, reflected, averaged)
- **Loss Symmetry** — Chamfer-based symmetry penalty on the predicted clean shape

### Latent Space: LION + Soft Symmetry

Soft symmetry constraints injected into the VAE of the [LION](https://github.com/nv-tlabs/LION) framework:
- **Reconstruction symmetry loss** — penalizes asymmetry in decoded point clouds
- **Latent symmetry loss** — enforces equivariance in the latent code

---

## Repository Structure

```
sym-noise-3d/
├── cfgs/                        # YAML experiment configurations
│   ├── final_experiments/       #   ↳ Final thesis experiments (8 models)
│   └── ...                      #   ↳ Development / ablation configs
├── src/
│   ├── train.py                 # Main training entry point
│   ├── sample.py                # Point cloud generation
│   ├── eval.py                  # Metric computation (CD, EMD, MMD, COV, 1-NNA)
│   ├── post_train.py            # Post-training fine-tuning
│   ├── train_autoencoder.py     # Autoencoder training (latent pipeline)
│   ├── models/                  # Network architectures
│   │   ├── pointtransformer_dit.py                         # Vanilla DiT
│   │   ├── pointtransformer_true_joint_dit.py              # Single-plane Joint DiT
│   │   ├── pointtransformer_true_joint_multiplane_dit.py   # Multi-plane Joint DiT
│   │   ├── pointtransformer_true_joint_multiplane_relative_dit.py  # Relative DiT
│   │   ├── pointtransformer_true_joint_multiplane_sparse_dit.py    # Sparse DiT
│   │   └── ...
│   ├── losses/                  # Loss functions (MSE, joint, consistency, etc.)
│   ├── metrics/                 # CD, EMD, RSD evaluation metrics
│   ├── samplers/                # DDPM / DDIM samplers
│   ├── schedulers/              # Noise schedules (linear, cosine)
│   └── utils/                   # Symmetry plane utilities, visualization
├── scripts/
│   ├── precompute_symmetry_planes.py   # Offline plane cache generation
│   ├── download_data_from_huggingface.py
│   └── ...
├── sym-lion/                    # LION framework submodule (latent-space experiments)
├── tests/                       # Pipeline sanity checks
├── env.yaml                     # Conda environment specification
└── requirements.txt             # Pip dependencies
```

---

## Installation

### Prerequisites

- Python 3.10
- CUDA 12.1
- Conda (recommended)

### Setup

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/<user>/sym-noise-3d.git
cd sym-noise-3d

# 2. Create Conda environment (PyTorch 2.5 + CUDA 12.1)
conda env create -f env.yaml
conda activate sym-noise-3d

# 3. Build LION submodule components (for latent-space experiments)
python sym-lion/build_pkg.py

# 4. Download ShapeNet data
python scripts/download_data_from_huggingface.py --categories 02691156,03001627
```

Data is expected under `data/ShapeNetCore/` by default.

---

## Usage

### 1. Precompute Symmetry Plane Caches

Joint diffusion models require a precomputed cache of symmetry planes per object. Run once before training:

```bash
# Orthogonal cache (3 canonical planes)
python scripts/precompute_symmetry_planes.py \
    --cfg cfgs/final_experiments/pt_true_joint_relative_orthogonal.yaml \
    --type orthogonal --canonical --high_precision

# Dihedral cache (6 canonical planes)
python scripts/precompute_symmetry_planes.py \
    --cfg cfgs/final_experiments/pt_true_joint_relative_dihedral.yaml \
    --type dihedral --k 6 --canonical --high_precision

# Sparse cache (3 per-object optimized planes)
python scripts/precompute_symmetry_planes.py \
    --cfg cfgs/final_experiments/pt_true_joint_relative_sparse_3p.yaml \
    --type per_object --n 3 --high_precision

# Sparse cache (6 per-object optimized planes)
python scripts/precompute_symmetry_planes.py \
    --cfg cfgs/final_experiments/pt_true_joint_relative_sparse_6p.yaml \
    --type per_object --n 6 --high_precision
```

Optionally, inspect the cache to tune the symmetry score threshold:

```bash
python scripts/test_symmetry_classes.py \
    data/symmetry_cache/symmetry_cache_table_dihedral_6p_canonical.pt \
    --threshold 0.01 --num-planes 6
```

### 2. Training

**Run all final experiments sequentially** (each runs in an isolated subprocess to prevent OOM):

```bash
python src/train.py --cfg cfgs/final_experiments
```

**Or train individual models:**

```bash
# Vanilla DiT baseline
python -m src.train --cfg cfgs/final_experiments/pointtransformer_dit_baseline.yaml

# DiT + Symmetry Noise
python -m src.train --cfg cfgs/final_experiments/pointtransformer_dit_sym_noise.yaml

# DiT + Loss Symmetry
python -m src.train --cfg cfgs/final_experiments/pointtransformer_dit_sym_loss.yaml

# Joint Diffusion — Single X Plane (control)
python -m src.train --cfg cfgs/final_experiments/pt_true_joint_no_multiplane_x.yaml

# Joint Diffusion — 3 Orthogonal Planes
python -m src.train --cfg cfgs/final_experiments/pt_true_joint_relative_orthogonal.yaml

# Joint Diffusion — 6 Dihedral Planes
python -m src.train --cfg cfgs/final_experiments/pt_true_joint_relative_dihedral.yaml

# Joint Diffusion — 3 Sparse Planes
python -m src.train --cfg cfgs/final_experiments/pt_true_joint_relative_sparse_3p.yaml

# Joint Diffusion — 6 Sparse Planes
python -m src.train --cfg cfgs/final_experiments/pt_true_joint_relative_sparse_6p.yaml
```

Checkpoints are saved under `runs/<date>/<exp_name>_<run_id>/`.

### 3. Sampling

```bash
python src/sample.py \
    --cfg cfgs/final_experiments/pt_true_joint_relative_orthogonal.yaml \
    --ckpt runs/<date>/<exp_name>_<run_id>/best.pt
```

Samples are saved under `samples/<date>/<run_name>/` in `.npy` or `.ply` format.

### 4. Evaluation

Compute generative metrics against the held-out test set:

```bash
python -m src.eval --ckpt runs/<date>/<exp_name>_<run_id> --eval_all --compute_emd
```

| Flag | Description |
|------|-------------|
| `--eval_all` | Use the full test split (ignores `--num_samples`) |
| `--compute_emd` | Enable EMD-based metrics: MMD, COV, 1-NNA |

Metrics computed:
- **Chamfer Distance (CD)** and **Earth Mover's Distance (EMD)** — fidelity
- **Minimum Matching Distance (MMD)** — sample quality
- **Coverage (COV)** — diversity
- **1-Nearest Neighbor Accuracy (1-NNA)** — distributional faithfulness
- **Reflective Symmetry Distance (RSD)** — geometric symmetry quality

---

## LION Latent-Space Experiments

For the VAE + latent diffusion pipeline, see the `sym-lion/` submodule. The workflow involves:

1. Train the LION VAE autoencoder with soft symmetry losses:
   ```bash
   python src/train_autoencoder.py --cfg cfgs/lion_autoencoder_soft_symmetry.yaml
   ```

2. Train latent diffusion conditioned on the frozen encoder:
   ```bash
   python src/train.py --cfg cfgs/lion_latent_diffusion_soft_symmetry.yaml \
       --ae_ckpt runs/<date>/ae_<exp>_<id>/best.pt
   ```

---

## Pipeline Tests

Sanity-check each stage before launching long training runs:

```bash
python tests/test_01_data_loading.py --cfg cfgs/final_experiments/pointtransformer_dit_baseline.yaml
python tests/test_02_noise_generation.py --cfg cfgs/final_experiments/pointtransformer_dit_baseline.yaml
python tests/test_03_model_forward.py --cfg cfgs/final_experiments/pointtransformer_dit_baseline.yaml
python tests/test_04_denoising.py --cfg cfgs/final_experiments/pointtransformer_dit_baseline.yaml
```

---

## Configuration

All experiments are controlled via YAML files in `cfgs/`. Key configuration groups:

| Section | Key Parameters |
|---------|---------------|
| `model` | `name`, `hidden_dim`, `num_heads`, `num_layers`, `num_planes` |
| `diffusion` | `T` (1000), `beta_start` (1e-4), `beta_end` (0.02), `schedule` (linear) |
| `loss` | `lambda_diff` (1.0), `lambda_plane` (1.0), `weighting` |
| `train` | `batch_size`, `epochs`, `num_points` (2048), `amp` (true) |
| `data` | `categories`, `soft_cut`, `soft_cut_margin`, `symmetry_plane_cache_path` |
| `joint_symmetry` | `geometry_mode`, `plane_mode`, `inactive_plane_norm_threshold` |

See [`cfgs/example_config.yaml`](cfgs/example_config.yaml) for a fully documented reference.

---

## Citation

```bibtex
@thesis{sym-noise-3d-2026,
    title   = {Incorporación de Simetría Geométrica en Modelos de Difusión
               para Nubes de Puntos 3D},
    author  = {Rafael Tapia},
    school  = {Universidad de Chile, Departamento de Ingeniería Eléctrica},
    year    = {2026},
    type    = {Memoria de Título}
}
```

## License

This code was developed as part of a thesis project. Please contact the author for usage permissions.
