# sym-noise-3d

Diffusion models for 3D point clouds with symmetry-aware noise schedules.

This repo supports two main training workflows:
- **Point-space diffusion**: train a diffusion model directly on 3D point clouds.
- **Latent-space diffusion**: first train a point-cloud autoencoder, then train diffusion in the latent space.

Below are concise instructions to run each stage end-to-end.

## Naming note (conditional vs joint)

The current conditional symmetry method appears as `joint` in some internal code names by naming mistake.

Use the following meaning in experiments and reporting:
- `joint` in legacy/internal symbols means the current conditional pipeline.
- `joint_true` means a true joint formulation.

## 1. Setup (Unified Installation)

It is highly recommended to use a single, unified Conda environment for the entire repository, including the `sym-lion` submodule. This avoids dependency conflicts and lets you run both standard models and LION baselines from the same terminal.

1. Create and activate the Conda environment (this includes PyTorch 2.5, CUDA 12.1, and all pip requirements):

```pwsh
conda env create -f env.yaml
conda activate sym-noise-env
```

2. Build the `sym-lion` submodule components (requires the environment to be active):

```pwsh
python sym-lion/build_pkg.py
```

3. Download ShapeNet data (if you haven’t already):

```pwsh
python scripts/download_data_from_huggingface.py --categories 02691156,03001627
```

By default configs assume ShapeNetCore is under `data/ShapeNetCore`.

## 2. Autoencoder training (latent space setup)

The autoencoder compresses point clouds into latent vectors that the diffusion model will later operate on.

- **Config**: `cfgs/latent_diffusion.yaml` (section `autoencoder` controls `latent_dim`, `hidden_dim`, `epochs`, etc.).
- **Entry point**: `src/train_autoencoder.py`.

Run:

```pwsh
& .venv/Scripts/Activate.ps1
python src/train_autoencoder.py --cfg cfgs/latent_diffusion.yaml
```

Outputs:
- Checkpoints under `runs/ae_<exp_name>_<timestamp>/` (e.g. `best.pt`, `last.pt`, `epoch_XXX.pt`).
- `splits.json` with the train/val/test indices used.

You will need the path to `best.pt` (or another AE checkpoint) for latent diffusion training.

## 3. Diffusion training in latent space

Once the AE is trained, enable latent diffusion and provide its checkpoint.

- **Config**: `cfgs/latent_diffusion.yaml` with `use_latent_diffusion: true`.
- **Entry point**: `src/train.py`.

Recommended command (explicit AE checkpoint):

```pwsh
& .venv/Scripts/Activate.ps1
python src/train.py --cfg cfgs/latent_diffusion.yaml --ae_ckpt runs/ae_latent-diffusion_YYYYMMDD_HHMMSS/best.pt
```

Alternatively, set the environment variable `AE_CHECKPOINT` and omit `--ae_ckpt`:

```pwsh
$env:AE_CHECKPOINT = "C:/Dev/sym-noise-3d/runs/ae_latent-diffusion_YYYYMMDD_HHMMSS/best.pt"
python src/train.py --cfg cfgs/latent_diffusion.yaml
```

Outputs:
- Diffusion checkpoints under `runs/<exp_name>_<timestamp>/`.
- Training history JSON (losses per epoch, best epoch, etc.).

## 4. Diffusion training in point space (baseline)

To train directly on point clouds (no autoencoder):

- **Configs**: `cfgs/default.yaml`, `cfgs/pointnet.yaml`, `cfgs/pointtransformer.yaml`.
- Ensure `use_latent_diffusion: false` in the chosen config.

Run, for example, a PointNet baseline:

```pwsh
& .venv/Scripts/Activate.ps1
python src/train.py --cfg cfgs/pointnet.yaml
```

## 5. Sampling from trained models

Use `src/sample.py` to generate point clouds from a trained diffusion model.

Example (point-space or latent-space, depending on config):

```pwsh
& .venv/Scripts/Activate.ps1
python src/sample.py --cfg cfgs/pointnet.yaml --ckpt runs/pointnet-baseline_YYYYMMDD_HHMMSS/best.pt
```

Key options (set in the YAML under `sampler`):
- `name`: `ddpm` or `ddim`.
- `eta`: 1.0 for stochastic DDPM, 0.0 for deterministic DDIM.
- `num_samples`: number of generated shapes.
- `save_dir`: output directory (e.g. `samples`).
- `save_format`: `npy` or `ply`.

## 6. Evaluation

Use `src/eval.py` to compute metrics like Chamfer Distance (and optionally EMD) on the test split.

Point-space example:

```pwsh
& .venv/Scripts/Activate.ps1
python src/eval.py --cfg cfgs/pointnet.yaml --ckpt runs/pointnet-baseline_YYYYMMDD_HHMMSS/best.pt --metric cd
```

Latent diffusion example (requires both AE and diffusion checkpoints):

```pwsh
& .venv/Scripts/Activate.ps1
python src/eval.py --cfg cfgs/latent_diffusion.yaml --ckpt runs/latent-diffusion_YYYYMMDD_HHMMSS/best.pt --metric cd --ae_ckpt runs/ae_latent-diffusion_YYYYMMDD_HHMMSS/best.pt
```

## 7. Quick pipeline tests

There are small tests to sanity-check each stage before long runs:

```pwsh
& .venv/Scripts/Activate.ps1
python tests/test_01_data_loading.py --cfg cfgs/pointnet.yaml
python tests/test_02_noise_generation.py --cfg cfgs/pointnet.yaml
python tests/test_03_model_forward.py --cfg cfgs/pointnet.yaml
python tests/test_04_denoising.py --cfg cfgs/pointnet.yaml

# Autoencoder reconstruction test (needs trained AE)
python tests/test_05_autoencoder_reconstruction.py --cfg cfgs/latent_diffusion.yaml --ckpt runs/ae_latent-diffusion_YYYYMMDD_HHMMSS/best.pt
```

These tests write visualizations to `tests/outputs/` for manual inspection.

For more detailed architecture and config documentation, see `.github/copilot-instructions.md`.
