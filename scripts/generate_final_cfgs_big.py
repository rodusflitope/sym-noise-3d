import yaml
import os
import argparse
from pathlib import Path
import copy

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False)

def scale_config(
    cfg: dict,
    hidden_dim: int = 384,
    time_dim: int = 384,
    num_layers: int = 8,
    num_heads: int = 6,
    batch_size: int = 64,
    epochs: int = 1500,
    weight_decay: float = 1e-6,
    warmup_steps: int = 1000,
    ema_decay: float = 0.999,
):
    cfg = copy.deepcopy(cfg)
    
    # Scale Data
    if 'data' in cfg:
        cfg['data'].pop('max_models', None)
        cfg['data']['symmetry_precompute_num_points'] = 4096
    
    # Scale Model
    if 'model' in cfg:
        cfg['model']['hidden_dim'] = hidden_dim
        cfg['model']['time_dim'] = time_dim
        cfg['model']['num_layers'] = num_layers
        cfg['model']['num_heads'] = num_heads
            
    # Standardize Training Params
    if 'train' not in cfg:
        cfg['train'] = {}
    cfg['train']['num_points'] = 2048
    cfg['train']['amp'] = True
    cfg['train']['amp_dtype'] = 'fp16'
    cfg['train']['batch_size'] = batch_size
    cfg['train']['num_workers'] = 8
    cfg['train']['epochs'] = epochs
    cfg['train']['save_every'] = 100
    cfg['train']['log_every'] = 50
    cfg['train']['weight_decay'] = weight_decay

    # Optimizer & LR Scheduler (Cosine Warmup with lr=1e-4, wd=1e-6 as in paper)
    if 'lr_scheduler' not in cfg:
        cfg['lr_scheduler'] = {}
    cfg['lr_scheduler']['name'] = 'cosine_warmup'
    cfg['lr_scheduler']['base_lr'] = 0.0001
    cfg['lr_scheduler']['min_lr'] = 1.0e-06
    cfg['lr_scheduler']['warmup_steps'] = warmup_steps
    cfg['lr_scheduler']['weight_decay'] = weight_decay

    # EMA configuration (gamma = 0.999 as stated in paper)
    cfg['ema'] = {
        'use': True,
        'decay': ema_decay,
        'warmup_steps': 500
    }

    # Seam Artifact Mitigation (Reviewer revision)
    if 'joint_symmetry' in cfg:
        cfg['joint_symmetry']['filter_seam_before_reflection'] = True
        cfg['joint_symmetry']['seam_resample_method'] = 'fps'
    
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Generate final scaled configurations for all 5 seeds")
    parser.add_argument("--hidden_dim", type=int, default=384, help="Model hidden dimension (default: 384)")
    parser.add_argument("--time_dim", type=int, default=384, help="Time embedding dimension (default: 384)")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of DiT layers (default: 8)")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads (default: 6)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=1500, help="Epochs (default: 1500)")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay (default: 1e-6)")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps (default: 1000)")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay (default: 0.999)")
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2, 3, 4], help="5 seeds (default: 0 1 2 3 4)")
    parser.add_argument("--src_dir", type=str, default="cfgs/final_experiments", help="Source directory")
    parser.add_argument("--dst_dir", type=str, default="cfgs/final_experiments_big", help="Destination directory")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)

    experiment_files = [
        "pointtransformer_dit_baseline.yaml",
        "pointtransformer_dit_sym_loss.yaml",
        "pointtransformer_dit_sym_noise.yaml",
        "pt_true_joint_no_multiplane_x.yaml",
        "pt_true_joint_relative_dihedral.yaml",
        "pt_true_joint_relative_orthogonal.yaml",
        "pt_true_joint_relative_sparse_3p.yaml",
        "pt_true_joint_relative_sparse_6p.yaml",
    ]

    for fname in experiment_files:
        src_path = src_dir / fname
        if not src_path.exists():
            print(f"ERROR: {src_path} does not exist!")
            return

    print(f"Generating scaled configs in {dst_dir} with:")
    print(f"  hidden_dim: {args.hidden_dim}, time_dim: {args.time_dim}")
    print(f"  num_layers: {args.num_layers}, num_heads: {args.num_heads}")
    print(f"  batch_size: {args.batch_size}, epochs: {args.epochs}")
    print(f"  weight_decay: {args.weight_decay}, warmup_steps: {args.warmup_steps}")
    print(f"  ema_decay: {args.ema_decay}")
    print(f"  5 seeds: {args.seeds}")

    # Generate explicit seed subfolders for all 5 seeds (seed_0, seed_1, seed_2, seed_3, seed_4)
    for seed in args.seeds:
        seed_subfolder = f"seed_{seed}"
        for fname in experiment_files:
            src_path = src_dir / fname
            cfg = load_yaml(src_path)
            cfg_scaled = scale_config(
                cfg,
                hidden_dim=args.hidden_dim,
                time_dim=args.time_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                batch_size=args.batch_size,
                epochs=args.epochs,
                weight_decay=args.weight_decay,
                warmup_steps=args.warmup_steps,
                ema_decay=args.ema_decay,
            )
            cfg_scaled['seed'] = seed
            stem = Path(fname).stem
            cfg_scaled['exp_name'] = f"{stem}_big_s{seed}"

            dst_path = dst_dir / seed_subfolder / fname
            save_yaml(cfg_scaled, dst_path)

    # Also generate root configs corresponding to seed 0 for quick single-run execution
    for fname in experiment_files:
        src_path = src_dir / fname
        cfg = load_yaml(src_path)
        cfg_scaled = scale_config(
            cfg,
            hidden_dim=args.hidden_dim,
            time_dim=args.time_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            batch_size=args.batch_size,
            epochs=args.epochs,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            ema_decay=args.ema_decay,
        )
        cfg_scaled['seed'] = args.seeds[0]
        stem = Path(fname).stem
        cfg_scaled['exp_name'] = f"{stem}_big"
        
        dst_path = dst_dir / fname
        save_yaml(cfg_scaled, dst_path)

    print(f"\nSuccessfully generated 5 seed subfolders (seed_0 to seed_4) + root in {dst_dir}!")
    print(f"Total YAML files: {len(args.seeds) * len(experiment_files) + len(experiment_files)}")

if __name__ == '__main__':
    main()
