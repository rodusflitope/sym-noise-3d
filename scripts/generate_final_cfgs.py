import yaml
import os
import argparse
from pathlib import Path

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def scale_config(cfg):
    # Scale Data
    if 'data' in cfg:
        cfg['data'].pop('max_models', None)
        cfg['data']['symmetry_precompute_num_points'] = 4096
    
    # Scale Model
    if 'model' in cfg:
        cfg['model']['hidden_dim'] = 128
        cfg['model']['time_dim'] = 128
        cfg['model']['num_layers'] = 3
        cfg['model']['num_heads'] = 4
            
    # Standardize training params
    if 'train' in cfg:
        cfg['train']['num_points'] = 2048

        cfg['train']['amp'] = True
        cfg['train']['amp_dtype'] = 'fp16'
        
        cfg['train']['batch_size'] = 64
        cfg['train']['num_workers'] = 8
            
        cfg['train']['epochs'] = 1000
        cfg['train']['save_every'] = 100
    
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Generate final configurations for different seeds")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3, 4, 5], help="List of seeds to generate configs for. Default: 1 2 3 4 5")
    args = parser.parse_args()

    mappings = {
        "cfgs/pointtransformer_dit.yaml": "cfgs/final_experiments/pointtransformer_dit_baseline.yaml",
        "cfgs/pointtransformer_dit_sym_noise.yaml": "cfgs/final_experiments/pointtransformer_dit_sym_noise.yaml",
        "cfgs/pointtransformer_dit_sym_loss.yaml": "cfgs/final_experiments/pointtransformer_dit_sym_loss.yaml",
        "cfgs/pointtransformer_true_joint_multiplane_relative_dit.yaml": "cfgs/final_experiments/pt_true_joint_relative_orthogonal.yaml",
        "cfgs/pointtransformer_true_joint_multiplane_dit_dihedral.yaml": "cfgs/final_experiments/pt_true_joint_relative_dihedral.yaml",
        "cfgs/pointtransformer_true_joint_multiplane_dit_sparse_3p.yaml": "cfgs/final_experiments/pt_true_joint_relative_sparse_3p.yaml",
        "cfgs/pt_true_joint_no_multiplane_x.yaml": "cfgs/final_experiments/pt_true_joint_no_multiplane_x.yaml",
    }

    for seed in args.seeds:
        print(f"\n--- Generating configs for seed {seed} ---")
        for src, dst in mappings.items():
            if os.path.exists(src):
                cfg = load_yaml(src)
                cfg = scale_config(cfg)
                
                # Apply seed
                cfg['seed'] = seed
                
                path_obj = Path(dst)
                dst_seeded = str(path_obj.parent / f"seed_{seed}" / path_obj.name)
                cfg['exp_name'] = f"{path_obj.stem}_s{seed}"
                
                save_yaml(cfg, dst_seeded)
                print(f"Generated {dst_seeded}")
            else:
                print(f"WARNING: {src} not found")

        # Generate 6p sparse
        src_6p = "cfgs/pointtransformer_true_joint_multiplane_dit_sparse.yaml"
        if os.path.exists(src_6p):
            cfg_6p = load_yaml(src_6p)
            cfg_6p = scale_config(cfg_6p)
            
            # Apply seed
            cfg_6p['seed'] = seed
            
            cfg_6p['exp_name'] = f"pt_true_joint_relative_sparse_6p_s{seed}"
            if 'data' in cfg_6p:
                cfg_6p['data']['num_symmetry_planes'] = 6
                cfg_6p['data']['symmetry_plane_cache_path'] = "data/symmetry_cache/symmetry_cache_table_per_object_6p_optimized.pt"
            if 'model' in cfg_6p:
                cfg_6p['model']['num_planes'] = 6
                
            dst_6p = f"cfgs/final_experiments/seed_{seed}/pt_true_joint_relative_sparse_6p.yaml"
            save_yaml(cfg_6p, dst_6p)
            print(f"Generated {dst_6p}")
        else:
            print(f"WARNING: {src_6p} not found")

if __name__ == '__main__':
    main()
