import yaml
import os
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
        model_name = cfg['model'].get('name', '').lower()
        if 'pvcnn' in model_name:
            cfg['model']['hidden_dim'] = 128
            cfg['model']['time_dim'] = 128
            cfg['model']['resolution'] = 32
            cfg['model']['num_blocks'] = 3
        else:
            cfg['model']['hidden_dim'] = 256
            cfg['model']['time_dim'] = 256
            cfg['model']['num_layers'] = 4
            cfg['model']['num_heads'] = 8
            
    # Standardize training params
    if 'train' in cfg:
        cfg['train']['num_points'] = 2048

        cfg['train']['amp'] = True
        cfg['train']['amp_dtype'] = 'fp16'
        
        model_name = cfg.get('model', {}).get('name', '').lower()
        if 'pvcnn' in model_name:
            cfg['train']['batch_size'] = 64
        else:
            cfg['train']['batch_size'] = 32
            
        cfg['train']['epochs'] = 1000
        cfg['train']['save_every'] = 100
    
    return cfg

mappings = {
    "cfgs/pointtransformer_dit.yaml": "cfgs/final_experiments/pointtransformer_dit_baseline.yaml",
    "cfgs/PVCNN.yaml": "cfgs/final_experiments/pvcnn_baseline.yaml",
    "cfgs/pointtransformer_symmetric_reflected.yaml": "cfgs/final_experiments/pointtransformer_dit_sym_noise.yaml",
    "cfgs/PVCNN_symmetric_reflected_legacy.yaml": "cfgs/final_experiments/pvcnn_sym_noise.yaml",
    "cfgs/pointtransformer_symmetric_loss.yaml": "cfgs/final_experiments/pointtransformer_dit_sym_loss.yaml",
    "cfgs/PVCNN_symmetric_loss_legacy.yaml": "cfgs/final_experiments/pvcnn_sym_loss.yaml",
    "cfgs/pointtransformer_true_joint_multiplane_relative_dit.yaml": "cfgs/final_experiments/pt_true_joint_relative_orthogonal.yaml",
    "cfgs/pointtransformer_true_joint_multiplane_dit_dihedral.yaml": "cfgs/final_experiments/pt_true_joint_relative_dihedral.yaml",
    "cfgs/pointtransformer_true_joint_multiplane_dit_sparse_3p.yaml": "cfgs/final_experiments/pt_true_joint_relative_sparse_3p.yaml",
}

for src, dst in mappings.items():
    if os.path.exists(src):
        cfg = load_yaml(src)
        cfg = scale_config(cfg)
        cfg['exp_name'] = Path(dst).stem
        save_yaml(cfg, dst)
        print(f"Generated {dst}")
    else:
        print(f"WARNING: {src} not found")

# Generate 6p sparse
src_6p = "cfgs/pointtransformer_true_joint_multiplane_dit_sparse.yaml"
if os.path.exists(src_6p):
    cfg_6p = load_yaml(src_6p)
    cfg_6p = scale_config(cfg_6p)
    cfg_6p['exp_name'] = "pt_true_joint_relative_sparse_6p"
    if 'data' in cfg_6p:
        cfg_6p['data']['num_symmetry_planes'] = 6
        cfg_6p['data']['symmetry_plane_cache_path'] = "data/symmetry_cache/symmetry_cache_table_per_object_6p_optimized.pt"
    if 'model' in cfg_6p:
        cfg_6p['model']['num_planes'] = 6
    save_yaml(cfg_6p, "cfgs/final_experiments/pt_true_joint_relative_sparse_6p.yaml")
    print("Generated cfgs/final_experiments/pt_true_joint_relative_sparse_6p.yaml")
else:
    print(f"WARNING: {src_6p} not found")
