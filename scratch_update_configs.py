import os
import glob
import yaml
from pathlib import Path

def get_sym_type_from_path(path):
    if not path:
        return "orthogonal"
    path = str(path).lower()
    if "dihedral" in path:
        return "dihedral"
    if "arbitrary" in path or "per_object" in path:
        return "per_object"
    return "orthogonal"

def update_yaml_files():
    cfg_dir = Path("cfgs")
    yaml_files = glob.glob(str(cfg_dir / "*.yaml"))
    
    updated_count = 0
    for yf in yaml_files:
        with open(yf, "r") as f:
            content = f.read()
            
        try:
            data = yaml.safe_load(content)
        except:
            continue
            
        if not data or not isinstance(data, dict):
            continue
            
        data_cfg = data.get("data", {})
        if not data_cfg or "symmetry_plane_cache_path" not in data_cfg:
            continue
            
        old_path = data_cfg["symmetry_plane_cache_path"]
        
        category = "airplane"
        categories = data_cfg.get("categories", None)
        if categories and len(categories) > 0:
            category = categories[0]
            
        sym_type = get_sym_type_from_path(old_path)
        num_planes = data_cfg.get("num_symmetry_planes", 1)
        use_canonical = bool(data_cfg.get("canonical_symmetry_planes", False))
        canonical_str = "canonical" if use_canonical else "optimized"
        
        new_path = f"data/symmetry_cache/symmetry_cache_{category}_{sym_type}_{num_planes}p_{canonical_str}.pt"
        
        if old_path != new_path:
            # We must be careful not to just dump yaml because it loses comments/formatting.
            # A simple string replace for the specific line might be better.
            new_content = []
            for line in content.split("\n"):
                if "symmetry_plane_cache_path:" in line:
                    # preserve leading spaces
                    leading_spaces = len(line) - len(line.lstrip())
                    new_line = " " * leading_spaces + f'symmetry_plane_cache_path: "{new_path}"'
                    new_content.append(new_line)
                else:
                    new_content.append(line)
                    
            with open(yf, "w") as f:
                f.write("\n".join(new_content))
            print(f"Updated {yf}: {old_path} -> {new_path}")
            updated_count += 1
            
    print(f"Total updated: {updated_count}")

if __name__ == '__main__':
    update_yaml_files()
