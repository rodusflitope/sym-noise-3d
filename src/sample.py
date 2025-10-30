import argparse
import pathlib
import yaml
from src.models import build_model

def parse_args():
    p = argparse.ArgumentParser(description="Baseline Diffusion - Sample")
    p.add_argument("--cfg", type=str, default="cfgs/default.yaml",
                   help="Ruta al archivo de configuración YAML.")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Ruta al checkpoint (.pt) cuando exista.")
    return p.parse_args()

def load_cfg(path: str):
    path = pathlib.Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)

    print("== Muestreo (sólo setup mínimo) ==")
    print(f"Config: {args.cfg}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Sampler: {cfg.get('sampler', {})}")

    print("OK: punto de entrada de muestreo listo.")

if __name__ == "__main__":
    main()
