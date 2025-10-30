import argparse
import pathlib
import yaml
from src.models import build_model

def parse_args():
    p = argparse.ArgumentParser(description="Baseline Diffusion - Train")
    p.add_argument("--cfg", type=str, default="cfgs/default.yaml",
                   help="Ruta al archivo de configuración YAML.")
    return p.parse_args()

def load_cfg(path: str):
    path = pathlib.Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)

    print("== Entrenamiento (sólo setup mínimo) ==")
    print(f"Config cargada desde: {args.cfg}")
    print(f"Nombre experimento: {cfg.get('exp_name', 'N/A')}")
    print(f"Semilla: {cfg.get('seed', 'N/A')}")
    print(f"Device: {cfg.get('device', 'auto')}")
    print(f"Modelo (placeholder): {cfg.get('model', {})}")

    # NOTA: en este paso NO instanciamos el modelo ni corremos nada todavia.
    print("OK: punto de entrada de entrenamiento listo.")

if __name__ == "__main__":
    main()
