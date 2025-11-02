import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import ShapeNetDataset
from torch.utils.data import DataLoader


def main():
    print("=" * 60)
    print("Prueba del Dataset de ShapeNet")
    print("=" * 60)
    
    print("\n[1] Cargando dataset...")
    dataset = ShapeNetDataset(root_dir="data/ShapeNetCore", num_points=2048, normalize=True)
    
    print(f"✓ Dataset cargado exitosamente")
    print(f"  - Total de modelos: {len(dataset)}")
    
    print("\n[2] Probando un sample...")
    sample = dataset[0]
    print(f"✓ Sample shape: {sample.shape}")
    print(f"  - Mean: {sample.mean():.6f}")
    print(f"  - Std: {sample.std():.6f}")
    print(f"  - Min: {sample.min():.6f}")
    print(f"  - Max: {sample.max():.6f}")
    
    print("\n[3] Probando con DataLoader...")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    print(f"✓ Batch shape: {batch.shape}")
    
    print("\n" + "=" * 60)
    print("✓ Todas las pruebas pasaron exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    main()
