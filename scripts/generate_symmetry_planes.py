import argparse
import torch
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Generate symmetry planes")
    parser.add_argument("--type", type=str, required=True, choices=["orthogonal", "dihedral", "arbitrary"])
    parser.add_argument("--k", type=int, default=3, help="Order of dihedral symmetry")
    parser.add_argument("--n", type=int, default=3, help="Number of arbitrary planes")
    parser.add_argument("--out", type=str, default="planes.pt", help="Output file to save the planes tensor")
    return parser.parse_args()

def generate_orthogonal_planes():
    return torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

def generate_dihedral_planes(k: int):
    planes = []
    for i in range(k):
        angle = i * math.pi / k
        planes.append([math.cos(angle), math.sin(angle), 0.0, 0.0])
    return torch.tensor(planes, dtype=torch.float32)

def generate_arbitrary_planes(n: int):
    # Generates random planes passing through origin
    normals = torch.randn(n, 3)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    offsets = torch.zeros(n, 1)
    return torch.cat([normals, offsets], dim=-1)

def main():
    args = parse_args()
    
    if args.type == "orthogonal":
        planes = generate_orthogonal_planes()
    elif args.type == "dihedral":
        planes = generate_dihedral_planes(args.k)
    elif args.type == "arbitrary":
        planes = generate_arbitrary_planes(args.n)
    
    torch.save(planes, args.out)
    print(f"[{args.type}] Generated {planes.shape[0]} planes and saved to {args.out}")
    print(planes)

if __name__ == "__main__":
    main()
