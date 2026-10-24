#!/usr/bin/env bash
# Precomputes all 4 symmetry plane caches required by the final experiments.
# Can be run sequentially. Each job will auto-resume if interrupted.

set -e

mkdir -p data/symmetry_cache

echo "=========================================================="
echo "1/4: Precomputing Orthogonal Canonical Symmetry (3 planes)"
echo "Target: data/symmetry_cache/symmetry_cache_table_orthogonal_3p_canonical.pt"
echo "=========================================================="
python -m scripts.precompute_symmetry_planes \
  --cfg cfgs/final_experiments_big/seed_0/pt_true_joint_relative_orthogonal.yaml \
  --canonical \
  --type orthogonal \
  --device cuda

echo "=========================================================="
echo "2/4: Precomputing Dihedral Canonical Symmetry (6 planes)"
echo "Target: data/symmetry_cache/symmetry_cache_table_dihedral_6p_canonical.pt"
echo "=========================================================="
python -m scripts.precompute_symmetry_planes \
  --cfg cfgs/final_experiments_big/seed_0/pt_true_joint_relative_dihedral.yaml \
  --canonical \
  --type dihedral \
  --k 6 \
  --device cuda

echo "=========================================================="
echo "3/4: Precomputing Sparse Per-Object Symmetry (3 planes)"
echo "Target: data/symmetry_cache/symmetry_cache_table_per_object_3p_optimized.pt"
echo "=========================================================="
python -m scripts.precompute_symmetry_planes \
  --cfg cfgs/final_experiments_big/seed_0/pt_true_joint_relative_sparse_3p.yaml \
  --type per_object \
  --n 3 \
  --device cuda

echo "=========================================================="
echo "4/4: Precomputing Sparse Per-Object Symmetry (6 planes)"
echo "Target: data/symmetry_cache/symmetry_cache_table_per_object_6p_optimized.pt"
echo "=========================================================="
python -m scripts.precompute_symmetry_planes \
  --cfg cfgs/final_experiments_big/seed_0/pt_true_joint_relative_sparse_6p.yaml \
  --type per_object \
  --n 6 \
  --device cuda

echo "=========================================================="
echo "All 4 symmetry caches precomputed successfully!"
echo "=========================================================="
