import os

# ==============================================================================
# Global Configuration for External Servers
# ==============================================================================
# You can edit this file directly on your external server to change
# the global paths without having to modify each YAML file one by one.
#
# If you use environment variables (e.g. export DATA_ROOT_DIR=/path/to/data),
# they will automatically take priority.

# Base path where the dataset is located (e.g. ShapeNetCore)
DATA_ROOT_DIR = os.getenv("DATA_ROOT_DIR", "data/ShapeNetCore")

# Base path where symmetry caches (.pt) are saved
CACHE_ROOT_DIR = os.getenv("CACHE_ROOT_DIR", "data/symmetry_cache")

# Base path where results, logs, and checkpoints will be saved
OUT_ROOT_DIR = os.getenv("OUT_ROOT_DIR", "runs")
