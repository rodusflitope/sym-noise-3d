import os

# ==============================================================================
# Global Configuration for External Servers
# ==============================================================================
# Puedes editar este archivo directamente en tu servidor externo para cambiar
# las rutas globales sin tener que modificar cada archivo YAML uno por uno.
#
# Si usas variables de entorno (ej. export DATA_ROOT_DIR=/ruta/a/data),
# tomarán prioridad automáticamente.

# Ruta base donde se encuentra el dataset (ej. ShapeNetCore)
DATA_ROOT_DIR = os.getenv("DATA_ROOT_DIR", "data/ShapeNetCore")

# Ruta base donde se guardan los cachés de simetría (.pt)
CACHE_ROOT_DIR = os.getenv("CACHE_ROOT_DIR", "data/symmetry_cache")

# Ruta base donde se guardarán los resultados, logs y checkpoints
OUT_ROOT_DIR = os.getenv("OUT_ROOT_DIR", "runs")
