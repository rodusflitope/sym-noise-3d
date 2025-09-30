from huggingface_hub import list_repo_files, hf_hub_download
import os, zipfile, argparse

HF_REPO = "ShapeNet/ShapeNetCore"
DEST = "data/ShapeNetCore"

def download_synset(syn, dest):
    """Descarga y extrae un synset específico."""
    print(f"➡️ Descargando {syn}...")
    try:
        zip_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=syn,
            repo_type="dataset",
            local_dir=dest,
        )
        out_dir = os.path.join(dest, syn.replace(".zip", ""))
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        print(f"✅ {syn} listo en {out_dir}")
    except Exception as e:
        print(f"❌ Error al descargar {syn}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Descargar ShapeNetCore desde Hugging Face"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="all",
        help="Lista de categorías (separadas por coma) o 'all' para todo el dataset"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=DEST,
        help="Carpeta destino (default: data/)"
    )
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    # Obtener lista de todos los zips disponibles en el repo
    files = list_repo_files(HF_REPO, repo_type="dataset")
    all_synsets = [f for f in files if f.endswith(".zip")]

    # Filtrar según parámetro
    if args.categories == "all":
        synsets = all_synsets
    else:
        requested = [c.strip() for c in args.categories.split(",")]
        synsets = [c if c.endswith(".zip") else f"{c}.zip" for c in requested]

    for syn in synsets:
        download_synset(syn, args.dest)

if __name__ == "__main__":
    main()
