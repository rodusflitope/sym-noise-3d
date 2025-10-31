import pathlib, torch

def save_ckpt(model, out_dir: str, exp_name: str, filename: str = "last.pt"):
    d = pathlib.Path(out_dir) / exp_name
    d.mkdir(parents=True, exist_ok=True)
    path = d / filename
    torch.save({"model": model.state_dict()}, path)
    return str(path)

def load_ckpt(model, ckpt_path: str, map_location=None):
    sd = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(sd["model"], strict=True)
    return model
