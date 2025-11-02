import torch

print("CUDA disponible:", torch.cuda.is_available())
print("Versión CUDA (build):", torch.version.cuda)
print("Versión cuDNN:", torch.backends.cudnn.version())
print("cuDNN activado:", torch.backends.cudnn.is_available())
