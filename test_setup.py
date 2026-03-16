import torch
import tokenizers
import datasets
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"Tokenizers: {tokenizers.__version__}")
print(f"Datasets: {datasets.__version__}")
print(f"NumPy: {np.__version__}")

x = torch.tensor([1.0, 2.0, 3.0])
print(f"✅ Tensor de prueba: {x}")
print(f"   Suma: {x.sum().item()}")