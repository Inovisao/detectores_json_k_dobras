# Testa se a GPU está ok usando uma chamada para o sistema operacional e rodando no nvidia-smi
import os
os.system('nvidia-smi')

# Testa pytorch
import torch

# Testa se a GPU está disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Testa se o pytorch está instalado corretamente
x = torch.empty(5, 3)
print(x)

# Testa se o MMCV está instalado corretamente
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())


