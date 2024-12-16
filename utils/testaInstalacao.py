# Testa se a GPU está ok usando uma chamada para o sistema operacional e rodando no nvidia-smi
import os
print('Testando se os drives de GPU estão rodando mo Linux:')
os.system('nvidia-smi')

# Testa pytorch
import torch

# Testa se a GPU está disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Esta executando o pytorch em: ',device)

# Testa se o pytorch está instalado corretamente
print('Realizando uma operação matricial com o python:')
x = torch.empty(5, 3)
print(x)

# Testa se o MMCV está instalado corretamente
print('Testando o MMCV:')
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print('CUDA Version pelo MMCV: ',get_compiling_cuda_version())
print('Compiler Version pelo MMCV: ',get_compiler_version())


