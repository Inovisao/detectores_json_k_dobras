# Use os dois comandos abaixo para criar e ativar o ambiente 
# CONDA ANTES DE RODAR OS OUTROS. RODE ESTES COMANDOS
# FORA DO SCRIPT
#
#
# conda create --name detectores python=3.7 -y
# conda activate detectores


# Se precisar instalar versão antigo do CUDA, pode
# tentar seguir isso: https://varhowto.com/install-pytorch-cuda-10-0/
# Se precisar instalar o cuda de outra forma, tente isso aqui:
# https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=deblocal


conda install cudatoolkit=11.3 pytorch torchvision -c pytorch

# Recomendo entrar no python e rodar os comandos abaixo para
# ver se o pytorch está identificando a GPU. Se não tiver,
# tem que resolver antes de continuar
python
>> import torch
>> print('Torch: ',torch.__version__, torch.cuda.is_available())

# Para saber a versão do pytorch e cuda instalada na sua máquina
conda list | grep cuda

# Se der erro de incompatilibilidade tenta usar o comando abaixo
# para ver a versão do cuda no sistema
# nvcc --version
# E tenta atualizar baixando o toolkit do site da nvidia
pip install mmcv-full==1.3.5

#git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  
pip install dicttoxml albumentations terminaltables imagecorruptions 
cd ..

pip install funcy sklearn

# Para remover o ambiente e começar tudo de novo
# conda deactivate
# conda remove --name detectores --all
