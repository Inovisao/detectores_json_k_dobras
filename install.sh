# Sugestões para instalação dos pacotes
# NÃO RODE ESTE CÓDIGO ... TEM QUE COPIAR E COLAR OS COMANDOS NO TERMINAL


# SUGESTÃO 1: 
# Testada em 19/12/2024 por Hemerson Pistori (pistori@ucdb.br)
# Ubuntu 22.04 GPU: RTX A4000  NVIDIA-SMI 535.183.01   CUDA Version: 12.2  
# -----------------------------------------------------------------------------------------------
conda create --name detectores python=3.8 -y
conda activate detectores
conda install -y pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge
python -c 'import torch;print(torch.__version__);print(torch.version.cuda)'
pip install --no-input dicttoxml albumentations terminaltables imagecorruptions funcy scikit-learn pycocotools wget
pip install --no-input -U openmim
mim install "mmengine==0.8.4"
mim install "mmcv==1.3.17"
mim install "mmcv-full==1.7.1"
mim install "mmdet==2.28.2"
pip install yapf==0.40.1


#-------------------------------------------------------------------------------------------------

# SUGESTÃO 2: 
# Testada em 22/03/2022 por Junior Souza (junior.souza@ifms.edu.br)
# Ubuntu 20.04 GPU: Titan Xp CUDA 11.4 Driver: Nvidia 470.182.03
# -----------------------------------------------------------------------------------------------
conda create --name detectores python=3.6 -y
conda activate detectores
pip install -U torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.3
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  
cd ..
pip install dicttoxml albumentations terminaltables imagecorruptions funcy sklearn wget
#-------------------------------------------------------------------------------------------------

# DICAS QUE PODEM SER INTERESSANTES:

# Para saber qual é a GPU da máquina:
sudo lshw -C display

# Para saber qual o CUDA e Driver instalado
nvidia-smi

# Para ver a versão do pytorch e do cuda (e se estão instalados mesmo)
python -c 'import torch;print(torch.__version__);print(torch.version.cuda)'

# Para saber a versão do pytorch e cuda instalada na sua máquina
conda list | grep cuda

# Para remover o ambiente e começar tudo de novo
conda deactivate
conda remove --name detectores --all

# Sites consultados para ajudar na instalação:
https://mmdetection.readthedocs.io/en/latest/get_started.html
https://mmcv.readthedocs.io/en/latest/get_started/installation.html



