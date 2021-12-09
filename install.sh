# Use os dois comandos abaixo para criar e ativar o ambiente 
# CONDA ANTES DE RODAR OS OUTROS. RODE ESTES COMANDOS
# FORA DO SCRIPT
#
#
# conda create --name detectores python=3.7.10 -y
# conda activate detectores


# Downgrade do cudatoolkit para conseguir compilar o mmcv
# (se tiver uma GPU mais nova, pode não ser necessário fazer
# o downgrade)
conda install cudatoolkit=11.2

# Instala o pytorch (um concorrente do tensorflow)
conda install pytorch torchvision -c pytorch -y  

# Para saber a versão do pytorch e cuda instalada na sua máquina
conda list | grep cuda

# Tem que trocar torch1.8.1 e cu101 pela versão correspondente ao
# pytorch e cuda instalados na sua máquina (ver resultado do comando
# anterior)
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
