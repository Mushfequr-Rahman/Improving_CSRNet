conda create -n CSRNet python=3.6
conda activate CSRNet
conda install numpy
conda install -c conda-forge opencv
conda install pytorch=1.2.0
conda install torchvision
conda install pillow=6.1
conda install -c conda-forge tensorboardx
conda install tqdm
conda install matplotlib
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git