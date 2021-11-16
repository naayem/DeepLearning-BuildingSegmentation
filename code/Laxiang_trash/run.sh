# install dependencies: 
pip install -r ./requirements.txt
pip install pyyaml==5.1
gcc --version
pip install gdown
# opencv is pre-installed on colab

# install detectron2: (Colab has CUDA 10.2 + torch 1.8)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html
# exit(0)  # After installation, you need to "restart runtime" in Colab. This line can also restart runtime


 # ============ Affnet  ============
pip install kornia pydegensac extract_patches
pip install opencv-python
pip install opencv-contrib-python==3.4.2.17 


# download weights of affnet
# wget https://github.com/ducha-aiki/affnet/raw/master/convertJIT/AffNetJIT.pt  -P ./data/affnet_weight
# wget https://github.com/ducha-aiki/affnet/raw/master/convertJIT/OriNetJIT.pt -P ./data/affnet_weight
# wget https://github.com/ducha-aiki/affnet/raw/master/test-graf/H1to6p  -P ./data/affnet_weight/

