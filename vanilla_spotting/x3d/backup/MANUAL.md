## SET ENVIRONMENT
```
# Minimum requirements to run the model through the torch hub

conda create -n py311 python=3.11 -y
conda activate py311

# According to your cuda env.
pip install torch==2.8.0 torchvision==0.23.0  --index-url https://download.pytorch.org/whl/cu129
pip install 'git+https://github.com/facebookresearch/fvcore'
conda install av -c conda-forge -y

pip install opencv-python scikit-learn pandas matplotlib 

```


---
#### Based on INSTALL.md
```
pip install numpy simplejson psutil tensorboard 
pip install torch==1.4.0
pip install 'git+https://github.com/facebookresearch/fvcore'

conda install av -c conda-forge -y
pip install -U iopath
conda install torchvision -c  pytorch -y
pip install  moviepy 

pip install --upgrade \
    git+https://github.com/facebookresearch/pytorchvideo.git@mainpytorchvideo

pip install 'git+https://github.com/facebookresearch/fairscale'
pip install scikit-learn

pip install -U torch torchvision cython
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
# You can find more details at https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

pip install  pandas matplotlib pillow

export PYTHONPATH="C:\Users\USER\Desktop\jjs\SlowFast\slowfast"
```

### Execution
```
python tools/run_net.py --cfg configs/Kinetics/X3D_XS.yaml DATA."C:\KJM\abnormal_behavior\DB\UCF_Crimes\Action_Regnition_splits(classifiction)"
```