# Downloading Dataset

ePillID Dataset found here https://github.com/usuyama/ePillID-benchmark/releases

Download the ePillID dataset and place it under the /datasets/ folder

# Setting up the env

conda create -n epill python=3.9.18
conda activate epill

conda install -c conda-forge scikit-learn=1.4.1.post1

conda install cudatoolkit pytorch=2.0.0 torchvision=0.15.0 torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

pip install imutils==0.5.4

pip install opencv-python

pip install tqdm

conda install imgaug=0.4.0

conda install ipykernel

- imgaug library is 4+ years old now. They use the depricated np.bool.

    - change np.bool in this file to np.bool_

    - cd /anaconda3/envs/epill/lib/python3.9/site-packages/imgaug/augmenters/meta.py

# Setting up the env by PIP
 pip install -r requirements.txt

#Trian

 DINOv1: run python -m torch.distributed.run --nproc_per_node=1 train_dinov1.py
 DINOv2: run python -m torch.distributed.run --nproc_per_node=1 train_dinov2.py

#Evaluation

DINOv1: run python eval_ePillID.py 
DINOv2: run python eval_ePillIDv2.py



