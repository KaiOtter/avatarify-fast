# avatarify-fast
One-click video face-swap. This is a pure python scripts without any GUI. It's based on avatarify-python and uses face landmarks lib to implement a pipeline for processing frames automaticly.

# I merge code from:
https://github.com/alievk/avatarify-python  
https://github.com/AliaksandrSiarohin/motion-cosegmentation  
https://github.com/ainrichman/Peppa-Facial-Landmark-PyTorch  


## Requirements
pytorch  
opencv-python==4.2.0.34  
face-alignment==1.3.3  
pyzmq==20.0.0  
msgpack-numpy==0.4.7.1  
pyyaml==5.3.1  


## Weights prepare
Downlowd models' weights of avatarify-python "vox-adv-cpk.pth.tar" from  
https://github.com/alievk/avatarify-python/tree/master/docs#download-network-weights

avatarify-python need face-alligment library.

    git clone https://github.com/1adrianb/face-alignment
    cd face-alignment
    pip install -r requirements.txt
    python setup.py install

## Runnig Steps
#### 1. run extract_audio.sh to gain soundtrack mp3 file.
#### 2. run demo.py with re-write params in local path.
#### 3. run merge_audio.sh to merge soundtrack and video.




