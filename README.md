# Code for RelPose++

## Setup Dependencies

We recommend using a conda environment to manage dependencies. Install a version of
Pytorch compatible with your CUDA version from the [Pytorch website](https://pytorch.org/get-started/locally/).

```
git clone --depth 1 https://github.com/amyxlase/relpose-plus-plus.git
conda create -n relposepp python=3.8
conda activate relposepp
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

Then, follow the directions to install Pytorch3D [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).


## Run Demo

Download pre-trained weights:
```
mkdir -p weights
gdown https://drive.google.com/uc?id=1FGwMqgLXv4R0xMzEKVv3n3Aghn0MQXKY&export=download
unzip relposepp_weigths.zip -d weights
```

The demo can be run on any image directory with 2-8 images. Each image must be
associated with a bounding box. The colab notebook has an interactive interface for
selecting bounding boxes.

The bounding boxes can either be extracted automatically from masks or specified in a
json file.

Running demo with masks:
```
python relpose/demo.py  --image_dir examples/robot/images \
    --mask_dir examples/robot/masks --output_path robot.html
```

Running demo with specified bounding boxes:
```
python relpose/demo.py  --image_dir examples/robot/images \
    --bbox_path examples/robot/bboxes.json --output_path robot.html
```

The demo will output an html file that can be opened in a browser. The html file will
display the input images and predicted cameras. An example is shown [here](https://amyxlase.github.io/relpose-plus-plus/robot.html).

## Pre-processing CO3D

Download the CO3Dv2 dataset from [here](https://github.com/facebookresearch/co3d/tree/main).

Then, pre-process the annotations:
```
python -m preprocess.preprocess_co3d --category all --precompute_bbox \
    --co3d_v2_dir /path/to/co3d_v2
python -m preprocess.preprocess_co3d --category all \
    --co3d_v2_dir /path/to/co3d_v2
```


## Training

Trainer should be run via:
```
torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=8 \
relpose/trainer_ddp.py --batch_size=48 --num_images=8 --random_num_images=true  \
--gpu_ids=0,1,2,3,4,5,6,7 --lr=1e-5 --normalize_cameras --use_amp 
```
Our released model was trained to 800,000 iterations using 8 GPUS (A6000).


Pending Items for Code Release: 
- [ ] Add more example images
- [ ] Colab Notebook
- [ ] Evaluation code
