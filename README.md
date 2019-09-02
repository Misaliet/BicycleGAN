# Deep Generative Streets
## This project is based on Bicycle GAN. The original description can be seen below.
## Dataset can be downloaded from Digimap and pre-process with the scripts in Misc folder. Some well-trained models can be downloaded from the link in my report.
## Some useful examples with scripts in Misc folder.
### Create basic dataset (10 is number of original images, 0 is mode for AtoB and 4 is the size of naming):
```bash
bash makedata.sh satellite_images_path map_images_path 10 0 4
```
### Create dataset for spatial varying z method:
```bash
python imageProcess.py -s map_images_path -r satellite_images_path -m 0 -d 5
```
### Combine low-resolution images to high-resolution images in a large scale (3072 is resolution):
```bash
python combineImages.py images_path 3072
```
### Generate new low-resolution images and combine them to one high-resolution image (change code inside this script for different tasks):
```bash
bash whole.sh
```
### Extract seams from one high-resolution image and then use Seams Bicycle GAN to erase seams and replace new images back to high-resolution image (change code inside this script for different tasks), this can also be used for create Seams dataset:
```bash
bash replace.sh
```
## Some useful examples for training or testing.
### Basic training or testing (final is the name of dataset folder):
```bash
python train.py --dataroot ./datasets/final --name final_bicycleGAN --model bicycle_gan --direction AtoB --load_size 256
python test.py --dataroot ./datasets/final --direction AtoB --model bicycle_gan --name final_bicycleGAN_pretrained --results_dir ./results/final --phase test  --no_flip --load_size 256
```
### Seams Bicycle GAN training or testing (seamsL is the name of dataset folder):
```bash
python train.py --dataroot ./datasets/seamsL --name seamsL_bicycleGAN --model bicycle_gan --direction AtoB --load_size 256
python test.py --dataroot ./datasets/seamsL --direction AtoB --model bicycle_gan --name seamsL_bicycleGAN_pretrained --results_dir ./results/seamsL --phase test  --no_flip --load_size 256
```
### 6-channel Seams Bicycle GAN training or testing (6 channel images can not be saved so "--display_id -1 --no_html" must be added in training phase):
```bash
python train.py --dataroot ./datasets/seamsL6 --name seamsL6_bicycleGAN --model bicycle_gan --direction AtoB --load_size 256 --input_nc 6 --dataset_mode aligned6c --display_id -1 --no_html
python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL6_bicycleGAN_pretrained --results_dir ./results/seamsL6 --phase test  --no_flip --load_size 256 --input_nc 6 --dataset_mode aligned6c
```
### 7-channel Seams Bicycle GAN training or testing (mask_name is the mask file in /dataset/seamsL6/mask/):
```bash
python train.py --dataroot ./datasets/seamsL6 --name seamsL7m1 --model bicycle_gan --direction AtoB --load_size 256 --input_nc 7 --dataset_mode aligned7c --mask_name nborder2_25.png --display_id -1 --no_html
python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7m1_bicycleGAN_pretrained --results_dir ./results/seamsL7m1 --phase test  --no_flip --load_size 256 --input_nc 7 --dataset_mode aligned7c --mask_name nborder2_25.png 
```
### 7-channel Seams Bicycle GAN with new loss function training or testing (bicycle_ganML model should be used instead of bicycle_gan):
```bash
python train.py --dataroot ./datasets/seamsL6 --name seamsL7m1l --model bicycle_ganML --direction AtoB --load_size 256 --input_nc 7 --dataset_mode aligned7c --mask_name nborder2_25.png --lambda_ml 30 --display_id -1 --no_html
python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_ganML --name seamsL7m1l_bicycleGAN_pretrained --results_dir ./results/seamsL7m1l --num_test 50 --phase test  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name nborder2_25.png
```
### Spatial varying z method (bicycle_gan4TZ model should be used instead of bicycle_gan, only add z in input layer so "--where_add input" is added):
```bash
python train.py --dataroot ./datasets/z --name 4tz_bicycleGAN --model bicycle_gan4TZ --direction AtoB --load_size 256 --where_add input
python test.py --dataroot ./datasets/z --direction AtoB --model bicycle_gan4TZ --name 4tz_bicycleGAN_pretrained --results_dir ./results/4tz --phase test  --no_flip --load_size 256 --where_add input
```
#
# The following is original README from Bicycle GAN.
<img src='imgs/day2night.gif' align="right" width=360>

<br><br><br><br>

# BicycleGAN
[Project Page](https://junyanz.github.io/BicycleGAN/) |  [Paper](https://arxiv.org/abs/1711.11586) | [Video](https://youtu.be/JvGysD2EFhw)


Pytorch implementation for multimodal image-to-image translation. For example,  given the same night image, our model is able to synthesize possible day images with different types of lighting, sky and clouds. The training requires paired data.

**Note**: The current software works well with PyTorch 0.41+. Check out the older [branch](https://github.com/junyanz/BicycleGAN/tree/pytorch0.3.1) that supports PyTorch 0.1-0.3.

<img src='imgs/teaser.jpg' width=850>  

**Toward Multimodal Image-to-Image Translation.**  
[Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/),
 [Richard Zhang](https://richzhang.github.io/), [Deepak Pathak](http://people.eecs.berkeley.edu/~pathak/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/), [Oliver Wang](http://www.oliverwang.info/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/).  
 UC Berkeley and Adobe Research  
In Neural Information Processing Systems, 2017.

## Example results
<img src='imgs/results_matrix.jpg' width=820>  


## Other Implementations
- [[Tensorflow]](https://github.com/gitlimlab/BicycleGAN-Tensorflow) by Youngwoon Lee (USC CLVR Lab).
- [[Tensorflow]](https://github.com/kvmanohar22/img2imgGAN) by Kv Manohar.

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN


## Getting Started ###
### Installation
- Clone this repo:
```bash
git clone -b master --single-branch https://github.com/junyanz/BicycleGAN.git
cd BicycleGAN
```
- Install PyTorch and dependencies from http://pytorch.org
- Install python libraries [visdom](https://github.com/facebookresearch/visdom), [dominate](https://github.com/Knio/dominate), and [moviepy](https://github.com/Zulko/moviepy).   

For pip users:
```bash
bash ./scripts/install_pip.sh
```

For conda users:
```bash
bash ./scripts/install_conda.sh
```


### Use a Pre-trained Model
- Download some test photos (e.g., edges2shoes):
```bash
bash ./datasets/download_testset.sh edges2shoes
```
- Download a pre-trained model (e.g., edges2shoes):
```bash
bash ./pretrained_models/download_model.sh edges2shoes
```

- Generate results with the model
```bash
bash ./scripts/test_edges2shoes.sh
```
The test results will be saved to a html file here: `./results/edges2shoes/val/index.html`.

- Generate results with synchronized latent vectors
```bash
bash ./scripts/test_edges2shoes.sh --sync
```
Results can be found at `./results/edges2shoes/val_sync/index.html`.

### Generate Morphing Videos
- We can also produce a morphing video similar to this [GIF](imgs/day2night.gif) and Youtube [video](http://www.youtube.com/watch?v=JvGysD2EFhw&t=2m21s).
```bash
bash ./scripts/video_edges2shoes.sh
```
Results can be found at `./videos/edges2shoes/`.

### Model Training
- To train a model, download the training images (e.g., edges2shoes).
```bash
bash ./datasets/download_dataset.sh edges2shoes
```

- Train a model:
```bash
bash ./scripts/train_edges2shoes.sh
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out  `./checkpoints/edges2shoes_bicycle_gan/web/index.html`
- See more training details for other datasets in `./scripts/train.sh`.

### Datasets (from pix2pix)
Download the datasets using the following script. Many of the datasets are collected by other researchers. Please cite their papers if you use the data.
- Download the testset.
```bash
bash ./datasets/download_testset.sh dataset_name
```
- Download the training and testset.
```bash
bash ./datasets/download_dataset.sh dataset_name
```
- `facades`: 400 images from [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade). [[Citation](datasets/bibtex/facades.tex)]
- `maps`: 1096 training images scraped from Google Maps
- `edges2shoes`: 50k training images from [UT Zappos50K dataset](http://vision.cs.utexas.edu/projects/finegrained/utzap50k). Edges are computed by [HED](https://github.com/s9xie/hed) edge detector + post-processing. [[Citation](datasets/bibtex/shoes.tex)]
- `edges2handbags`: 137K Amazon Handbag images from [iGAN project](https://github.com/junyanz/iGAN). Edges are computed by [HED](https://github.com/s9xie/hed) edge detector + post-processing. [[Citation](datasets/bibtex/handbags.tex)]
- `night2day`: around 20K natural scene images from  [Transient Attributes dataset](http://transattr.cs.brown.edu/) [[Citation](datasets/bibtex/transattr.tex)]

## Models
Download the pre-trained models with the following script.
```bash
bash ./pretrained_models/download_model.sh model_name
```
- `edges2shoes` (edge -> photo) trained on UT Zappos50K dataset.
- `edges2handbags` (edge -> photo) trained on Amazon handbags images..
```bash
bash ./pretrained_models/download_model.sh edges2handbags
bash ./datasets/download_testset.sh edges2handbags
bash ./scripts/test_edges2handbags.sh
```
- `night2day` (nighttime scene -> daytime scene) trained on around 100 [webcams](http://transattr.cs.brown.edu/).
```bash
bash ./pretrained_models/download_model.sh night2day
bash ./datasets/download_testset.sh night2day
bash ./scripts/test_night2day.sh
```
- `facades` (facade label -> facade photo) trained on the CMP Facades dataset.
```bash
bash ./pretrained_models/download_model.sh facades
bash ./datasets/download_testset.sh facades
bash ./scripts/test_facades.sh
```
- `maps` (map photo -> aerial photo) trained on 1096 training images scraped from Google Maps.
```bash
bash ./pretrained_models/download_model.sh maps
bash ./datasets/download_testset.sh maps
bash ./scripts/test_maps.sh
```

### Citation

If you find this useful for your research, please use the following.

```
@inproceedings{zhu2017toward,
  title={Toward multimodal image-to-image translation},
  author={Zhu, Jun-Yan and Zhang, Richard and Pathak, Deepak and Darrell, Trevor and Efros, Alexei A and Wang, Oliver and Shechtman, Eli},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}

```
If you use modules from CycleGAN or pix2pix paper, please use the following:
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```
### Acknowledgements

This code borrows heavily from the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.
