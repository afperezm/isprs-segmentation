This repository contains code for the paper:

ColorMapGAN: Unsupervised Domain Adaptation for Semantic Segmentation Using Color Mapping Generative Adversarial Networks. [arXiv, 2019](https://arxiv.org/pdf/1907.12859.pdf).

## Requirements

- Python 3
- PyTorch 2.2.2 >=
- Lightning 2.2.1 >=

## Contents

This is the implementation of the core method of ColorMapGAN, containing a total of three files:

+ deeplabv2.py: This is the segmentation model. The original paper used Unet, here I use deeplabv2
+ generator.py: This is the implementation of the generator in ColorMapGAN and the core innovation of the paper.
+ discriminator.py: This is the implementation of the discriminator in ColorMapGAN.


## Preparing the dataset

https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx

```bash
python -u utils/split_isprs_dataset.py --images_dir $HOME/data/Potsdam/2_Ortho_RGB/ --labels_dir $HOME/data/Potsdam/5_Labels_all/ --output_dir $HOME/data/potsdam-rgb-dataset-512-256/ --patch_size 512 --stride 256 --seed 42 --pad
python -u utils/split_isprs_dataset.py --images_dir $HOME/data/Potsdam/3_Ortho_IRRG/ --labels_dir $HOME/data/Potsdam/5_Labels_all/ --output_dir $HOME/data/potsdam-irrg-dataset-512-256/ --patch_size 512 --stride 256 --seed 42 --pad
python -u utils/split_isprs_dataset.py --images_dir $HOME/data/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top/ --labels_dir $HOME/data/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/ --output_dir $HOME/data/vaihingen-irrg-dataset-512-256/ --patch_size 512 --stride 256 --seed 42 --pad
```

## Train baseline segmentation models

```bash
python -u train.py --data_dir $HOME/data/potsdam-rgb-dataset-256-128/ --results_dir ./results/segmentation/ --epochs 50 --batch_size 32 --learning_rate 0.0001 0.0001 --model deeplabv3 --dataset isprs --comment "Train DeepLabV3 segmentation model with ResNet-50 backbone on Potsdam RGB images of size 256x256 cropped with a stride of 128"
python -u train.py --data_dir $HOME/data/potsdam-irrg-dataset-256-128/ --results_dir ./results/segmentation/ --epochs 50 --batch_size 32 --learning_rate 0.0001 0.0001 --model deeplabv3 --dataset isprs --comment "Train DeepLabV3 segmentation model with ResNet-50 backbone on Potsdam IRRG images of size 256x256 cropped with a stride of 128"
python -u train.py --data_dir $HOME/data/vaihingen-irrg-dataset-256-128/ --results_dir ./results/segmentation/ --epochs 50 --batch_size 32 --learning_rate 0.0001 0.0001 --model deeplabv3 --dataset isprs --comment "Train DeepLabV3 segmentation model with ResNet-50 backbone on Vaihingen IRRG images of size 256x256 cropped with a stride of 128"
```

## Train domain adaptation models

```bash
python -u train.py --data_dir $HOME/data/potsdam-rgb-dataset/ $HOME/data/vaihingen-irrg-dataset/ --results_dir ./results/cyclegan/ --epochs 50 --batch_size 1 --learning_rate 0.00001 0.00001 --dataset unpaired --model cyclegan --comment "Potsdam RGB to Vaihingen IRRG"
python -u train.py --data_dir $HOME/data/vaihingen-irrg-dataset/ $HOME/data/potsdam-rgb-dataset/ --results_dir ./results/cyclegan/ --epochs 50 --batch_size 1 --learning_rate 0.00001 0.00001 --dataset unpaired --model cyclegan --comment "Vaihingen IRRG to Potsdam RGB"
python -u train.py --data_dir $HOME/data/potsdam-irrg-dataset/ $HOME/data/vaihingen-irrg-dataset/ --results_dir ./results/cyclegan/ --epochs 50 --batch_size 1 --learning_rate 0.00001 0.00001 --dataset unpaired --model cyclegan --comment "Potsdam IRRG to Vaihingen IRRG"
python -u train.py --data_dir $HOME/data/vaihingen-irrg-dataset/ $HOME/data/potsdam-irrg-dataset/ --results_dir ./results/cyclegan/ --epochs 50 --batch_size 1 --learning_rate 0.00001 0.00001 --dataset unpaired --model cyclegan --comment "Vaihingen IRRG to Potsdam IRRG"
```

```bash
python -u train.py --data_dir $HOME/data/potsdam-rgb-dataset-256-128/ $HOME/data/vaihingen-irrg-dataset-256-128/ --results_dir ./results/isprs/ --epochs 10 --batch_size 32 --learning_rate 0.00005 0.00001 --dataset unpaired --model colormapgan --comment "Vaihingen IRRG to Potsdam RGB"
python -u train.py --data_dir $HOME/data/vaihingen-irrg-dataset-256-128/ $HOME/data/potsdam-rgb-dataset-256-128/ --results_dir ./results/isprs/ --epochs 10 --batch_size 32 --learning_rate 0.00005 0.00001 --dataset unpaired --model colormapgan --comment "Potsdam RGB to Vaihingen IRRG"
python -u train.py --data_dir $HOME/data/potsdam-irrg-dataset/ $HOME/data/vaihingen-irrg-dataset/ --results_dir ./results/colormapgan/ --epochs 5 --batch_size 1 --learning_rate 0.0001 0.00001 --dataset unpaired --model colormapgan --comment "Potsdam IRRG to Vaihingen IRRG"
python -u train.py --data_dir $HOME/data/vaihingen-irrg-dataset/ $HOME/data/potsdam-irrg-dataset/ --results_dir ./results/colormapgan/ --epochs 5 --batch_size 1 --learning_rate 0.0001 0.00001 --dataset unpaired --model colormapgan --comment "Vaihingen IRRG to Potsdam IRRG"
```

## Train PASTIS domain adaptation models

```bash
python -u train.py --data_dir $HOME/data/pastis-dataset-exploded/fold_4/ $HOME/data/pastis-dataset-exploded/folds_1_2_3/ --results_dir ./results/pastis/ --epochs 50 --batch_size 16 --learning_rate 0.00001 0.00001 --dataset unpaired --model cyclegan --comment "PASTIS tiles 1-2-3 to PASTIS tile 4"
python -u train.py --data_dir $HOME/data/pastis-dataset-exploded/fold_4/ $HOME/data/pastis-dataset-exploded/folds_1_2_3/ --results_dir ./results/pastis/ --epochs 10 --batch_size 32 --learning_rate 0.00005 0.00001 --dataset unpaired --model colormapgan --comment "PASTIS tiles 1-2-3 to PASTIS tile 4"
```

## Test domain adaptation models

```bash
python -u test.py --data_dir $HOME/data/potsdam-rgb-dataset-256-128/ $HOME/data/vaihingen-irrg-dataset/ --output_dir ./submits/ --model cyclegan --dataset unpaired --ckpt_path ./results/cyclegan/cyclegan-240501-033802/checkpoints/epoch=20-step=666372.ckpt --enable_progress_bar --predict_only
python -u test.py --data_dir $HOME/data/vaihingen-irrg-dataset-256-128/ $HOME/data/potsdam-rgb-dataset/ --output_dir ./submits/ --model cyclegan --dataset unpaired --ckpt_path ./results/cyclegan/cyclegan-240501-033914/checkpoints/epoch=20-step=666372.ckpt --enable_progress_bar --predict_only
python -u test.py --data_dir $HOME/data/potsdam-irrg-dataset-256-128/ $HOME/data/vaihingen-irrg-dataset/ --output_dir ./submits/ --model cyclegan --dataset unpaired --ckpt_path ./results/cyclegan/cyclegan-240501-033956/checkpoints/epoch=19-step=634640.ckpt --enable_progress_bar --predict_only
python -u test.py --data_dir $HOME/data/vaihingen-irrg-dataset-256-128/ $HOME/data/potsdam-irrg-dataset/ --output_dir ./submits/ --model cyclegan --dataset unpaired --ckpt_path ./results/cyclegan/cyclegan-240501-034122/checkpoints/epoch=20-step=666372.ckpt --enable_progress_bar --predict_only
```

```bash
python -u test.py --data_dir $HOME/data/potsdam-rgb-dataset-256-128/ $HOME/data/vaihingen-irrg-dataset/ --output_dir ./submits/ --model colormapgan --dataset unpaired --ckpt_path ./results/colormapgan/colormapgan-240505-043518/checkpoints/epoch=4-step=158660.ckpt --enable_progress_bar --predict_only
python -u test.py --data_dir $HOME/data/vaihingen-irrg-dataset-256-128/ $HOME/data/potsdam-rgb-dataset/ --output_dir ./submits/ --model colormapgan --dataset unpaired --ckpt_path ./results/colormapgan/colormapgan-240505-043720/checkpoints/epoch=4-step=158660.ckpt --enable_progress_bar --predict_only
python -u test.py --data_dir $HOME/data/potsdam-irrg-dataset-256-128/ $HOME/data/vaihingen-irrg-dataset/ --output_dir ./submits/ --model colormapgan --dataset unpaired --ckpt_path ./results/colormapgan/colormapgan-240505-043759/checkpoints/epoch=4-step=158660.ckpt --enable_progress_bar --predict_only
python -u test.py --data_dir $HOME/data/vaihingen-irrg-dataset-256-128/ $HOME/data/potsdam-irrg-dataset/ --output_dir ./submits/ --model colormapgan --dataset unpaired --ckpt_path ./results/colormapgan/colormapgan-240505-043836/checkpoints/epoch=4-step=158660.ckpt --enable_progress_bar --predict_only
```

## FLAIR experiments

```bash
python -u train.py --data_dir $HOME/data/flair-dataset/ --results_dir ./results/flair/ --epochs 50 --batch_size 8 --learning_rate 0.0001 0.0001 --model deeplabv3 --dataset flair --comment "Train DeepLabV3 segmentation model with ResNet-50 backbone on FLAIR RGB images of of size 512x512"
python -u test.py --data_dir $HOME/data/flair-dataset/ --output_dir ./submits/ --ckpt_path ./results/flair/colormapgan-240929-041910/checkpoints/epoch=9-step=10040.ckpt --dataset unpaired-flair --model deeplabv3 --enable_progress_bar --test_only
```

```bash
python -u train.py --data_dir $HOME/data/flair-dataset/ --results_dir ./results/flair/ --epochs 10 --batch_size 32 --learning_rate 0.00005 0.00001 --dataset unpaired-flair --model colormapgan --comment "FLAIR test subset to FLAIR train subset"
```
