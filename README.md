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
python -u utils/dataset_split.py --images_dir $HOME/data/Potsdam/2_Ortho_RGB/ --labels_dir $HOME/data/Potsdam/5_Labels_all/ --output_dir $HOME/data/potsdam-rgb-dataset/ --patch_size 256 --stride 256 --scale 1.0 --seed 42 --crop
python -u utils/dataset_split.py --images_dir $HOME/data/Potsdam/3_Ortho_IRRG/ --labels_dir $HOME/data/Potsdam/5_Labels_all/ --output_dir $HOME/data/potsdam-irrg-dataset/ --patch_size 256 --stride 256 --scale 1.0 --seed 42 --crop
```

```bash
python -u utils/dataset_split.py --images_dir $HOME/data/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top/ --labels_dir $HOME/data/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/ --output_dir $HOME/data/vaihingen-irrg-dataset/ --patch_size 256 --stride 256 --scale 1.0 --seed 42 --crop
```

## Train domain adaptation models

```bash
python -u train.py --source_dir $HOME/data/potsdam-rgb-dataset/ --target_dir $HOME/data/vaihingen-irrg-dataset/ --epochs 50 --batch_size 1 --learning_rate_gen 0.00001 --learning_rate_dis 0.00001 --model cyclegan --comment "Potsdam RGB to Vaihingen IRRG"
python -u train.py --source_dir $HOME/data/vaihingen-irrg-dataset/ --target_dir $HOME/data/potsdam-rgb-dataset/ --epochs 50 --batch_size 1 --learning_rate_gen 0.00001 --learning_rate_dis 0.00001 --model cyclegan --comment "Vaihingen IRRG to Potsdam RGB"
python -u train.py --source_dir $HOME/data/potsdam-irrg-dataset/ --target_dir $HOME/data/vaihingen-irrg-dataset/ --epochs 50 --batch_size 1 --learning_rate_gen 0.00001 --learning_rate_dis 0.00001 --model cyclegan --comment "Potsdam IRRG to vaihingen IRRG"
python -u train.py --source_dir $HOME/data/vaihingen-irrg-dataset/ --target_dir $HOME/data/potsdam-irrg-dataset/ --epochs 50 --batch_size 1 --learning_rate_gen 0.00001 --learning_rate_dis 0.00001 --model cyclegan --comment "vaihingen IRRG to Potsdam IRRG"
```

```bash
python -u train.py --source_dir $HOME/data/potsdam-rgb-dataset/ --target_dir $HOME/data/vaihingen-irrg-dataset/ --epochs 5 --batch_size 1 --learning_rate_gen 0.0005 --learning_rate_dis 0.0001 --model colormapgan-potsdam-rgb-to-vaihingen-irrg
python -u train.py --source_dir $HOME/data/potsdam-rgb-dataset/ --target_dir $HOME/data/vaihingen-irrg-dataset/ --epochs 5 --batch_size 1 --learning_rate_gen 0.001 --learning_rate_dis 0.0001 --model colormapgan-potsdam-rgb-to-vaihingen-irrg
python -u train.py --source_dir $HOME/data/potsdam-rgb-dataset/ --target_dir $HOME/data/vaihingen-irrg-dataset/ --epochs 5 --batch_size 1 --learning_rate_gen 0.0001 --learning_rate_dis 0.00001 --model colormapgan-potsdam-rgb-to-vaihingen-irrg
```

## Test domain adaptation models

```bash
python -u test.py --source_dir $HOME/data/potsdam-rgb-dataset/ --target_dir $HOME/data/vaihingen-irrg-dataset/ --output_dir ./submits/ --checkpoints_dir ./results/cyclegan-240417-202255/checkpoints/ --model epoch=4-step=158700.ckpt --enable_progress_bar
```
