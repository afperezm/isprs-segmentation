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
