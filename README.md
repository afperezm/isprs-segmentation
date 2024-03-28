# ColorMapGAN-PyTorch
A PyTorch implementation of ColorMapGAN (TGRS2020)
<a href="https://arxiv.org/pdf/1907.12859.pdf">ColorMapGAN: Unsupervised Domain Adaptation for Semantic Segmentation Using Color Mapping Generative Adversarial Networks</a>

This is the implementation of the core method of ColorMapGAN, containing a total of three files:

+ deeplabv2.py: This is the segmentation model. The original paper used Unet, here I use deeplabv2
+ generator.py: This is the implementation of the generator in ColorMapGAN and the core innovation of the paper.
+ discriminator.py: This is the implementation of the discriminator in ColorMapGAN.

The complete code for data loading, training, and testing is not included here. If you need the complete code, please follow the official account and leave a message in the background (WeChat search: Achai and her CV learning diary)
[![QR code](https://github.com/AI-Chen/ColorMapGAN/blob/main/qrcode_for_gh_e41e549f33cd_344.jpg "QR code")]
