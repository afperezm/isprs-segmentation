"""
Based on https://github.com/VMarsocci/geomultitasknet
"""

import albumentations as A
import numpy as np

from albumentations.pytorch import ToTensorV2


def get_training_augmentations(mean=None, std=None):
    if std is None:
        std = [1, 1, 1]
    if mean is None:
        mean = [0, 0, 0]
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [A.CLAHE(p=1),
             A.RandomBrightnessContrast(contrast_limit=0, p=1),
             A.RandomGamma(p=1),
             ],
            p=0.9, ),
        A.OneOf(
            [A.Sharpen(p=1),
             A.Blur(blur_limit=3, p=1),
             A.MotionBlur(blur_limit=3, p=1),
             ],
            p=0.9, ),
        A.OneOf(
            [A.RandomBrightnessContrast(p=1),
             A.HueSaturationValue(p=1),
             ],
            p=0.9, ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)


def get_validation_augmentations(mean=None, std=None):
    if std is None:
        std = [1, 1, 1]
    if mean is None:
        mean = [0, 0, 0]
    train_transform = [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)


def randaugment(N, M, p, mean, std, mode="all", cut_out=False):
    # Magnitude(M) search space    
    shift_x = np.linspace(0, 150, 10)
    shift_y = np.linspace(0, 150, 10)
    rot = np.linspace(0, 30, 10)
    shear = np.linspace(0, 10, 10)
    sola = np.linspace(0, 256, 10)
    post = [4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    cont = [np.linspace(-0.8, -0.1, 10), np.linspace(0.1, 2, 10)]
    bright = np.linspace(0.1, 0.7, 10)
    shar = np.linspace(0.1, 0.9, 10)
    cut = np.linspace(0, 60, 10)
    # Transformation search space  
    Aug = [
        # 0 - geometrical
        A.ShiftScaleRotate(shift_limit_x=shift_x[M] / 256, rotate_limit=0, shift_limit_y=0,
                           shift_limit=shift_x[M] / 256, p=p),
        A.ShiftScaleRotate(shift_limit_y=shift_y[M] / 256, rotate_limit=0, shift_limit_x=0,
                           shift_limit=shift_y[M] / 256, p=p),
        A.Affine(rotate=rot[M], p=p),
        A.Affine(shear=shear[M], p=p),
        A.InvertImg(p=p),
        # 5 - Color Based
        A.Equalize(p=p),
        A.Solarize(threshold=sola[M], p=p),
        A.Posterize(num_bits=post[M], p=p),
        A.RandomBrightnessContrast(contrast_limit=(cont[0][M], cont[1][M]), brightness_limit=0, p=p),
        A.RandomBrightnessContrast(brightness_limit=bright[M], contrast_limit=0, p=p),
        A.Sharpen(alpha=shar[M], lightness=shar[M], p=p)
    ]
    # Sampling from the Transformation search space
    if mode == "geo":
        ops = np.random.choice(Aug[0:5], N)
    elif mode == "color":
        ops = np.random.choice(Aug[5:], N)
    else:
        ops = np.random.choice(Aug, N)

    ops = list(ops)

    if cut_out:
        ops.append(A.CoarseDropout(num_holes_range=(8, 8), hole_height_range=(int(cut[M]), int(cut[M])),
                                   hole_width_range=(int(cut[M]), int(cut[M])), p=p))

    ops.append(A.Normalize(mean=mean, std=std))
    ops.append(ToTensorV2())

    return A.Compose(ops)


def choose_training_augmentations(mean, std, aug_type):

    if aug_type == 'randaugment':
        print("RandAugment selected!")
        train_trans = randaugment(N=2, M=9, p=0.8, mean=mean, std=std)
    elif aug_type == 'yes':
        print("Augmentation selected!")
        train_trans = get_training_augmentations(mean=mean, std=std)
    else:
        print("No augmentation selected!")
        train_trans = get_validation_augmentations(mean=mean, std=std)

    return train_trans
