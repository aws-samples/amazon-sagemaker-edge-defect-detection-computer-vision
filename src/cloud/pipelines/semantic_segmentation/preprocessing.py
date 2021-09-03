# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import sys
import os
import subprocess

# Install packages previous to executing the rest of the script. You can also build your own custom container
#   with your individual dependencies if needed
subprocess.check_call([sys.executable, "-m", "pip", "install", "wget", "opencv-python","albumentations","tqdm"])
os.system("apt-get update")
os.system("apt-get install ffmpeg libsm6 libxext6  -y")

import argparse
import json
from glob import glob
import shutil
from PIL import Image
from pathlib import Path

import cv2
from tqdm import tqdm
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip

from sklearn.model_selection import train_test_split


# Constants

# the "folders" in the S3 bucket for images and their ground truth masks
PREFIX_NAME_IMAGE = 'images'
PREFIX_NAME_MASK = 'masks'

# The images size used
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

def augment_data(path, augment=True):
    save_path = path
    images = sorted(glob(os.path.join(path, PREFIX_NAME_IMAGE + "/*")))
    masks = sorted(glob(os.path.join(path, PREFIX_NAME_MASK + "/*")))
    
    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("/")[-1].split(".")
        
        img_name = name[0]
        image_extn = name[1]

        name = y.split("/")[-1].split(".")
        mask_name = name[0]
        mask_extn = name[1]

        # Read image mask
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        # Augment dataset
        if augment == True:
            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = GridDistortion(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            save_images = [x, x1, x2, x3, x4, x5]
            save_masks =  [y, y1, y2, y3, y4, y5]

        else:
            save_images = [x]
            save_masks = [y]

        """ Saving the image and mask. """
        idx = 0
        Path(save_path + "/" + PREFIX_NAME_IMAGE ).mkdir(parents=True, exist_ok=True)
        Path(save_path + "/" + PREFIX_NAME_MASK ).mkdir(parents=True, exist_ok=True)
        for i, m in zip(save_images, save_masks):
            i = cv2.resize(i, (IMAGE_WIDTH, IMAGE_HEIGHT))
            m = cv2.resize(m, (IMAGE_WIDTH, IMAGE_HEIGHT))

            if len(images) == 1:
                tmp_img_name = f"{img_name}.{image_extn}"
                tmp_mask_name = f"{mask_name}.{mask_extn}"
            else:
                tmp_img_name = f"{img_name}_{idx}.{image_extn}"
                tmp_mask_name = f"{mask_name}_{idx}.{mask_extn}"

            image_path = os.path.join(save_path, PREFIX_NAME_IMAGE, tmp_img_name)
            mask_path = os.path.join(save_path, PREFIX_NAME_MASK, tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1


def resize_images(path, width, height):
    """Resize all images in a given path (in-place). Please note that this method
    overwrites existing images in the path"""
    files = glob(os.path.join(path, '*.png')) + glob(os.path.join(path, '*.jpg'))
    for file in files:
        im = Image.open(file)
        im_resized = im.resize((width, height), Image.ANTIALIAS)
        im_resized.save(file)
        
def get_square_image(img, padding_color=(0, 0, 0)):
    """Returns a squared image by adding black padding"""
    width, height = img.size
    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), padding_color)
        result.paste(img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(img.mode, (height, height), padding_color)
        result.paste(img, ((height - width) // 2, 0))
        return result

def square_images(path, padding_color=(0,0,0)):
    """Squares all images in a given path (in-place). Please note that this
    method overwrites existing images in the path."""
    files = glob(os.path.join(path, '*.png')) + glob(os.path.join(path, '*.jpg'))
    for file in files:
        im = Image.open(file)
        im_squared = get_square_image(im, padding_color)
        im_squared.save(file)
        
def load_data(path, split=0.1):
    images = sorted(glob(os.path.join(path, PREFIX_NAME_IMAGE + "/*")))
    masks = sorted(glob(os.path.join(path, PREFIX_NAME_MASK + "/*")))

    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)
    print(total_size)
    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=float, default=0.1)
    args, _ = parser.parse_known_args()

    print('Received arguments {}'.format(args))

    # Define the paths
    input_data_base_path = '/opt/ml/processing/input'
    train_output_base_path = '/opt/ml/processing/train'
    test_output_base_path = '/opt/ml/processing/test'
    val_output_base_path = '/opt/ml/processing/val'
    report_output_base_path = '/opt/ml/processing/report'
    
    #Augment images and save in new directory
    augment_data(input_data_base_path)
    
    print('Squaring images...')
    square_images(os.path.join(input_data_base_path, PREFIX_NAME_IMAGE))
    square_images(os.path.join(input_data_base_path, PREFIX_NAME_MASK), padding_color=(0))
    
    # Resize the images in-place in the container image
    print('Resizing images...')
    resize_images(os.path.join(input_data_base_path, PREFIX_NAME_IMAGE), IMAGE_WIDTH, IMAGE_HEIGHT)
    resize_images(os.path.join(input_data_base_path, PREFIX_NAME_MASK), IMAGE_WIDTH, IMAGE_HEIGHT)

    # Create train test validation split
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(input_data_base_path, split=float(args.split))
    
    # Copy to the output paths
    Path(train_output_base_path + "/" + PREFIX_NAME_IMAGE ).mkdir(parents=True, exist_ok=True)
    Path(train_output_base_path + "/" + PREFIX_NAME_MASK ).mkdir(parents=True, exist_ok=True)
    Path(val_output_base_path + "/" + PREFIX_NAME_IMAGE ).mkdir(parents=True, exist_ok=True)
    Path(val_output_base_path + "/" + PREFIX_NAME_MASK ).mkdir(parents=True, exist_ok=True)
    Path(test_output_base_path + "/" + PREFIX_NAME_IMAGE ).mkdir(parents=True, exist_ok=True)
    Path(test_output_base_path + "/" + PREFIX_NAME_MASK ).mkdir(parents=True, exist_ok=True)
    for file in train_x :
        shutil.copy(file, os.path.join(train_output_base_path, PREFIX_NAME_IMAGE + '/' + os.path.basename(file)))
    for file in train_y :
        shutil.copy(file, os.path.join(train_output_base_path, PREFIX_NAME_MASK + '/'+ os.path.basename(file)))
    for file in valid_x :
        shutil.copy(file, os.path.join(val_output_base_path, PREFIX_NAME_IMAGE + '/'+ os.path.basename(file)))
    for file in valid_y :
        shutil.copy(file, os.path.join(val_output_base_path, PREFIX_NAME_MASK + '/'+ os.path.basename(file)))
    for file in test_x :
        shutil.copy(file, os.path.join(test_output_base_path, PREFIX_NAME_IMAGE + '/'+ os.path.basename(file)))
    for file in test_y :
        shutil.copy(file, os.path.join(test_output_base_path, PREFIX_NAME_MASK + '/'+ os.path.basename(file)))
    # Save the preprocessing report to make information available to downstream steps
    preprocessing_report = {
        'preprocessing': {
            'dataset': {
                'num_training_samples': len(train_x),
                'num_test_samples': len(test_x),
                'num_val_samples': len(valid_x)
            }
        }
    }
    print('Preprocessing report:', preprocessing_report)
    report_output_path = os.path.join(report_output_base_path, 'preprocessing_report.json')
    with open(report_output_path, "w") as f:
            f.write(json.dumps(preprocessing_report))

    
