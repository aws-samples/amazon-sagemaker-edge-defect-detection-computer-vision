# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import sys
import os
import subprocess

# Install packages previous to executing the rest of the script. You can also build your own custom container
#   with your individal dependecies if needed
subprocess.check_call([sys.executable, "-m", "pip", "install", "Augmentor", "wget", "mxnet", "opencv-python"])
os.system("apt-get update -y")
os.system("apt-get install ffmpeg libsm6 libxext6  -y")

import argparse
import json
import warnings
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
import shutil
import wget
from PIL import Image
import Augmentor

from sklearn.model_selection import train_test_split


# Constants

# the "folders" in the S3 bucket which define which images are good or bad
PREFIX_NAME_NORMAL = 'normal'
PREFIX_NAME_ANOMALOUS = 'anomalous'

# The images size used
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# Download im2rec.py tool for RecordIO conversion
filename_im2rec_tool = wget.download("https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py")

def augment_data(path, sample_count):
    """Augments the image dataset in the given path by adding rotation, zoom,,
    brightness, contrast to the dataset"""
    p = Augmentor.Pipeline(path, output_directory=path)

    # Define augmentation operations
    p.rotate(probability=0.4, max_left_rotation=8, max_right_rotation=8)
    p.zoom(probability=0.3, min_factor=1.1, max_factor=1.3)
    p.random_brightness(probability=0.3, min_factor=0.4, max_factor=0.9)
    p.random_contrast(probability=0.2, min_factor=0.9, max_factor=1.4)

    p.sample(sample_count)


def load_data(path, split=0.1):
    """Load the images from the path and do a train-test-split"""

    label_map = { 'good': 0, 'bad': 1 }
    bad = sorted(glob(os.path.join(path, "%s/*" % PREFIX_NAME_ANOMALOUS)))
    good = sorted(glob(os.path.join(path, "%s/*" % PREFIX_NAME_NORMAL)))
    
    images = bad + good
    labels = ([label_map['bad']] * len(bad)) + ([label_map['good']] * len(good))

    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)
    print('Total size:', total_size)
    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(labels, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def resize_images(path, width, height):
    """Resize all images in a given path (in-place). Please note that this method
    overwrites existing images in the path"""
    files = glob(os.path.join(path, '*.png')) + glob(os.path.join(path, '*.jpg'))
    for file in files:
        im = Image.open(file)
        im_resized = im.resize((width, height), Image.ANTIALIAS)
        im_resized.save(file)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment-count', type=int, default=0)
    parser.add_argument('--split', type=float, default=0.1)
    args, _ = parser.parse_known_args()

    print('Received arguments {}'.format(args))

    # Define the paths
    input_data_base_path = '/opt/ml/processing/input'
    train_output_base_path = '/opt/ml/processing/train'
    test_output_base_path = '/opt/ml/processing/test'
    val_output_base_path = '/opt/ml/processing/val'
    report_output_base_path = '/opt/ml/processing/report'

    input_path_normal = os.path.join(input_data_base_path, PREFIX_NAME_NORMAL)
    input_path_anomalous = os.path.join(input_data_base_path, PREFIX_NAME_ANOMALOUS)

    # Resize the images in-place in the container image
    print('Resizing images...')
    resize_images(input_path_normal, IMAGE_WIDTH, IMAGE_HEIGHT)
    resize_images(input_path_anomalous, IMAGE_WIDTH, IMAGE_HEIGHT)

    # Augment images if needed
    # TODO: Fix hardcoded augment count, get from parameter
    augment_data(input_path_normal, int(args.augment_count))
    augment_data(input_path_anomalous, int(args.augment_count))

    # Create train test validation split
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(input_data_base_path, split=float(args.split))

    # Create list files for RecordIO transformation
    base_dir_recordio = './'

    with open(base_dir_recordio+'train.lst', 'w+') as f:
        for indx, s in enumerate(train_x):
            f.write(f'{indx}\t{train_y[indx]}\t{s}\n')

    with open(base_dir_recordio+'val.lst', 'w+') as f:
        for indx, s in enumerate(valid_x):
            f.write(f'{indx}\t{valid_y[indx]}\t{s}\n')

    with open(base_dir_recordio+'test.lst', 'w+') as f:
        for indx, s in enumerate(test_x):
            f.write(f'{indx}\t{test_y[indx]}\t{s}\n')
            
    # Run im2rec.py file to convert to RecordIO
    print('Running im2rec.py tool for recordio conversion')
    os.system('python3 ./im2rec.py train.lst ./')
    os.system('python3 ./im2rec.py val.lst ./')
    os.system('python3 ./im2rec.py test.lst ./')
    
    # Copy to the output paths
    shutil.copy('train.rec', os.path.join(train_output_base_path, 'train.rec'))
    shutil.copy('val.rec', os.path.join(val_output_base_path, 'val.rec'))
    shutil.copy('test.rec', os.path.join(test_output_base_path, 'test.rec'))

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

    