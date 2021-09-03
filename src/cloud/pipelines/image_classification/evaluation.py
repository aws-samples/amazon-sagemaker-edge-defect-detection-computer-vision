# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import sys
import os
import subprocess

# Install packages previous to executing the rest of the script. You can also build your own custom container
#   with your individal dependecies if needed
subprocess.check_call([sys.executable, "-m", "pip", "install", "mxnet", "opencv-python"])
os.system("apt-get update")
os.system("apt-get install ffmpeg libsm6 libxext6  -y")

import argparse
import json
import warnings
import logging
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
import tarfile
from PIL import Image
from glob import glob
import re

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, gluon
from mxnet.gluon.data.vision import transforms
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Constants

# The images size used

CLASS_LABELS = ['good', 'bad']

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-width', type=int, default=224)
    parser.add_argument('--image-height', type=int, default=224)
    args, _ = parser.parse_known_args()

    logger.info('Received arguments {}'.format(args))

    # Define the paths
    test_data_base_path = '/opt/ml/processing/test'
    model_data_base_path = '/opt/ml/processing/model'
    report_output_base_path = '/opt/ml/processing/report'
    
    IMAGE_WIDTH = int(args.image_width)
    IMAGE_HEIGHT = int(args.image_height)

    # Unzipping the model
    model_filename = 'model.tar.gz'
    model_path = os.path.join(model_data_base_path, model_filename)
    model_path_extracted = './model/'

    with tarfile.open(model_path) as tar:
        tar.extractall(path=model_path_extracted)

    # Get the files needed for loading, parse some strings
    symbol_file = glob(os.path.join(model_path_extracted, '*symbol.json'))[0]
    params_file = glob(os.path.join(model_path_extracted, '*.params'))[0]
    
    logger.info('Symbol file: %s' % symbol_file)
    logger.info('Params file: %s' % params_file)
    
    symbol_filename = os.path.basename(symbol_file)
    params_filename = os.path.basename(params_file)

    # Extract name and epoch needed for loading
    model_name = re.search(r".+(?=-symbol\.json)", symbol_filename).group(0)
    epoch = int(re.search(r"[0-9]+(?=\.params)", params_filename).group(0))

    # Loading model
    logger.info('Loading model from artifacts...')
    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(model_path_extracted, model_name), epoch)
    model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=['data'])
    model.bind(for_training=False, data_shapes=[('data', (1,3,IMAGE_WIDTH,IMAGE_HEIGHT))], 
            label_shapes=model._label_shapes)
    model.set_params(arg_params, aux_params, allow_missing=True)

    # Load test data into record iterator (batch size 1)
    test_data = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(test_data_base_path, 'test.rec'),
        data_shape  = (3, IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size  = 1,
        shuffle     = True
    )

    # Lists for the predicted and true labels
    y_true = []
    y_pred = []

    # For each batch (size=1) predict the class
    # TODO: make batch prediction work
    for batch in test_data:
        res = model.predict(eval_data=batch.data[0])
        pred_class = int(np.argmax(res[0]).asnumpy()[0])
        y_pred.append(pred_class)
        y_true.append(int(batch.label[0].asnumpy()))

    clf_report = classification_report(y_true, y_pred, target_names=CLASS_LABELS, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)

    # Save the preprocessing report to make information available to downstream steps
    evaluation_report = {
        'multiclass_classification_metrics': {
            'accuracy': {
                'value': accuracy,
                'standard_deviation': 'NaN'
            },
            'weighted_recall': {
                'value': clf_report['weighted avg']['recall'],
                'standard_deviation': 'NaN'
            },
            'weighted_precision': {
                'value': clf_report['weighted avg']['precision'],
                'standard_deviation': 'NaN'
            },
            'weighted_f1': {
                'value': clf_report['weighted avg']['f1-score'],
                'standard_deviation': 'NaN'
            }
        },
        'classification_report': clf_report
    }
    print('Evaluation report:', evaluation_report)
    report_output_path = os.path.join(report_output_base_path, 'evaluation_report.json')
    with open(report_output_path, "w") as f:
            f.write(json.dumps(evaluation_report))
