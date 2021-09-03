# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import argparse
import numpy as np
import os
from glob import glob
import cv2
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMAGE_WIDTH=224
IMAGE_HEIGHT=224

def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    
    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT))
    x = x/255.0
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([IMAGE_WIDTH, IMAGE_HEIGHT, 3])
    y.set_shape([IMAGE_WIDTH, IMAGE_HEIGHT, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


def model():
    inputs = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), name="input_image")
    
    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    f = [16, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    
    model = Model(inputs, x)
    return model


def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def get_train_data(train_files_path,validation_files_path):
    
    train_x = sorted(glob(os.path.join(train_files_path, "images/*")))
    train_y = sorted(glob(os.path.join(train_files_path, "masks/*")))
    
    valid_x = sorted(glob(os.path.join(validation_files_path, "images/*")))
    valid_y = sorted(glob(os.path.join(validation_files_path, "masks/*")))
    
    
    
    return train_x,train_y,valid_x,valid_y


if __name__ == "__main__":
        
    args, _ = parse_args()
    EPOCHS = args.epochs
    BATCH = args.batch_size
    LR = args.learning_rate
    
    train_x,train_y,valid_x,valid_y = get_train_data(args.train,args.validation)
    train_dataset = tf_dataset(train_x, train_y, batch=BATCH)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH)
    print(train_dataset)
    
    
    device = '/cpu:0' 
    print(device)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))

    with tf.device(device):
        
        model = model()
        opt = tf.keras.optimizers.Nadam(LR)
        metrics = [dice_coef, Recall(), Precision()]
        model.compile(loss=dice_loss, optimizer=opt, metrics=metrics)
        
        callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
        ]
        
        train_steps = len(train_x)//BATCH
        valid_steps = len(valid_x)//BATCH

        if len(train_x) % BATCH != 0:
            train_steps += 1
        if len(valid_x) % BATCH != 0:
            valid_steps += 1
        model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
        )
        # evaluate on train set
        scores = model.evaluate(train_dataset,steps=train_steps)
        print("\ntrain bce :", scores)
                
        # evaluate on val set
        scores = model.evaluate(valid_dataset,steps=valid_steps)
        print("\nval bce :", scores)
        
        # save model
        #model.save(args.model_dir + '/1')
        
        #Save as .h5, neo supports only .h5 format for keras , set 'include_optimizer=False' to remove operators that do not   compile
        filepath=args.model_dir + '/unet_mobilenetv2.h5'
        tf.keras.models.save_model(
            model, filepath, overwrite=True, include_optimizer=False, save_format='h5'#,
    #signatures=None, options=None, save_traces=True
        )
    