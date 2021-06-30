# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import numpy as np
import boto3
import requests
import PIL
import io
import base64

def create_dataset(X, time_steps=1, step=1):
    '''
        Format a timeseries buffer into a multidimensional tensor
        required by the model
    '''
    Xs = []
    for i in range(0, len(X) - time_steps, step):
        v = X[i:(i + time_steps)]
        Xs.append(v)
    return np.array(Xs)

def get_aws_credentials(cred_endpoint, thing_name, cert_file, key_file, ca_file):
    '''
        Invoke SageMaker Edge Manager endpoint to exchange the certificates
        by temp credentials
    '''
    resp = requests.get(
        cred_endpoint,
        cert=(cert_file, key_file, ca_file),
    )
    if not resp:
        raise Exception('Error while getting the IoT credentials: ', resp)
    credentials = resp.json()
    return (credentials['credentials']['accessKeyId'],
        credentials['credentials']['secretAccessKey'],
        credentials['credentials']['sessionToken'])

def get_client(service_name, iot_params):
    '''
        Build a boto3 client of a given service
        It uses the temp credentials exchanged by the certificates
    '''
    access_key_id,secret_access_key,session_token = get_aws_credentials(
        iot_params['sagemaker_edge_provider_aws_iot_cred_endpoint'],
        iot_params['sagemaker_edge_core_device_name'],
        iot_params['sagemaker_edge_provider_aws_cert_file'],
        iot_params['sagemaker_edge_provider_aws_cert_pk_file'],
        iot_params['sagemaker_edge_provider_aws_ca_cert_file']
    )
    return boto3.client(
        service_name, iot_params['sagemaker_edge_core_region'],
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        aws_session_token=session_token
    )

def create_b64_img_from_mask(mask):
    """Creates binary stream from (1, SIZE, SIZE)-shaped binary mask"""
    img_size = mask.shape[1]
    mask_reshaped = np.reshape(mask, (img_size, img_size))
    img = PIL.Image.fromarray(np.uint8(mask_reshaped)*255)
    img_binary = io.BytesIO()
    img.save(img_binary, 'PNG')
    img_b64 = base64.b64encode(img_binary.getvalue())
    return img_b64