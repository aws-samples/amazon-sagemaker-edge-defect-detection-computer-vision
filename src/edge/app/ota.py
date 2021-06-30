# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Source code partially modified from: https://github.com/aws-samples/amazon-sagemaker-edge-manager-demo/blob/main/04_EdgeApplication/turbine/ota.py

import ssl
import paho.mqtt.client as mqtt
import logging
import json
import os
import io
import time
import requests
import boto3
import tarfile
import glob
import threading
import app

class OTAModelUpdate(object):
    def __init__(self, device_name, iot_params, mqtt_host, mqtt_port, update_callback, model_path, models_supported):
        '''
            This class is responsible for listening to IoT topics and receiving
            a Json document with the metadata of a new model. This module also
            downloads the SageMaker Edge Manager deployment package, unpacks it to
            a local dir and also controls versioning.
        '''
        if model_path is None or update_callback is None:
            raise Exception("You need to inform a model_path and an update_callback methods")
        self.device_name = device_name
        self.model_path = model_path
        self.update_callback = update_callback
        self.iot_params = iot_params
        self.models_supported = models_supported

        ## initialize an mqtt client
        self.mqttc = mqtt.Client()
        self.mqttc.tls_set(
            iot_params['sagemaker_edge_provider_aws_ca_cert_file'],
            certfile=iot_params['sagemaker_edge_provider_aws_cert_file'],
            keyfile=iot_params['sagemaker_edge_provider_aws_cert_pk_file'],
            cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2, ciphers=None
        )
        self.mqttc.enable_logger(logger=logging)
        self.mqttc.on_message = self.__on_message__
        self.mqttc.on_connect = self.__on_connect__
        self.mqttc.on_disconnect = self.__on_disconnect__
        self.connected = False

        self.processing_lock = threading.Lock()
        self.processed_jobs = []

        # start the mqtt client
        self.mqttc.connect(mqtt_host, mqtt_port, 45)
        self.mqttc.loop_start()

    def model_update_check(self):
        '''
            Check manually if there is a new model available
        '''
        if self.connected:
            self.mqttc.publish('$aws/things/%s/jobs/get' % self.device_name)

    def __on_message__(self, client, userdata, message):
        '''
            This callback is invoked by MQTTC each time a new message is published
            to one of the subscribed topics
        '''
        logging.debug("New message. Topic: %s; Message: %s;" % (message.topic, message.payload))

        if message.topic.endswith('notify'):
            self.model_update_check()

        elif message.topic.endswith('accepted'):
            resp = json.loads(message.payload)
            logging.debug(resp)
            if resp.get('queuedJobs') is not None: # request to list jobs
                # get the description of each queued job
                for j in resp['queuedJobs']:
                    ## get the job description
                    self.mqttc.publish('$aws/things/%s/jobs/%s/get' % ( self.device_name, j['jobId'] ) )
                    break
            elif resp.get('inProgressJobs') is not None: # request to list jobs
                # get the description of each queued job
                for j in resp['inProgressJobs']:
                    ## get the job description
                    self.mqttc.publish('$aws/things/%s/jobs/%s/get' % ( self.device_name, j['jobId'] ) )
                    break
            elif resp.get('execution') is not None: # request to get job description
                # check if this is a job description message
                job_meta = resp.get('execution')

                # we have the job metadata, let's process it
                self.__update_job_status__(job_meta['jobId'], 'IN_PROGRESS', 'Trying to get/load the model')
                self.__process_job__(job_meta['jobId'], job_meta['jobDocument'])
            else:
                logging.debug('Other message: ', resp)

    def __on_connect__(self, client, userdata, flags, rc):
        '''
            This callback is invoked just after MQTTC managed to connect
            to the MQTT endpoint
        '''
        self.connected = True
        logging.info("OTA Model Manager Connected to the MQTT endpoint!")
        self.mqttc.subscribe('$aws/things/%s/jobs/notify' % self.device_name)
        self.mqttc.subscribe('$aws/things/%s/jobs/accepted' % self.device_name)
        self.mqttc.subscribe('$aws/things/%s/jobs/rejected' % self.device_name)
        time.sleep(1)
        self.model_update_check()

    def __on_disconnect__(self, client, userdata, flags):
        '''
            This callback is invoked when MQTTC disconnected from the MQTT endpoint
        '''
        self.connected = False
        logging.info("OTA Model Manager Disconnected!")

    def __del__(self):
        '''
            Object destructor
        '''
        logging.info("OTA Model Manager Deleting this object")
        self.mqttc.loop_stop()
        self.mqttc.disconnect()

    def __update_job_status__(self, job_id, status, details):
        '''
            After receiving a new signal that there is a model to be deployed
            Update the IoT Job to inform the user the current status of this
            process
        '''
        payload = json.dumps({
            "status": status,
            "statusDetails": {"info": details },
            "includeJobExecutionState": False,
            "includeJobDocument": False,
            "stepTimeoutInMinutes": 2,
        })
        logging.info("Updating IoT job status: %s" % details)
        self.mqttc.publish('$aws/things/%s/jobs/%s/update' % ( self.device_name, job_id), payload)


    def __process_job__(self, job_id, msg):
        '''
            This method is responsible for:
                1. validate the new model version
                2. download the model package
                3. unpack it to a local dir
                4. notify the main application
        '''
        self.processing_lock.acquire()
        if job_id in self.processed_jobs:
            self.processing_lock.release()
            return
        self.processed_jobs.append(job_id)
        try:
            if msg.get('type') == 'new_model':
                model_version = msg['model_version']
                model_name = msg['model_name']

                # Check if the application supports the model with the name incoming
                if model_name not in self.models_supported:
                    msg = 'New model %s from incoming deployment is not in list of supported models. Skipping deployment.' % model_name
                    logging.info(msg)
                    self.__update_job_status__(job_id, 'FAILED', msg)
                    self.processing_lock.release()
                    return

                logging.info("Downloading new model package")
                s3_client = app.get_client('s3', self.iot_params)

                package = io.BytesIO(s3_client.get_object(
                    Bucket=msg['model_package_bucket'],
                    Key=msg['model_package_key'])['Body'].read()
                )
                logging.info("Unpacking model package")
                with tarfile.open(fileobj=package) as p:
                    p.extractall(os.path.join(self.model_path, msg['model_name'], msg['model_version']))

                self.__update_job_status__(job_id, 'SUCCEEDED', 'Model deployed')
                self.update_callback(model_name, model_version)
            else:
                logging.info("Model '%s' version '%f' is the current one or it is obsolete" % (self.model_metadata['model_name'], self.model_metadata['model_version']))
        except Exception as e:
            self.__update_job_status__(job_id, 'FAILED', str(e))
            logging.error(e)

        self.processing_lock.release()