# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import threading
import json
import logging
import app.util as util

IOT_BASE_TOPIC = 'edge-manager-app'

class Logger(object):
    def __init__(self, device_name, iot_params):
        '''
            This class is responsible for sending application logs
            to the cloud via MQTT and IoT Topics
        '''
        self.device_name = device_name
        logging.info("Device Name: %s" % self.device_name)
        self.iot_params = iot_params

        self.__update_credentials()

        self.logs_buffer = []
        self.__log_lock = threading.Lock()

    def __update_credentials(self):
        '''
            Get new temp credentials
        '''
        logging.info("Getting the IoT Credentials")
        self.iot_data_client = util.get_client('iot-data', self.iot_params)

    def __run_logs_upload_job__(self):        
        '''
            Launch a thread that will read the logs buffer
            prepare a json document and send the logs
        '''
        self.cloud_log_sync_job = threading.Thread(target=self.__upload_logs__)
        self.cloud_log_sync_job.start()
        
    def __upload_logs__(self):
        '''
            Invoked by the thread to publish the latest logs
        '''
        self.__log_lock.acquire(True)
        f = json.dumps({'logs': self.logs_buffer})
        self.logs_buffer = [] # clean the buffer
        try:
            self.iot_data_client.publish( topic='%s/logs/%s' % (IOT_BASE_TOPIC, self.device_name), payload=f.encode('utf-8') )
        except Exception as e:
            logging.error(e)
            self.__update_credentials()
            self.iot_data_client.publish( topic='%s/logs/%s' % (IOT_BASE_TOPIC, self.device_name), payload=f.encode('utf-8') )

        logging.info("New log file uploaded. len: %d" % len(f))
        self.__log_lock.release()
 
    def publish_logs(self, data):
        '''
            Invoked by the application, it buffers the logs            
        '''
        buffer_len = 0
        if self.__log_lock.acquire(False):
            self.logs_buffer.append(data)
            buffer_len = len(self.logs_buffer)
            self.__log_lock.release()
        # else: job is running, discard the new data
        if buffer_len > 10:
            # run the sync job
            self.__run_logs_upload_job__()