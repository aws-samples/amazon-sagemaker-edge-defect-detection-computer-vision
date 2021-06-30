# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# From: https://github.com/aws-samples/amazon-sagemaker-edge-manager-demo/blob/main/04_EdgeApplication/turbine/edgeagentclient.py

import grpc
import logging
import app.agent_pb2 as agent
import app.agent_pb2_grpc as agent_grpc
import struct
import numpy as np
import uuid

class EdgeAgentClient(object):
    """ Helper class that uses the Edge Agent stubs to
        communicate with the SageMaker Edge Agent through unix socket.

        To generate the stubs you need to use protoc. First install/update:
        pip3 install -U grpcio-tools grpcio protobuf
        then generate the code using the provided agent.proto file

        python3 -m grpc_tools.protoc \
            --proto_path=$PWD/agent/docs/api --python_out=./app --grpc_python_out=./app $PWD/agent/docs/api/agent.proto

    """
    def __init__(self, channel_path):
        # connect to the agent and list the models
        self.channel = grpc.insecure_channel('unix://%s' % channel_path )
        self.agent = agent_grpc.AgentStub(self.channel)
        self.model_map = {}

    def __update_models_list__(self):
        models_list = self.agent.ListModels(agent.ListModelsRequest())
        self.model_map = {m.name:{'in': m.input_tensor_metadatas, 'out': m.output_tensor_metadatas} for m in models_list.models}
        return self.model_map

    def capture_data(self, model_name, input_data, output_data):
        """The CaptureData request to the edge agent"""
        try:
            logging.info('Capturing data for model %s' % model_name)
            req = agent.CaptureDataRequest()
            req.model_name = model_name
            req.capture_id = str(uuid.uuid4())
            req.input_tensors.append( self.create_tensor(input_data, 'input'))
            req.output_tensors.append( self.create_tensor(output_data, 'output'))
            resp = self.agent.CaptureData(req)
        except Exception as e:
            logging.error('Error in capture_data: %s' % e)

    def create_tensor(self, x, tensor_name):
        """Creates a Edge agent tensor from a numpy float32 array"""
        if (x.dtype != np.float32):
            raise Exception( "It only supports numpy float32 arrays for this tensor but type for tensor %s was %s" % (tensor_name, x.dtype))
        tensor = agent.Tensor()
        tensor.tensor_metadata.name = tensor_name.encode()
        tensor.tensor_metadata.data_type = agent.FLOAT32
        for s in x.shape: tensor.tensor_metadata.shape.append(s)
        tensor.byte_data = x.tobytes()
        return tensor

    def predict(self, model_name, x, shm=False):
        """
        Invokes the model and get the predictions
        """
        try:
            if self.model_map.get(model_name) is None:
                raise Exception('Model %s not loaded' % model_name)
            # Create a request
            req = agent.PredictRequest()
            req.name = model_name
            # Then load the data into a temp Tensor
            tensor = agent.Tensor()
            meta = self.model_map[model_name]['in'][0]
            tensor.tensor_metadata.name = meta.name
            tensor.tensor_metadata.data_type = meta.data_type
            for s in meta.shape: tensor.tensor_metadata.shape.append(s)

            if shm:
                tensor.shared_memory_handle.offset = 0
                tensor.shared_memory_handle.segment_id = x
            else:
                tensor.byte_data = x.astype(np.float32).tobytes()

            req.tensors.append(tensor)

            # Invoke the model
            resp = self.agent.Predict(req)

            # Parse the output
            meta = self.model_map[model_name]['out'][0]
            tensor = resp.tensors[0]
            data = np.frombuffer(tensor.byte_data, dtype=np.float32)
            return data.reshape(tensor.tensor_metadata.shape)
        except Exception as e:
            logging.error('Error in predict: %s' % e)
            return None

    def is_model_loaded(self, model_name):
        return self.model_map.get(model_name) is not None

    def load_model(self, model_name, model_path):
        """ Load a new model into the Edge Agent if not loaded yet"""
        try:
            if self.is_model_loaded(model_name):
                logging.info( "Model %s was already loaded" % model_name )
                return self.model_map
            req = agent.LoadModelRequest()
            req.url = model_path
            req.name = model_name
            resp = self.agent.LoadModel(req)

            return self.__update_models_list__()
        except Exception as e:
            logging.error('Error in load_model: %s' % e)
            return None

    def unload_model(self, model_name):
        """ UnLoad model from the Edge Agent"""
        try:
            if not self.is_model_loaded(model_name):
                logging.info( "Model %s was not loaded" % model_name )
                return self.model_map

            req = agent.UnLoadModelRequest()
            req.name = model_name
            resp = self.agent.UnLoadModel(req)

            return self.__update_models_list__()
        except Exception as e:
            logging.error('Error in unload_model: %s' % e)
            return None