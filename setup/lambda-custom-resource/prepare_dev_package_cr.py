# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Lambda-backed custom resource function to create the SageMaker Edge Manager device package.
Support SageMaker Edge Agent Version: 
"""
import json
import os
import logging
import stat
from botocore.parsers import LOG
import urllib3
import boto3
import tarfile
import io
from botocore.exceptions import ClientError

http = urllib3.PoolManager()

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

BUCKET_NAME = os.environ['BUCKET_NAME']
PROJECT_NAME = os.environ['PROJECT_NAME']
AWS_REGION = os.environ['AWS_REGION']

LOCAL_DIR_PREFIX = '/tmp/'  # Needed for running in AWS Lambda

iot_client = boto3.client('iot')
sm_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')

# Global variables
# This information needs to match with what was defined in the CloudFormation template
sm_edge_device_name = 'edge-device-defect-detection-%s' % PROJECT_NAME
iot_policy_name = 'defect-detection-policy-%s' % PROJECT_NAME
iot_thing_name = 'edge-device-%s' % PROJECT_NAME
iot_thing_group_name='defect-detection-%s-group' % PROJECT_NAME
sm_em_fleet_name = 'defect-detection-%s' % PROJECT_NAME
role_alias = 'SageMakerEdge-%s' % sm_em_fleet_name


def cfn_cleanup():
    """Clean up resources created in the custom resources"""

    LOGGER.info('Deleting role alias if exists')
    try:
        iot_client.delete_role_alias(roleAlias=role_alias)
    except:
        LOGGER.info('Role alias deletion failed, continuing anyways')
    
    LOGGER.info('Deregistering device from edge fleet if exists')
    try:
        sm_client.deregister_devices(
            DeviceFleetName=sm_em_fleet_name,
            DeviceNames=[sm_edge_device_name]
        )
    except:
        LOGGER.info('Device deregistration failed, continuing anyways')

    LOGGER.info('Detaching certificates')
    try:
        cert_arn = iot_client.list_thing_principals(thingName=iot_thing_name)['principals'][0]
        cert_id = cert_arn.split('/')[-1]
        iot_client.detach_policy(policyName=iot_policy_name, target=cert_arn)
        iot_client.detach_thing_principal(thingName=iot_thing_name, principal=cert_arn)
        iot_client.update_certificate(certificateId=cert_id, newStatus='INACTIVE')
        iot_client.delete_certificate(certificateId=cert_id, forceDelete=True)
        iot_client.delete_thing_group(thingGroupName=iot_thing_group_name)
    except:
        LOGGER.info('Detaching certificates failed, continuing anyways')




def lambda_handler(event, context):
    '''Handle Lambda event from AWS'''

    try:
        LOGGER.info('REQUEST RECEIVED:\n %s', event)
        LOGGER.info('REQUEST RECEIVED:\n %s', context)
        if event['RequestType'] == 'Create':
            LOGGER.info('CREATE!')

            LOGGER.info('Starting device packaging...')
            try:
                prepare_device_package(event, context)
                send_response(event, context, "SUCCESS",
                          {"Message": "Resource creation successful!"})
            except Exception as e:
                send_response(event, context, "FAILED", {"Message": "Resource creation failed during device packaging!", "Error": str(e)})
        elif event['RequestType'] == 'Update':
            LOGGER.info('UPDATE!')
            send_response(event, context, "SUCCESS",
                          {"Message": "Resource update successful!"})
        elif event['RequestType'] == 'Delete':
            LOGGER.info('DELETE!')
            # Start cleanup method
            cfn_cleanup()
            send_response(event, context, "SUCCESS",
                          {"Message": "Resource deletion successful!"})
        else:
            LOGGER.info('FAILED!')
            send_response(event, context, "FAILED",
                          {"Message": "Unexpected event received from CloudFormation"})
    except: #pylint: disable=W0702
        LOGGER.info('FAILED!')
        send_response(event, context, "FAILED", {
            "Message": "Exception during processing"})


def send_response(event, context, response_status, response_data):
    '''Send a resource manipulation status response to CloudFormation'''
    response_body = json.dumps({
        "Status": response_status,
        "Reason": "See the details in CloudWatch Log Stream: " + context.log_stream_name,
        "PhysicalResourceId": context.log_stream_name,
        "StackId": event['StackId'],
        "RequestId": event['RequestId'],
        "LogicalResourceId": event['LogicalResourceId'],
        "Data": response_data
    })

    print("Response body:")
    print(response_body)

    response_url = event['ResponseURL']

    headers = {
        'content-type' : '',
        'content-length' : str(len(response_body))
    }

    try:
        response = http.request('PUT', response_url, headers=headers, body=response_body)
        print("Status code:", response.status)

    except Exception as e:

        print("send(..) failed executing http.request(..):", e)


def setup_agent(thing_group_name, thing_group_arn):
    """Creates configuration file and sets up SageMaker Edge Agent for deployment
    onto a Amazon S3 bucket. Registers a device with a device fleet, creates IoT
    certificates and attaches them to the previously created IoT thing. Saves 
    certificates onto local disk to make it ready for uploading to S3.

    Args:
        thing_group_name (string): a name for the IoT thing group
        thing_group_arn (string): the ARN of the IoT thing group
    """

    local_base_path = LOCAL_DIR_PREFIX + "agent/certificates/iot/edge_device_cert_%s.pem"
    relative_base_path = "agent/certificates/iot/edge_device_cert_%s.pem"
    thing_arn_template = thing_group_arn.replace('thinggroup', 'thing').replace(thing_group_name, '%s')
    cred_host = iot_client.describe_endpoint(endpointType='iot:CredentialProvider')['endpointAddress']

    # Check length of device name string
    if len(sm_edge_device_name) > 64:
        LOGGER.error("Device name for edge device is too long. Needs to be <64 characters.")
        raise ClientError('Device name for edge device is longer than 64 characters. Please choose a shorter value for ProjectName.')

    # register the device in the fleet    
    # the device name needs to have 36 chars
    dev = [{'DeviceName': sm_edge_device_name, 'IotThingName': iot_thing_name}]    
    try:        
        sm_client.describe_device(DeviceFleetName=sm_em_fleet_name, DeviceName=sm_edge_device_name)
        LOGGER.info("Device was already registered on SageMaker Edge Manager")
    except ClientError as e:
        if e.response['Error']['Code'] != 'ValidationException': raise e
        LOGGER.info("Registering a new device %s on fleet %s" % (sm_edge_device_name, sm_em_fleet_name))
        sm_client.register_devices(DeviceFleetName=sm_em_fleet_name, Devices=dev)
        iot_client.add_thing_to_thing_group(
            thingGroupName=thing_group_name,
            thingGroupArn=thing_group_arn,
            thingName=iot_thing_name,
            thingArn=thing_arn_template % iot_thing_name
        )        
    
    # if you reach this point you need to create new certificates
    # generate the certificates    
    cert = local_base_path % ('cert')
    key = local_base_path % ('pub')
    pub = local_base_path % ('key')

    # Relative paths needed for setting path in config file
    cert_relative = relative_base_path % ('cert')
    key_relative = relative_base_path % ('pub')
    pub_relative = relative_base_path % ('key')
    
    cert_meta=iot_client.create_keys_and_certificate(setAsActive=True)
    cert_arn = cert_meta['certificateArn']
    with open(cert, 'w') as c: c.write(cert_meta['certificatePem'])
    with open(key,  'w') as c: c.write(cert_meta['keyPair']['PrivateKey'])
    with open(pub,  'w') as c: c.write(cert_meta['keyPair']['PublicKey'])
        
    # attach the certificates to the policy and to the thing
    iot_client.attach_policy(policyName=iot_policy_name, target=cert_arn)
    iot_client.attach_thing_principal(thingName=iot_thing_name, principal=cert_arn)        
    
    LOGGER.info("Creating agent config JSON file")

    # Please note that the $WORKDIR variables need to be replaced by the absolute path of the working directory of your project.
    # If you follow the guide, the install script will automatically replace those.
    agent_params = {
        "sagemaker_edge_core_device_name": sm_edge_device_name,
        "sagemaker_edge_core_device_fleet_name": sm_em_fleet_name,
        "sagemaker_edge_core_region": AWS_REGION,
        "sagemaker_edge_provider_provider": "Aws",
        "sagemaker_edge_provider_provider_path" : "$WORKDIR/agent/lib/libprovider_aws.so",
        "sagemaker_edge_core_root_certs_path": "$WORKDIR/agent/certificates/root",
        "sagemaker_edge_provider_aws_ca_cert_file": "$WORKDIR/agent/certificates/iot/AmazonRootCA1.pem",
        "sagemaker_edge_provider_aws_cert_file": "$WORKDIR/%s" % cert_relative,
        "sagemaker_edge_provider_aws_cert_pk_file": "$WORKDIR/%s" % key_relative,
        "sagemaker_edge_provider_aws_iot_cred_endpoint": "https://%s/role-aliases/%s/credentials" % (cred_host,role_alias),
        "sagemaker_edge_core_capture_data_destination": "Cloud",
        "sagemaker_edge_provider_s3_bucket_name": BUCKET_NAME,
        "sagemaker_edge_core_folder_prefix": "edge-agent-inference-data-capture",
        "sagemaker_edge_core_capture_data_buffer_size": 30,
        "sagemaker_edge_core_capture_data_batch_size": 10,
        "sagemaker_edge_core_capture_data_push_period_seconds": 10,
        "sagemaker_edge_core_capture_data_base64_embed_limit": 2,
        "sagemaker_edge_log_verbose": False
    }
    with open(LOCAL_DIR_PREFIX + 'agent/conf/config_edge_device.json', 'w') as conf:
        conf.write(json.dumps(agent_params, indent=4))


def prepare_device_package(event, context):
    """Prepares the edge device package in a lambda function and uploads it to the S3 bucket"""

    # create a new thing group
    thing_group_arn = None
    agent_pkg_bucket = 'sagemaker-edge-release-store-us-west-2-linux-x64'
    agent_config_package_prefix = 'edge-device-configuration/agent/config.tgz'

    # check if edge agent package has already been built
    try:
        s3_client.download_file(Bucket=BUCKET_NAME, Key=agent_config_package_prefix, Filename='/tmp/dump')
        LOGGER.info('The agent configuration package was already built! Skipping...')
        quit()
    except ClientError as e:
        pass
    
    # Create a new thing group if not found yet
    try:
        thing_group_arn = iot_client.describe_thing_group(thingGroupName=iot_thing_group_name)['thingGroupArn']
        LOGGER.info("Thing group found")
    except iot_client.exceptions.ResourceNotFoundException as e:
        LOGGER.info("Creating a new thing group")
        thing_group_arn = iot_client.create_thing_group(thingGroupName=iot_thing_group_name)['thingGroupArn']

    LOGGER.info("Creating the directory structure for the agent")
    # create a structure for the agent files
    os.makedirs(LOCAL_DIR_PREFIX + 'agent/certificates/root', exist_ok=True)
    os.makedirs(LOCAL_DIR_PREFIX + 'agent/certificates/iot', exist_ok=True)
    os.makedirs(LOCAL_DIR_PREFIX + 'agent/logs', exist_ok=True)
    os.makedirs(LOCAL_DIR_PREFIX + 'agent/model', exist_ok=True)
    os.makedirs(LOCAL_DIR_PREFIX + 'agent/conf', exist_ok=True)
    
    LOGGER.info("Downloading root certificate and agent binary")
    # then get some root certificates
    resp = http.request('GET', 'https://www.amazontrust.com/repository/AmazonRootCA1.pem')
    with open(LOCAL_DIR_PREFIX + 'agent/certificates/iot/AmazonRootCA1.pem', 'w') as c:
        c.write(resp.data.decode('utf-8'))
    
    # this certificate validates the edge manage package
    s3_client.download_file(
        Bucket=agent_pkg_bucket, 
        Key='Certificates/%s/%s.pem' % (AWS_REGION, AWS_REGION), 
        Filename=LOCAL_DIR_PREFIX + 'agent/certificates/root/%s.pem' % AWS_REGION
    )

    LOGGER.info("Adjusting file permissions of pem files")
    # adjust the permissions of the files
    os.chmod(LOCAL_DIR_PREFIX + 'agent/certificates/iot/AmazonRootCA1.pem', stat.S_IRUSR|stat.S_IRGRP)
    os.chmod(LOCAL_DIR_PREFIX + 'agent/certificates/root/%s.pem' % AWS_REGION, stat.S_IRUSR|stat.S_IRGRP)
    
    LOGGER.info("Processing the agent...")
    setup_agent(iot_thing_group_name, thing_group_arn )
    
    LOGGER.info("Creating the final package...")
    with io.BytesIO() as f:
        with tarfile.open(fileobj=f, mode='w:gz') as tar:
            tar.add(LOCAL_DIR_PREFIX + 'agent', 'agent', recursive=True)
        f.seek(0)
        LOGGER.info("Uploading to S3")
        s3_client.upload_fileobj(f, Bucket=BUCKET_NAME, Key=agent_config_package_prefix)
    LOGGER.info("Done!")