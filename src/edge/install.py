# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import boto3
import os
import tarfile
import stat
import io
import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

s3_client = boto3.client('s3')

# Default bucket for downloading the SM Edge Agent. Please note that your device needs access to this bucket through IAM
agent_config_package_prefix = 'edge-device-configuration/agent/config.tgz'
agent_version = '1.20210512.96da6cc'
agent_pkg_bucket = 'sagemaker-edge-release-store-us-west-2-linux-x64'

def download_config(bucket_name):
    # Check if agent is installed and configured already
    if not os.path.isdir('agent'):
        logger.info('No SM Edge Agent directory found. Proceding with download of configuration package...')

        # Get the configuration package with certificates and config files
        with io.BytesIO() as file:
            s3_client.download_fileobj(bucket_name, agent_config_package_prefix, file)
            file.seek(0)
            # Extract the files
            tar = tarfile.open(fileobj=file)
            tar.extractall('.')
            tar.close()

        # Download and install SageMaker Edge Manager
        agent_pkg_key = 'Releases/%s/%s.tgz' % (agent_version, agent_version)
        # get the agent package
        logger.info('Downloading and installing edge agent binary...')

        with io.BytesIO() as file:
            s3_client.download_fileobj(agent_pkg_bucket, agent_pkg_key, file)
            file.seek(0)
            # Extract the files
            tar = tarfile.open(fileobj=file)
            tar.extractall('agent')
            tar.close()
            # Adjust the permissions
            os.chmod('agent/bin/sagemaker_edge_agent_binary', stat.S_IXUSR|stat.S_IWUSR|stat.S_IXGRP|stat.S_IWGRP)

    # Finally, create SM Edge Agent client stubs, using protobuffer compiler
    logger.info('Creating protobuf agent stubs...')
    os.system('mkdir -p app/')
    os.system('python3 -m grpc_tools.protoc --proto_path=agent/docs/api --python_out=app/ --grpc_python_out=app/ agent/docs/api/agent.proto')

if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--project-name', type=str, required=True)
    parser.add_argument('--account-id', type=str, required=True)
    parser.add_argument('--region', type=str, required=True)
    args, _ = parser.parse_known_args()

    logger.info('Preparing device...')

    # Infer bucket name from project name and AWS Account ID as created in the CDK app
    bucket_name = '%s-%s-%s' % (args.project_name, args.region, args.account_id)

    # Run the installation script
    download_config(bucket_name)

    logger.info('Done!')

