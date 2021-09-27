# Python 3.9
import boto3
import uuid
import json
import os, sys

iot_client = boto3.client('iot')

iot_job_target_arn = os.environ['IOT_JOB_TARGET_ARN']
model_name = os.environ['MODEL_NAME']


def split_s3_path(s3_path):
    path_parts=s3_path.replace("s3://","").split("/")
    bucket=path_parts.pop(0)
    key="/".join(path_parts)
    return bucket, key

def handler(event, context):
    """Lambda Handler"""
    model_version = event['initResult']['Payload']['modelPackageVersion']
    s3_packaged_model_artifact_location = event['pollEdgePackageResult']['Payload']['packagedModelArtifactLocation']

    # Extract bucket and key from the S3 path
    model_bucket, model_key = split_s3_path(s3_packaged_model_artifact_location)

    # This needs to coincide with the logic on the edge device which will handle this incoming IoT job
    resp = iot_client.create_job(
        jobId=str(uuid.uuid4()),
        targets=[
            iot_job_target_arn
        ],
        document=json.dumps({
            'type': 'new_model',
            'model_version': model_version,
            'model_name': model_name,
            'model_package_bucket': model_bucket,
            'model_package_key': model_key
        }),
        targetSelection='SNAPSHOT'
    )


    return {}