# Python 3.9
import boto3
import os, sys
import time

sm_client = boto3.client('sagemaker')

s3_bucket_name = os.environ['S3_BUCKET_NAME']
s3_model_artifact_prefix = os.environ['S3_MODEL_ARTIFACT_PREFIX']
model_name = os.environ['MODEL_NAME']

sagemaker_execution_role_arn = os.environ['SAGEMAKER_EXECUTION_ROLE_ARN']

def handler(event, context):
    """Lambda Handler"""
    print(event)

    edge_packaging_job_name = '%s-%d' % (model_name, int(time.time()*1000))
    compilation_job_name = event['startNeoCompilationResult']['Payload']['compilationJobName']
    model_version = str(int(event['initResult']['Payload']['modelPackageVersion']))


    # Start the edge packaging job
    resp = sm_client.create_edge_packaging_job(
        EdgePackagingJobName=edge_packaging_job_name,
        CompilationJobName=compilation_job_name,
        ModelName=model_name,
        ModelVersion=model_version,
        RoleArn=sagemaker_execution_role_arn,
        OutputConfig={
            'S3OutputLocation': f's3://{s3_bucket_name}/{s3_model_artifact_prefix}/edge-packaging-output'
        }
    )


    return {
        'edgePackagingJobName': edge_packaging_job_name
    }