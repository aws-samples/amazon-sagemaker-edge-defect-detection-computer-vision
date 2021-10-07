# Python 3.9
import boto3
import time
import os
import json

sm_client = boto3.client('sagemaker')

s3_bucket_name = os.environ['S3_BUCKET_NAME']
s3_model_artifact_prefix = os.environ['S3_MODEL_ARTIFACT_PREFIX']
model_name = os.environ['MODEL_NAME']

compilation_job_data_input_config = os.environ['COMPILATION_JOB_DATA_INPUT_CONFIG']
compilation_job_framework = os.environ['COMPILATION_FRAMEWORK']
compilation_job_target_platform = json.loads(os.environ['COMPILATION_TARGET_PLATFORM'])

sagemaker_execution_role_arn = os.environ['SAGEMAKER_EXECUTION_ROLE_ARN']

def handler(event, context):
    """Lambda Handler"""

    s3_model_artifact_location = event['initResult']['Payload']['modelArtifactLocation']
    compilation_job_name = '%s-%d' % (model_name, int(time.time()*1000))

    resp = sm_client.create_compilation_job(
        CompilationJobName=compilation_job_name,
        RoleArn=sagemaker_execution_role_arn,
        InputConfig={
            'S3Uri': s3_model_artifact_location,
            'DataInputConfig': compilation_job_data_input_config,
            'Framework': compilation_job_framework
        },
        OutputConfig={
            'S3OutputLocation': f's3://{s3_bucket_name}/{s3_model_artifact_prefix}/compilation-output',
            'TargetPlatform': compilation_job_target_platform
        },
        StoppingCondition={ 'MaxRuntimeInSeconds': 900 }
    )


    return {
        'compilationJobArn': resp['CompilationJobArn'],
        'compilationJobName': compilation_job_name
    }