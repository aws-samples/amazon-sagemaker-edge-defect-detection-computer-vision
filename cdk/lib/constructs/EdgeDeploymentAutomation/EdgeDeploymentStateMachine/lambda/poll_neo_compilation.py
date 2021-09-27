# Python 3.9
import boto3
import os, sys

sm_client = boto3.client('sagemaker')

def handler(event, context):
    """Lambda Handler"""
    
    compilation_job_name = event['startNeoCompilationResult']['Payload']['compilationJobName']
    resp = sm_client.describe_compilation_job(CompilationJobName=compilation_job_name)

    return {
        'compilationJobStatus': resp['CompilationJobStatus']
    }
