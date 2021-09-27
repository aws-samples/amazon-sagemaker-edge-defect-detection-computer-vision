# Python 3.9
import boto3
import os, sys

sm_client = boto3.client('sagemaker')

def handler(event, context):
    """Lambda Handler"""
    print(event)

    edge_packaging_job_name = event['startEdgePackageResult']['Payload']['edgePackagingJobName']
    resp = sm_client.describe_edge_packaging_job(EdgePackagingJobName=edge_packaging_job_name)

    return {
        'edgePackagingJobStatus': resp['EdgePackagingJobStatus'],
        'packagedModelArtifactLocation': resp.get('ModelArtifact', 'pending')   # Get attribute only when exists (i.e. job has finished)
    }