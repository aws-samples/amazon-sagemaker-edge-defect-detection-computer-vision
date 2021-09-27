# Python 3.9
import boto3
import os, sys

sm_client = boto3.client('sagemaker')

def get_latest_approved_s3_model_location(model_package_group_name):
    """Returns the model location of the latest approved model version in a group"""
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus='Approved'
    )
    latest_version = max(response['ModelPackageSummaryList'], key=lambda x:x['ModelPackageVersion'])
    model_artifact_location = sm_client.describe_model_package(ModelPackageName=latest_version['ModelPackageArn'])['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    return model_artifact_location

def get_latest_approved_model_version(model_package_group_name):
    """Returns the model version of the latest approved model version in a group"""
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus='Approved'
    )
    latest_version = max(response['ModelPackageSummaryList'], key=lambda x:x['ModelPackageVersion'])
    return latest_version['ModelPackageVersion']


def handler(event, context):
    """Lambda Handler"""
    print(event)

    # Parse some data from EventBridge event payload
    model_package_arn = event['resources'][0]
    model_package_name = event['detail']['ModelPackageName']
    model_package_group_name = event['detail']['ModelPackageGroupName']
    model_package_version = int(event['detail']['ModelPackageVersion'])

    # Check if changed model version is actually the latest version in the registry
    latest_approved_model_version = get_latest_approved_model_version(model_package_group_name)
    if latest_approved_model_version is not model_package_version:
        raise Exception('Changed model package version is not the most recent version!')


    # Get the artifact location
    s3_model_artifact_location = get_latest_approved_s3_model_location(model_package_group_name)


    return {
        'modelPackageName': model_package_name,
        'modelPackageGroupName': model_package_group_name,
        'modelPackageVersion': model_package_version,
        'modelArtifactLocation': s3_model_artifact_location
    }