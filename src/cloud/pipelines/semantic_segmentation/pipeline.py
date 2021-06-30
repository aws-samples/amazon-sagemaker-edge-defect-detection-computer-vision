# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import numpy as np
import boto3
import time
import sagemaker
import sagemaker.session

from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.properties import PropertyFile
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline(
        region,
        role=None,
        default_bucket=None,
        pipeline_name="defect-detection-semantic-segmentation-pipeline",
        base_job_prefix="defect-detection-semantic-segmentation",
    ):
    """Gets a SageMaker ML Pipeline instance working with on DefectDetection data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    ## By enabling cache, if you run this pipeline again, without changing the input 
    ## parameters it will skip the training part and reuse the previous trained model
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")
    ts = time.strftime('%Y-%m-%d-%H-%M-%S')

    # Data prep
    processing_instance_type = ParameterString( # instance type for data preparation
        name="ProcessingInstanceType",
        default_value="ml.m5.xlarge"
    )
    processing_instance_count = ParameterInteger( # number of instances used for data preparation
        name="ProcessingInstanceCount",
        default_value=1
    )

    # Training
    training_instance_type = ParameterString( # instance type for training the model
        name="TrainingInstanceType",
        default_value="ml.c5.xlarge"
    )
    training_instance_count = ParameterInteger( # number of instances used to train your model
        name="TrainingInstanceCount",
        default_value=1
    )
    training_epochs = ParameterString( 
        name="TrainingEpochs",
        default_value="100"
    )

    # Dataset input data: S3 path
    input_data = ParameterString(
        name="InputData",
        default_value="",
    )
    
    # Model Approval State
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval"
    )

    # Model package group name for registering in model registry
    model_package_group_name = ParameterString(
        name="ModelPackageGroupName",
        default_value="defect-detection-semantic-segmentation-model-group"
    )

    # The preprocessor
    preprocessor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        max_runtime_in_seconds=7200,
    )

    # A preprocessing report to store some information from the preprocessing step for next steps
    preprocessing_report = PropertyFile(
        name='PreprocessingReport',
        output_name='preprocessing_report',
        path='preprocessing_report.json'
    )

    # Preprocessing Step
    step_process = ProcessingStep(
        name="DefectDetectionPreprocessing",
        code=os.path.join(BASE_DIR, 'preprocessing.py'), ## this is the script defined above
        processor=preprocessor,
        inputs=[
            ProcessingInput(source=input_data, destination='/opt/ml/processing/input')
        ],
        outputs=[
            ProcessingOutput(output_name='train_data', source='/opt/ml/processing/train'),
            ProcessingOutput(output_name='test_data', source='/opt/ml/processing/test'),
            ProcessingOutput(output_name='val_data', source='/opt/ml/processing/val'),
            ProcessingOutput(output_name='preprocessing_report', source='/opt/ml/processing/report')
        ],
        job_arguments=['--split', '0.1'],
        property_files=[preprocessing_report]
    )

    from sagemaker.tensorflow import TensorFlow
    model_dir = '/opt/ml/model'
    hyperparameters = {'epochs': training_epochs, 'batch_size': 8, 'learning_rate': 0.0001}
    estimator = TensorFlow(source_dir=BASE_DIR,
        entry_point='train_tf.py',
        model_dir=model_dir,
        instance_type=training_instance_type,
        #instance_type='local',
        instance_count=training_instance_count,
        hyperparameters=hyperparameters,
        role=role,
        output_path='s3://{}/{}/{}/{}'.format(default_bucket, 'models', base_job_prefix, 'training-output'),
        framework_version='2.2.0',
        py_version='py37',
        script_mode=True
    )
 
    step_train = TrainingStep(
        name="DefectDetectionSemanticSegmentationTrain",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                content_type='image/png',
                s3_data_type='S3Prefix'
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["val_data"].S3Output.S3Uri,
                content_type='image/png',
                s3_data_type='S3Prefix'
            )
        },
        cache_config=cache_config
    )

    # Register model step that will be conditionally executed
    step_register = RegisterModel(
        name="DefectDetectionSemanticSegmentationRegister",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["image/png"],
        response_types=["application/json"],
        inference_instances=["ml.c5.2xlarge", "ml.p3.2xlarge"],
        transform_instances=["ml.c5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            training_instance_count,
            training_epochs,
            input_data,
            model_approval_status,
            model_package_group_name
        ],
        steps=[step_process, step_train,  step_register],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
