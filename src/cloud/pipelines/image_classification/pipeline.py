# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import numpy as np
import glob
import boto3
import time
import sagemaker
import sagemaker.session

from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep, CacheConfig
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.inputs import TrainingInput, CreateModelInput
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.model import Model
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.image_uris import retrieve

from botocore.exceptions import ClientError, ValidationError

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
        pipeline_name="defect-detection-img-classification-pipeline",
        base_job_prefix="defect-detection-img-classification",
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
    
    # Input shape
    # --> Image size (height and width, as we need only use square images) desired for training. The 
    # pipeline will square the images to this size if they are not square already by adding padding.
    target_image_size = ParameterString( 
        name="TargetImageSize",
        default_value="224"
    )
    
    # Augement Count
    augment_count_normal = ParameterString( # by how many samples you want to augment the normal samples
        name="AugmentCountNormal",
        default_value="0"
    )
    augment_count_anomalous = ParameterString( # by how many samples you want to augment the anomalous samples
        name="AugmentCountAnomalous",
        default_value="0"
    )

    # Training
    training_instance_type = ParameterString( # instance type for training the model
        name="TrainingInstanceType",
        default_value="ml.p3.2xlarge"
    )
    training_instance_count = ParameterInteger( # number of instances used to train your model
        name="TrainingInstanceCount",
        default_value=1
    )
    training_epochs = ParameterString( 
        name="TrainingEpochs",
        default_value="15"
    )
    training_num_training_samples = ParameterString(
        name="TrainingNumTrainingSamples",
        default_value="3600" # Change this to the number of training samples used!
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
        default_value="defect-detection-img-classification-model-group"
    )
    

    aws_region = sagemaker_session.boto_region_name
    training_image = retrieve(framework='image-classification', region=aws_region, image_scope='training')
    
    # Hardcoded hyperparameters
    NUM_CLASSES = 2
    BATCH_SIZE = 8

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
        job_arguments=[
            '--split', '0.1',
            '--augment-count-normal', augment_count_normal,
            '--augment-count-anomalous', augment_count_anomalous,
            '--image-width', target_image_size,
            '--image-height', target_image_size
        ],
        property_files=[preprocessing_report]
    )

    # Define Image Classification Estimator
    hyperparameters = {
        'num_layers': 18,
        'image_shape': Join(on=',', values=['3', target_image_size, target_image_size]),
        'num_classes': NUM_CLASSES,
        'mini_batch_size': BATCH_SIZE,
        'num_training_samples': training_num_training_samples,
        'epochs': training_epochs,
        'learning_rate': 0.01,
        'top_k': 2,
        'use_pretrained_model': 1,
        'precision_dtype': 'float32'
    }
    
    ic_estimator = Estimator(
        image_uri=training_image,
        role=role,
        instance_count=training_instance_count,
        instance_type=training_instance_type,
        volume_size = 50,
        max_run = 360000,
        input_mode= 'Pipe',
        base_job_name='img-classification-training',
        output_path='s3://{}/{}/{}/{}'.format(default_bucket, 'models', base_job_prefix, 'training-output'),
        hyperparameters=hyperparameters
    )
    
    step_train = TrainingStep(
        name="DefectDetectionImageClassificationTrain",
        estimator=ic_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                content_type="application/x-recordio",
                s3_data_type='S3Prefix'
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["val_data"].S3Output.S3Uri,
                content_type="application/x-recordio",
                s3_data_type='S3Prefix'
            )
        },
        cache_config=cache_config
    )

    # Set up for the evaluation processing step
    evaluation_report = PropertyFile(
        name='EvaluationReport',
        output_name='evaluation_report',
        path='evaluation_report.json'
    )

    evalation_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        max_runtime_in_seconds=7200
    )

    step_eval = ProcessingStep(
        name="DefectDetectionEvaluation",
        code=os.path.join(BASE_DIR, 'evaluation.py'), ## this is the script defined above
        processor=evalation_processor,
        inputs=[
            ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri, destination='/opt/ml/processing/test'),
            ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, destination='/opt/ml/processing/model')

        ],
        outputs=[
            ProcessingOutput(output_name='evaluation_report', source='/opt/ml/processing/report')
        ],
        property_files=[evaluation_report],
        job_arguments=[
            '--image-width', target_image_size,
            '--image-height', target_image_size
        ],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation_report.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    # Register model step that will be conditionally executed
    step_register = RegisterModel(
        name="DefectDetectionImageClassificationRegister",
        estimator=ic_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/x-recordio"],
        response_types=["application/json"],
        inference_instances=["ml.c5.2xlarge", "ml.p3.2xlarge"],
        transform_instances=["ml.c5.xlarge"],
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
        approval_status=model_approval_status
    )

    # Condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(  # You can change the condition here
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="multiclass_classification_metrics.accuracy.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.
        ),
        right=0.8,  # You can change the threshold here
    )
    step_cond = ConditionStep(
        name="DefectDetectionImageClassificationAccuracyCondition",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            target_image_size,
            augment_count_normal,
            augment_count_anomalous,
            training_instance_type,
            training_instance_count,
            training_num_training_samples,
            training_epochs,
            input_data,
            model_approval_status,
            model_package_group_name
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )
    return pipeline