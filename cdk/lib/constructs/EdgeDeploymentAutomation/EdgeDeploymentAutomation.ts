import * as cdk from '@aws-cdk/core';
import * as iot from '@aws-cdk/aws-iot';
import * as iam from '@aws-cdk/aws-iam';
import * as s3 from '@aws-cdk/aws-s3';
import * as events from '@aws-cdk/aws-events';
import * as events_targets from '@aws-cdk/aws-events-targets';
import { EdgeDeploymentStateMachine } from './EdgeDeploymentStateMachine/EdgeDeploymentStateMachine';

/**
 * The configuration for the Neo Compilation Job
 */
export interface neoCompilationSettingsProps {

  /**
   * The data input config, which maps to the DataInputConfig argument in the create_compilation_job() method
   *  in the boto3 SDK (see https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_compilation_job).
   *  Typically a stringified dictionary like '{"data": [1,3,224,224]}' that needs to correspond to model 
   *  you want to compile.
   */
  dataInputConfig: string,

  /**
   * The framework for the Neo Compilation Job, e.g. "MXNET"
   */
  framework: string,

  /**
   * The target platform JSON for the Neo Compilation Job, e.g. {'Os': 'LINUX', 'Arch': 'X86_64'}
   */
  targetPlatform: {
    [key: string]: string
  }
}

interface EdgeDeploymentAutomationProps {

  /**
   * The S3 bucket where the model artifacts will be stored
   */
  s3Bucket: s3.IBucket,

  /**
   * The prefix where the model artifacts should be stored
   */
  s3ModelArtifactPrefix: string,

  /**
   * Thing or Thing Group ARNs to send this deployment to
   */
  iotJobTargetArn: string,

  /**
   * Model name for this pipeline, needs to match with logic on the edge device
   */
  modelName: string,

  /**
   * Name of the model package group to listen to for approval events
   */
  modelPackageGroupName: string,

  /**
   * Settings for the Neo Compilation Job
   */
  neoCompilationSettings: neoCompilationSettingsProps,

  /**
   * Role for Sagemaker Services like Neo and Edge Manager to use for accessing S3 resources
   */
  sagemakerExecutionRole: iam.IRole
}

/**
 * Automation Construct for automated edge deployments via SageMaker Model Registry
 */
export class EdgeDeploymentAutomation extends cdk.Construct {
  constructor(scope: cdk.Construct, id: string, props: EdgeDeploymentAutomationProps) {
    super(scope, id);

    const edgeDeploymentStateMachine = new EdgeDeploymentStateMachine(this, 'EdgeDeploymentStateMachine', {
      ...props
    });


    const onAcceptModelVersionRule = new events.Rule(this, 'OnAcceptModelRule', {
      description: `Triggered once a model in the group ${props.modelPackageGroupName} is approved in Sagemaker Model Registry`,
      eventPattern: {
        source: ['aws.sagemaker'],
        detailType: ['SageMaker Model Package State Change'],
        detail: {
          'ModelApprovalStatus': ['Approved'],
          'ModelPackageGroupName': [props.modelPackageGroupName]
        }
      },
      targets: [
        new events_targets.SfnStateMachine(edgeDeploymentStateMachine.stateMachine)
      ]
    });





  }
}
