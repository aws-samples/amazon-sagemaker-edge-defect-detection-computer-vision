import * as cdk from '@aws-cdk/core';
import * as iot from '@aws-cdk/aws-iot';
import * as iam from '@aws-cdk/aws-iam';
import * as s3 from '@aws-cdk/aws-s3';
import * as lambda from '@aws-cdk/aws-lambda';
import * as sfn from '@aws-cdk/aws-stepfunctions';
import * as sfn_tasks from '@aws-cdk/aws-stepfunctions-tasks';
import { neoCompilationSettingsProps } from '../EdgeDeploymentAutomation';

interface EdgeDeploymentStateMachineProps {
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
   * Settings for the Neo Compilation Job
   */
  neoCompilationSettings: neoCompilationSettingsProps,

  /**
   * Role for Sagemaker Services like Neo and Edge Manager to use for accessing S3 resources
   */
   sagemakerExecutionRole: iam.IRole
}

/**
 * State Machine to automatically prepare and deploy the model to the edge
 */
export class EdgeDeploymentStateMachine extends cdk.Construct {

  readonly stateMachine: sfn.StateMachine;

  constructor(scope: cdk.Construct, id: string, props: EdgeDeploymentStateMachineProps) {
    super(scope, id);

    // LAMBDAS

    const lambdaEnvVars = {
      'MODEL_NAME': props.modelName,
      'IOT_JOB_TARGET_ARN': props.iotJobTargetArn,
      'S3_BUCKET_NAME': props.s3Bucket.bucketName,
      'S3_MODEL_ARTIFACT_PREFIX': props.s3ModelArtifactPrefix,
      'COMPILATION_JOB_INPUT_DATA_SIZE': props.neoCompilationSettings.dataSize,
      'COMPILATION_FRAMEWORK': props.neoCompilationSettings.framework,
      'COMPILATION_TARGET_PLATFORM': JSON.stringify(props.neoCompilationSettings.targetPlatform),
      'SAGEMAKER_EXECUTION_ROLE_ARN': props.sagemakerExecutionRole.roleArn
    };

    const initDeployment = new lambda.Function(this, 'InitDeploymentLambda', {
      code: lambda.Code.fromAsset('lib/constructs/EdgeDeploymentAutomation/EdgeDeploymentStateMachine/lambda'),
      handler: 'init_deployment.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      timeout: cdk.Duration.seconds(30),
      environment: lambdaEnvVars
    });
    initDeployment.addToRolePolicy(new iam.PolicyStatement({
      actions: ['sagemaker:DescribeModelPackage', 'sagemaker:DescribeModelPackageGroup', 'sagemaker:ListModelPackages'],
      resources: ['*'] //TODO: Scope down to resources used
    }))

    const startNeoCompilation = new lambda.Function(this, 'StartNeoCompilationLambda', {
      code: lambda.Code.fromAsset('lib/constructs/EdgeDeploymentAutomation/EdgeDeploymentStateMachine/lambda'),
      handler: 'start_neo_compilation.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      timeout: cdk.Duration.seconds(5),
      environment: lambdaEnvVars
    });
    startNeoCompilation.addToRolePolicy(new iam.PolicyStatement({
      actions: ['sagemaker:CreateCompilationJob'],
      resources: ['*']
    }));
    startNeoCompilation.addToRolePolicy(new iam.PolicyStatement({
      actions: ['iam:PassRole'],
      resources: [props.sagemakerExecutionRole.roleArn]
    }));

    const pollNeoCompilation = new lambda.Function(this, 'PollNeoCompilationLambda', {
      code: lambda.Code.fromAsset('lib/constructs/EdgeDeploymentAutomation/EdgeDeploymentStateMachine/lambda'),
      handler: 'poll_neo_compilation.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      timeout: cdk.Duration.seconds(5)
    });
    pollNeoCompilation.addToRolePolicy(new iam.PolicyStatement({
      actions: ['sagemaker:DescribeCompilationJob'],
      resources: ['*']
    }));

    const startEdgePackage = new lambda.Function(this, 'StartEdgePackageLambda', {
      code: lambda.Code.fromAsset('lib/constructs/EdgeDeploymentAutomation/EdgeDeploymentStateMachine/lambda'),
      handler: 'start_edge_package.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      timeout: cdk.Duration.seconds(5),
      environment: lambdaEnvVars
    });
    startEdgePackage.addToRolePolicy(new iam.PolicyStatement({
      actions: ['sagemaker:CreateEdgePackagingJob'],
      resources: ['*']
    }));
    startEdgePackage.addToRolePolicy(new iam.PolicyStatement({
      actions: ['iam:PassRole'],
      resources: [props.sagemakerExecutionRole.roleArn]
    }));

    const pollEdgePackage = new lambda.Function(this, 'PollEdgePackageLambda', {
      code: lambda.Code.fromAsset('lib/constructs/EdgeDeploymentAutomation/EdgeDeploymentStateMachine/lambda'),
      handler: 'poll_edge_package.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      timeout: cdk.Duration.seconds(5)
    });
    pollEdgePackage.addToRolePolicy(new iam.PolicyStatement({
      actions: ['sagemaker:DescribeEdgePackagingJob'],
      resources: ['*']
    }));

    const createIotJob = new lambda.Function(this, 'createIotJob', {
      code: lambda.Code.fromAsset('lib/constructs/EdgeDeploymentAutomation/EdgeDeploymentStateMachine/lambda'),
      handler: 'create_iot_job.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      timeout: cdk.Duration.seconds(5),
      environment: lambdaEnvVars
    });
    createIotJob.addToRolePolicy(new iam.PolicyStatement({
      actions: ['iot:CreateJob'],
      resources: [
        props.iotJobTargetArn,
        `arn:aws:iot:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:job/*`
      ]
    }));

    props.s3Bucket.grantReadWrite(startEdgePackage);
    props.s3Bucket.grantReadWrite(startNeoCompilation);

    // --- STEP FUNCTION ---

    const initDeploymentState = new sfn_tasks.LambdaInvoke(this, 'InitDeploymentState', {
      lambdaFunction: initDeployment,
      inputPath: '$',
      resultPath: '$.initResult'
    });

    const startNeoCompilationState = new sfn_tasks.LambdaInvoke(this, 'StartNeoCompilationState', {
      lambdaFunction: startNeoCompilation,
      inputPath: '$',
      resultPath: '$.startNeoCompilationResult'
    });

    const pollNeoCompilationState = new sfn_tasks.LambdaInvoke(this, 'PollNeoCompilationState', {
      lambdaFunction: pollNeoCompilation,
      inputPath: '$',
      resultPath: '$.pollNeoCompilationResult'
    });

    const waitNeoCompilationState = new sfn.Wait(this, 'WaitNeoCompilationState', {
      time: sfn.WaitTime.duration(cdk.Duration.seconds(60))
    });

    const startEdgePackageState = new sfn_tasks.LambdaInvoke(this, 'StartEdgePackageState', {
      lambdaFunction: startEdgePackage,
      inputPath: '$',
      resultPath: '$.startEdgePackageResult'
    });

    const pollEdgePackageState = new sfn_tasks.LambdaInvoke(this, 'PollEdgePackageState', {
      lambdaFunction: pollEdgePackage,
      inputPath: '$',
      resultPath: '$.pollEdgePackageResult'
    });

    const waitEdgePackageState = new sfn.Wait(this, 'WaitEdgePackageState', {
      time: sfn.WaitTime.duration(cdk.Duration.seconds(60))
    });


    const createIotJobState = new sfn_tasks.LambdaInvoke(this, 'CreateIotJobState', {
      lambdaFunction: createIotJob,
      inputPath: '$',
      resultPath: '$.createIotJobResult'
    });

    const choiceNeoCompilationState = new sfn.Choice(this, 'ChoiceNeoCompilationState');
    const choiceEdgePackageState = new sfn.Choice(this, 'ChoiceEdgePackageState');

    const failState = new sfn.Fail(this, 'FailState');

    const deploymentSfnDefinition = initDeploymentState
      .next(startNeoCompilationState)
      .next(pollNeoCompilationState)
      .next(choiceNeoCompilationState
        .when(sfn.Condition.or(sfn.Condition.stringEquals('$.pollNeoCompilationResult.Payload.compilationJobStatus', 'INPROGRESS'), sfn.Condition.stringEquals('$.pollNeoCompilationResult.Payload.compilationJobStatus', 'STARTING')), waitNeoCompilationState.next(pollNeoCompilationState))
        .when(sfn.Condition.stringEquals('$.pollNeoCompilationResult.Payload.compilationJobStatus', 'COMPLETED'),
          startEdgePackageState
            .next(pollEdgePackageState)
            .next(choiceEdgePackageState
              .when(sfn.Condition.or(sfn.Condition.stringEquals('$.pollEdgePackageResult.Payload.edgePackagingJobStatus', 'INPROGRESS'), sfn.Condition.stringEquals('$.pollEdgePackageResult.Payload.edgePackagingJobStatus', 'STARTING')), waitEdgePackageState.next(pollEdgePackageState))
              .when(sfn.Condition.stringEquals('$.pollEdgePackageResult.Payload.edgePackagingJobStatus', 'COMPLETED'), createIotJobState)
              .otherwise(failState)
            )
        )
        .otherwise(failState)
      );

    this.stateMachine = new sfn.StateMachine(this, 'EdgeDeploymentStateMachine', {
      definition: deploymentSfnDefinition
    });









  }
}
