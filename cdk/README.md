# MLOps at the edge using Sagemaker Edge Manager and the AWS CDK

In this project, we show how you can use the AWS CDK to define the infrastructure for automated model deployment onto the edge using AWS IoT, SageMake Edge Manger Fleets, IoT Jobs and Step Functions for automation of deployment.

## Creating the infrastructure with this CDK app

1. Install the AWS CDK following the official guide.
2. Install the dependencies by running `npm install`.
3. Deploy the CDK application by running `cdk deploy --parameters ProjectName=<YOUR PROJECT NAME>`. Choose a project name to assign to this project. You may choose any name, which is compliant with S3 bucket names, e.g. `sm-edge-project`.

## CDK Constructs

### EdgeManagerDeviceConfiguration

This construct makes setting up edge devices with SageMaker Edge Manager easy by providing a simple abstraction of AWS IoT things, a custom resource which configures the bundled package for edge device configuration and the wiring of those resources into a SageMaker Edge Manager fleet.

### EdgeDeploymentAutomation

This construct provides an easy-to-use abstraction for an automated deployment mechanism onto edge devices by wiring up a Step Function workflow to events emitted by SageMake Model Registry status changes. It creates a Step Functions state machine which automatically calls AWS APIs to do the following:

1. Create a Neo Compilation job
2. Once finished successfully, create a SageMaker Edge Packaging job
3. Once finished successfully, create an AWS IoT targeted onto the thing group associated with your edge devices
