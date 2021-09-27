import * as cdk from '@aws-cdk/core';
import * as iot from '@aws-cdk/aws-iot';
import * as iam from '@aws-cdk/aws-iam';
import * as s3 from '@aws-cdk/aws-s3';
import * as sagemaker from '@aws-cdk/aws-sagemaker';
import { AgentPackageCustomResource } from './AgentPackageCustomResource/AgentPackageCustomResource';

interface EdgeManagerDeviceConfigurationProps {

  /**
   * The name of the IoT thing to create with this configuration
   */
  thingName: string,

  /**
   * The name of the thing group to use for this newly created IoT thing (creates new group if not exists)
   */
  thingGroupName: string,

  /**
   * The S3 bucket to save the configuration for Edge Agent in
   */
  assetsBucket: s3.IBucket,

  /**
   * The SageMaker Edge Manager Device Fleet to associate with this thing
   */
  smEdgeDeviceFleet: sagemaker.CfnDeviceFleet
}

/**
 * Configures a new IoT Thing for usage with SM Edge Manager and associates it with a Edge Fleet. Creates a 
 *  configuration package using a Lambda-backed custom resources and stores it in Amazon S3 in the provided 
 *  bucket.
 */
export class EdgeManagerDeviceConfiguration extends cdk.Construct {

  public readonly edgeDeviceIotThing: iot.CfnThing;

  constructor(scope: cdk.Construct, id: string, props: EdgeManagerDeviceConfigurationProps) {
    super(scope, id);

    // The IoT Thing
    this.edgeDeviceIotThing = new iot.CfnThing(this, 'EdgeDeviceIotThing', {
      thingName: props.thingName
    });

    const roleAliasName = `SageMakerEdge-${props.smEdgeDeviceFleet.ref}`;
    const topicName = 'defect-detection';

    const edgeDeviceIotPolicy = new iot.CfnPolicy(this, 'EdgeDeviceIotPolicy', {
      policyDocument: constructEdgeDeviceIotPolicy(props.thingName, cdk.Aws.REGION, cdk.Aws.ACCOUNT_ID, topicName, roleAliasName),
      policyName: `${props.thingName}-policy`
    });

    // The Custom Resource which builds the agent package to make it available for easy download and installation
    const agentPackageBuilder = new AgentPackageCustomResource(this, 'AgentPackageBuilder', {
      assetsBucket: props.assetsBucket,
      lambdaEnvVars: {
        SM_EDGE_DEVICE_NAME: this.edgeDeviceIotThing.thingName,
        IOT_ROLE_ALIAS_NAME: roleAliasName,
        SM_EDGE_FLEET_NAME: props.smEdgeDeviceFleet.ref,
        IOT_THING_NAME: props.thingName,
        IOT_POLICY_NAME: edgeDeviceIotPolicy.ref,
        IOT_THING_GROUP_NAME: props.thingGroupName,
        BUCKET_NAME: props.assetsBucket.bucketName
      }
    });

    agentPackageBuilder.customResource.node.addDependency(this.edgeDeviceIotThing);
  }
}


/**
 * Construct the edge device IoT Policy
 * @param thingName The thing name
 * @param region AWS region
 * @param accountId AWS account id
 * @param topicName the topic for interacting with the thing
 * @param roleAliasName the name of the role alias
 * @returns IoT Policy in JSON
 */
function constructEdgeDeviceIotPolicy(thingName: string, region: string, accountId: string, topicName: string, roleAliasName: string) {
  const policy = {
    Version: "2012-10-17",
    Statement: [{
      Effect: "Allow",
      Action: "iot:Connect",
      Resource: [`arn:aws:iot:${region}:${accountId}:client/*`]
    }, {
      Effect: "Allow",
      Action: ["iot:Publish", "iot:Receive"],
      Resource: [
        `arn:aws:iot:${region}:${accountId}:topic/${topicName}/*`,
        `arn:aws:iot:${region}:${accountId}:topic/$aws/*`
      ]
    }, {
      Effect: "Allow",
      Action: ["iot:Subscribe"],
      Resource: [
        `arn:aws:iot:${region}:${accountId}:topicfilter/${topicName}/*`,
        `arn:aws:iot:${region}:${accountId}:topicfilter/$aws/*`,
        `arn:aws:iot:${region}:${accountId}:topic/$aws/*`
      ]
    }, {
      Effect: "Allow",
      Action: ["iot:UpdateThingShadow"],
      Resource: [
        `arn:aws:iot:${region}:${accountId}:topicfilter/${topicName}/*`,
        `arn:aws:iot:${region}:${accountId}:thing/${thingName}`
      ]
    }, {
      Effect: "Allow",
      Action: ["iot:AssumeRoleWithCertificate"],
      Resource: [
        `arn:aws:iot:${region}:${accountId}:rolealias/${roleAliasName}`
      ]
    }]
  };
  return policy;
}
