import * as cdk from '@aws-cdk/core';
import * as iot from '@aws-cdk/aws-iot';
import * as iam from '@aws-cdk/aws-iam';
import * as s3 from '@aws-cdk/aws-s3';
import * as sagemaker from '@aws-cdk/aws-sagemaker';
import * as cr from '@aws-cdk/custom-resources';
import * as lambda from '@aws-cdk/aws-lambda';
import { assert } from 'console';

interface AgentPackageCustomResourceProps {
    lambdaEnvVars?: {
        [key: string]: string
    } | {},
    assetsBucket: s3.IBucket,
}

export class AgentPackageCustomResource extends cdk.Construct {

    public readonly customResourceLambda: lambda.Function;
    public readonly customResource: cdk.CustomResource;
    
    constructor(scope: cdk.Construct, id: string, props: AgentPackageCustomResourceProps) {
        super(scope, id);

        /**
         * Set up Custom Resource Lambda
         */
        this.customResourceLambda = new lambda.Function(this, 'AgentPackageCrLambda', {
            code: lambda.Code.fromAsset('lib/constructs/EdgeManagerDeviceConfiguration/AgentPackageCustomResource/lambda'),
            handler: 'em_custom_resource.handler',
            runtime: lambda.Runtime.PYTHON_3_8,
            environment: props.lambdaEnvVars,
            timeout: cdk.Duration.seconds(30)
        });
        this.customResourceLambda.addToRolePolicy(new iam.PolicyStatement({
            actions: ['iot:*', 'sagemaker:*'],
            resources: ['*']
        }));
        this.customResourceLambda.addToRolePolicy(new iam.PolicyStatement({
            actions: ['s3:GetObject', 's3:PutObject'],
            resources: ['arn:aws:s3:::sagemaker-edge-release-store-us-west-2-linux-x64/*']
        }));
        props.assetsBucket.grantReadWrite(this.customResourceLambda);


        //   const crProvider = new cr.Provider(this, 'AgentPackageCrProvider', {
        //       onEventHandler: this.crLambda
        //   });

        this.customResource = new cdk.CustomResource(this, 'AgentPackageCr',
            {
                serviceToken: this.customResourceLambda.functionArn,
                properties: props.lambdaEnvVars
            });

    }
}
