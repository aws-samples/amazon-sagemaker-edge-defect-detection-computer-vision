import * as cdk from '@aws-cdk/core';
import * as iot from '@aws-cdk/aws-iot';
import * as iam from '@aws-cdk/aws-iam';
import * as s3 from '@aws-cdk/aws-s3';
import * as sagemaker from '@aws-cdk/aws-sagemaker';

interface EdgeDeviceRoleProps {
    /**
     * The bucket to save the assets for the project
     */
    assetsBucket: s3.IBucket,

    /**
     * The name of the project, used to reference multiple resources in the setup
     */
    projectName: string
}

export class EdgeDeviceRole extends iam.Role {

    constructor(scope: cdk.Construct, id: string, props: EdgeDeviceRoleProps) {

        const assumedBy = new iam.CompositePrincipal(
            new iam.ServicePrincipal('sagemaker.amazonaws.com'),
            new iam.ServicePrincipal('iot.amazonaws.com'),
            new iam.ServicePrincipal('credentials.iot.amazonaws.com')
        );

        const roleName = `edge-device-role-${props.projectName}`;

        const devicePolicy = new iam.PolicyDocument({
            statements: [
                new iam.PolicyStatement({
                    actions: ['s3:ListAllMyBuckets'],
                    resources: ['*'],
                }),
                new iam.PolicyStatement({
                    actions: ['iot:CreateRoleAlias', 'iot:DescribeRoleAlias', 'iot:UpdateRoleAlias', 'iot:TagResource', 'iot:ListTagsForResource'],
                    resources: [`arn:aws:iot:${cdk.Aws.REGION}:${cdk.Aws.ACCOUNT_ID}:rolealias/SageMakerEdge*`],
                }),
                new iam.PolicyStatement({
                    actions: ['iam:GetRole', 'iam:PassRole'],
                    resources: [
                        `arn:aws:iam::${cdk.Aws.ACCOUNT_ID}:role/*SageMaker*`,
                        `arn:aws:iam::${cdk.Aws.ACCOUNT_ID}:role/*Sagemaker*`,
                        `arn:aws:iam::${cdk.Aws.ACCOUNT_ID}:role/*sagemaker*`,
                        `arn:aws:iam::${cdk.Aws.ACCOUNT_ID}:role/${roleName}`
                    ],
                }),
                new iam.PolicyStatement({
                    actions: ['sagemaker:GetDeviceRegistration', 'sagemaker:SendHeartbeat', 'sagemaker:DescribeDevice'],
                    resources: ['*'],
                }),
                new iam.PolicyStatement({
                    actions: ['iot:DescribeEndpoint'],
                    resources: ['*'],
                }),
                new iam.PolicyStatement({
                    actions: ['s3:*'],
                    resources: [props.assetsBucket.bucketArn, props.assetsBucket.bucketArn + '/*'],
                })
            ]
        });

        super(scope, id, { ...props, assumedBy: assumedBy, inlinePolicies: { EdgeDevicePolicy: devicePolicy }, roleName: roleName });

        // this.role.addToPolicy(
        //     new iam.PolicyStatement({
        //         actions: ['iam:GetRole', 'iam:PassRole'],
        //         resources: [this.role.roleArn]
        //     })
        // )
    }
}