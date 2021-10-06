#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from '@aws-cdk/core';
import { CdkEdgeMlSmStack } from './lib/cdk-edge-ml-sm-stack';

const app = new cdk.App();

new CdkEdgeMlSmStack(app, `CdkEdgeMlSmStack`, {projectName: app.node.tryGetContext('projectName')});
