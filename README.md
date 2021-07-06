# Defect detection using computer vision at the edge with Amazon SageMaker

In this workshop, we will walk you through a step by step process to build and train computer vision models with Amazon SageMaker and package and deploy them to the edge with [SageMaker Edge Manager](https://aws.amazon.com/sagemaker/edge-manager/). The workshop focuses on a defect detection use case in an industrial setting with models like image classification, and semantic segmentation to detect defects across several object types. We will complete the MLOps lifecycle with continuous versioned over-the-air model updates and data capture to the cloud.

## Architecture

The architecture that will be built during this workshop is shown below. Several key components can be highlighted:

1. **Model development and training on the cloud**: This repository contains code for two pipelines based on [SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html) for each model type. Those pipelines are being built and executed in a SageMaker Studio notebook.
2. **Model deployment to the edge**: Once a model building pipeline executed successfully, models will be compiled (with SageMaker Neo) and packaged with a SageMaker Edge Manager packaging job. As such, they can be deployed onto the edge device via IoT Jobs. On the edge device, there is an application running which will receive the job payload via MQTT and download the relevant model pacakge.
3. **Edge inference**: The edge device which is running the application for defect detection. In this workshop, we will use an EC2 instance to simulate an edge device - but any hardware device (RaspberryPi, Nvidia Jetson) can be used as long as SageMaker Neo compilations supported for it. During setup, a configuration package is being downloaded to edge device to configure SageMaker Edge Agent. The Edge Agent on the device can then load models deployed via OTA updates and make them available for prediction via a low-latency gRPC API (see [SageMaker Edge Manager documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/edge.html)).

![architecture](img/architecture.png)

## Dataset

This workshop is designed to be used with any dataset for defect detection that includes labels and masks. To be able to use both models (see section [Models](#models)), you will need a dataset of labelled images (*normal* and *anomalous*) as well as a set of respective *ground truth masks* which identify where the defect on a part is located. To train the models with the provided pipeline without any major code adjustments, you merely need to upload the dataset in the format together with correct path prefixes in an S3 bucket. Please refer to the [Getting Started](#getting-started) guide below on more details for model training with a dataset.

## Models

In this workshop, you will build two types of ML models:

* an image classification model based on the AWS [built-in SageMaker Image Classification algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html)
* a custom semantic segmentation model based built with Tensorflow/Keras using the [UNET deep learning architecture](https://arxiv.org/abs/1505.04597)

## Directory structure of this repository

This repository has the following directory structure:

```
├── setup           <-- contains the CloudFormation template for easy-to-use setup of AWS resources
└── src             <-- contains the actual source code for this project
    ├── cloud       <-- contains the code for model training in the cloud and initiation of OTA deployments to the edge
    └── edge        <-- contains the code that is running on the edge device
```

### Edge code directory structure

```
src/edge
├── app                         <-- python module for this application
│   ├── edgeagentclient.py      <-- abstractions for calling edge agent gRPC APIs
│   ├── logger.py               <-- utilities for logging output to AWS IoT Core
│   ├── ota.py                  <-- utilities for handling OTA IoT jobs
│   └── util.py                 <-- additional utilities
├── install.py                  <-- install script for downloading and configuring edge agent
├── models_config.json          <-- model configuration, also used for persisting model versions
├── run.py                      <-- runs the edge application
├── start_edge_agent.sh         <-- starts the SM edge agent
├── static                      <-- contains static images for Flask app, download test images here
└── templates                   <-- contains HTML Jinja templates for Flask app
```

### Cloud code directory structure

```
src/cloud
├── image_classification_pipeline.ipynb     <-- notebook for running the image classification pipeline
├── semantic_segmentation_pipeline.ipynb    <-- notebook for running the semantic segmentation pipeline
└── pipelines                               <-- model building code and pipeline definition
    ├── get_pipeline_definition.py          <-- CLI tool for CICD
    ├── run_pipeline.py                     <-- CLI tool for CICD
    ├── image_classification                <-- contains the pipeline code for image classification
    │   ├── evaluation.py                   <-- script to evaluate model performance on test dataset
    │   ├── pipeline.py                     <-- pipeline definition
    │   └── preprocessing.py                <-- script for preprocessing (augmentation, train/test/val split)
    └── semantic_segmentation               <-- contains the pipeline code for semantic segmentation
        ├── pipeline.py                     <-- pipeline definition
        ├── preprocessing.py                <-- script for preprocessing (augmentation, train/test/val split)
        ├── requirements.txt                <-- python dependencies needed for training
        └── train_tf.py                     <-- training script for training the unet model

```

## Getting started

Please follow the steps below to start building your own edge ML project. Please note that model training in the cloud and running inference on the edge are interdependent to another. We recommend you start by setting up the edge device first and then train the models as a second step. You can then directly deploy them to the edge after you have successfully trained the models.

### Setting up workshop resources by launching the CloudFormation stack

1. Launch a new CloudFormation stack with the provided template under `setup/template.yaml`.
2. Define a name for the stack and enter a *Project Name* parameter, that is unique in your account. The project name that you define during stack creation defines the name of many of the resources that are being created with the stack. Make sure to take note of this parameter.
3. Have a look at the CloudFormation stack outputs and take note of the provided information.

#### What is being created with the CloudFormation stack?

This stack configures several resources needed for this workshop. It sets up an IoT device together with certificates and roles, an Edge Manager fleet, registers the device with the fleet and creates a package for edge agent configuration. The following image illustrates the resources being created with the CloudFormation stack.

![edge config](img/cloudformation.png)

### Configuring the edge device

1. Launch an EC2 instance with Ubuntu Server 20 with SSH access (e.g. via [Session Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html)) into a public subnet and make sure it gets assigned a public IP (you will need this later to access the web application). Ensure that it has access to the S3 buckets containing your configuration package (find the bucket name in the CloudFormation output). It will also need access to the bucket containing the SageMaker Edge Agent binary. For more information, refer to the [SageMaker Edge Manager documentation pages](https://docs.aws.amazon.com/sagemaker/latest/dg/edge-device-fleet-about.html). This EC2 instance will from now be considered our "edge device".
2. Clone this GitHub repository onto the edge device or simply copy the `src/edge` directory onto the edge device.
3. Install the dependencies by running `sudo apt update -y && sudo apt install -y build-essential procps` and `pip install -U numpy sysv_ipc boto3 grpcio-tools grpcio protobuf sagemaker paho-mqtt waitress`.
4. Run the installation script by running `python3 install.py --project-name <YOUR PROJECT NAME> --account-id <YOUR ACCOUNT ID>`. This script will download the edge agent configuration package created during the CloudFormation deployment, download the edge agent binary and also generate the protobuf agent stubs. A newly created directory `./agent/` contains the files for the edge agent. The following image illustrated what happens in the installation script:

![edge config](img/edge_config.png)

5. Create an environment variable to define the location of the agent directory. If you haven't changed your current directory, this would likely be `export SM_EDGE_AGENT_HOME=$PWD/agent`.
6. Start the edge agent by running `./start_edge_agent.sh`, which launches the edge agent on the unix socket `tmp/edge_agent`. You should now the able to interact with the edge agent from your application.
7. Before running the actual application, you need to define an environment variable which determine whether you want to run the app with the Flask development server or the with a production-ready uWSGI server (using [waitress](https://github.com/Pylons/waitress)). For now, lets use the production server by setting `export SM_APP_ENV=prod`. For debugging, you might want to later change this to `dev`.
8. Run the application with `python3 run.py` to initialize the application, verify cloud connectivity, connect to the edge agent. This application is a [Flask](https://flask.palletsprojects.com/en/2.0.x/) web application running port port 8080 which is integrated with SageMaker Edge Agent and AWS IoT for OTA updates. You will see that, if you have no models deployed yet and have not downloaded any test images, nothing will happen yet in the application. It will stay idle until it can access test images in the `/static` folder and run inference on those with a deployed model. In the next step, we will see how we can run automated model training with SageMaker Pipelines and deploy them onto the edge device for local inference.
9.  Go to the EC2 dashboard and browse the public IP address on port 8080, i.e. `http://<PUBLIC_IP>:8080`. You should now see the application in your browser window. *(Toubleshoot: ensure that you allow ingress on port 8080 in your instance's security group. Also, make sure your local firewalls on your device allow ingress through port 8080)*

### Automated model training in the cloud with SageMaker Pipelines

1. Create a SageMaker Studio domain and user by following [this](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks.html) guide in the documentation. Make sure that the IAM role used has access to the S3 bucket created during the CloudFormation deployment.
2. Clone this repository or copy the `src/cloud` directory onto the SageMaker Studio domain.
3. Prepare your dataset for training: with the provided pipeline code you can train two model types (image classification and semantic segmentation). You might want to set aside some images to be used for local inference. Download those onto the edge device and save them into the `static` folder so they can be used for inference by the edge application. To use the pipelines without any code modifications you need to use structure your datasets as follows:
   * **Image Classification**: Your dataset needs to be split into `normal` and `anomalous` directories according to their respective label. Upload the data to your S3 bucket (e.g. under `s3://<BUCKET-NAME>/data/img-classification/`). Thus, your normal images will be located in `s3://<BUCKET-NAME>/data/img-classification/normal` and the anomalous ones in `s3://<BUCKET-NAME>/data/img-classification/anomalous`. Train / test / validation split will be done automatically in the preprocessing step of the pipeline.
   * **Semantic Segmentation**: Your dataset needs to be split into `images` and `masks` directories. Upload the data to your S3 bucket (e.g. under `s3://<BUCKET-NAME>/data/semantic-segmentation/`). Thus, your images will be located in `s3://<BUCKET-NAME>/data/semantic-segmentation/images` and the binary masks in `s3://<BUCKET-NAME>/data/semantic-segmentation/masks`. Train / test / validation split will be done automatically in the preprocessing step of the pipeline.
4. Execute the training pipeline: you will find a Jupyter Notebook for each of the model types in `src/cloud/`. Please adjust the project name you used during the CloudFormation deployment in the notebook. Also, you need to provide the S3 input data path as a parameter of the pipeline. Please make sure this aligns with the S3 path you used for uploading the dataset in step 3. You can monitor the pipeline execution in your SageMaker Studio domain. In case it finishes successfully, it should look similar to the one displayed below.

![pipeline](img/pipeline.png)

5. Once the pipeline finished successfully, your model is almost ready for use on the edge device. Verify that the latest model version in the model registry is approved to make it available for edge deployment. Execute the following cells of the notebook to run model compilation with SageMaker Neo and then package the model for usage with SageMaker Edge Manager. Finally, you can deploy the model package onto the edge by running the IoT Job as an Over-The-Air update. If your edge application is currently running, it should receive the OTA deployment job, download the model package and load it into the Edge Agent. Please verify that it works by checking the log output on the edge device. You can also verify the successful deployment of a new model version by verifying the successful execution of the IoT job in the AWS IoT Core Console (under "Manage" --> "Jobs") as shown below.

![pipeline](img/iot_job.png)

#### Persisting model configuration

You can set which models should be loaded initially by configuring the `model_config.json` file. The application will instruct the edge agent to load these models upon startup. You can update model versions by creating IoT jobs from the cloud. The OTA IoT client running alongside the application will listen to the job topics and download the model accordingly. Please also note that for each new model you deploy you might have to adjust your application code accordingly (e.g. if your input shape changes). The structure of the `model_config.json` file with a sample configuration is shown below.

In `"mappings"`, you can define which model should be used for each of the two inferences in the application this name needs to align with the model name you choose during OTA deployment. In `"models"`, information about the models loaded into the edge agent are persisted even after you shutdown the application. Please note that this is automatically filled out by the application and saved before you close out of the application. You do not need to manually configure this. In case you want to use a manually deployed model package with this application, you can instruct the application to load this model by manually adding a model definition into the JSON file under `"models"`.

```json
{
  "mappings": {
    "image-classification-app": "img-classification",
    "image-segmentation-app": "unet"
  },
  "models": [
    {
      "name": "img-classification",
      "version": "1",
      "identifier": "img-classification-1"
    }
  ]
}
```

### Running inference on the edge device

To run inference on the device, you need to have fulfilled the following requirements:

* The edge agent on the edge device is properly configured and can successfully authenticate against AWS IoT
* You have downloaded test images onto the edge device in the folder `static/`
* You have deployed at least one of the two models (image classification or semantic segmentation) via OTA updates
* The edge agent is running and the models could be loaded successfully (for troubleshooting check command line output or edge agent logs in `agent/logs/agent.log`)

### Deploying new model versions to the edge

You can now continuously retrain your model on new data or with new parameter configurations and deploy them onto the edge device by running again through steps 1-5 in [Automated model training in the cloud with Sagemaker Pipelines](#automated-model-training-in-the-cloud-with-sagemaker-pipelines). Your application on the edge device will automatically download the new model packages (if the versions provided is higher than the one used currently). It then unloads old model version from the edge agent and loads the newer version once available. It persists its model configuration in the JSON file described in section 5 of [Automated model training in the cloud with Sagemaker Pipelines](#automated-model-training-in-the-cloud-with-sagemaker-pipelines).

## Security

See [CONTRIBUTING](CONTRIBUTING.md) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
