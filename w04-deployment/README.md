# Deployment
## 1. Three ways of deploying a mode
The output from training step is model which will be deployed into production. There are several ways to deploy a model depending on when the prediction is needed, i.e: immediate prediction or with waiting time. 
- **Batch (offline)**. The model is not up and running all the time, it will be run on a regular interval.
- **Online**. The model is always available. This online deployment can be deployed as:
    - Web service. Available trough http request
    - Streaming. The service is listening for events on the stream.
### Batch (offline) mode
It will be run on a regular interval, e.g: hourly, daily, monthly, etc. Scoring/prediction job will pull the data from the database and applies the model to this data.

Example: churn prediction. This prediction can be deployed in batch mode because we don't need the prediction result immediately. It can be run on a daily or weekly interval.

### Online (Web Service)
The model needs to be online all the time because user needs to know the duration prediction each time a trip is made.

Example: ride duration prediction. The service need to be available immediately when a user is requesting a taxi drive.
### Online (Streaming)
In a streaming service we need producer and consumers. In a web service we have 1-on-1 relationship between the web service and ride duration service, while in streaming service it can be 1-to-many relationship between the producer and consumers.

Example: 
- When a ride started, the producer will send data related to the taxi trip and each consumer will run their own prediction based on the data receved from the producer. i.e: one consumer will run ride duration prediction, the other one will run tip prediction.
- When a user upload a video into streaming service such as Youtube, there will be several consumers to handle content moderation. One consumer can check for copyright violation, another consumer checking if there is an age-restricted content and another consumer to check for restricted content, e.g: violence, hate speech, etc.
## 2. Web Services: Deploying models with Flask and Docker

## 3. Getting the models from the model registry (MLflow)

## 4. (Optional) Streaming - Deploying models with Kinesis and Lambda

## 5. Batch: Preparing a scoring script

## Homework

[Homework Solution](homework.ipynb)