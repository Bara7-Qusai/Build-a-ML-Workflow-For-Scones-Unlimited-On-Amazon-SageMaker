# Build a ML Workflow For Scones Unlimited On Amazon SageMaker

## Project Overview
This project involves building and deploying an image classification model using Amazon SageMaker for Scones Unlimited, a logistics company focused on delivering fresh scones. The goal is to automate the identification of vehicle types used for deliveries, optimizing their logistics pipeline through machine learning.

---

## Dataset

We use the **CIFAR-100** dataset, which contains 60,000 color images (32×32 pixels) across 100 object categories, with 600 images per class. This simulates real-world vehicle classification challenges.

[Download CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

---
##  Lambda Functions

All three Lambda functions used in this workflow are defined inside a **single file: `lambda.py`**:

1. **Data Serialization** – Loads input image data from S3 and prepares it for inference  
2. **Model Inference** – Invokes the SageMaker endpoint to perform prediction  
3. **Result Deserialization** – Processes the output and returns the final response


---

## Step Functions Workflow

The complete pipeline is orchestrated using **AWS Step Functions**, connecting:

- S3 (data storage)
- Lambda functions (processing & inference)
- SageMaker (training & hosting)

📊 Workflow Diagram:  
![Step Function]( )

