# lambda.py

import json
import boto3
import base64

########################
# Function 1: serialize
########################
# This function reads an image file from S3, encodes it in base64, and returns the data

s3 = boto3.client('s3')

def serialize_handler(event, context):
    """
    A function to serialize target image data from S3.
    It reads the image file from S3, encodes it in base64, and returns it along with metadata.
    """
    # Extract S3 key and bucket name from the event
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the image to temporary location
    s3.download_file(bucket, key, "/tmp/image.png")

    # Encode the image as base64
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Return the image data and metadata
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

########################
# Function 2: classify
########################
# This function sends the base64 image to a SageMaker endpoint and adds inference results

ENDPOINT = "image-classification-2025-07-22-09-00-10-174"

def classify_handler(event, context):
    """
    A function to send image data to a SageMaker endpoint and classify the content.
    It appends inference results to the event and returns it.
    """
    # Decode the base64 image data
    image = base64.b64decode(event['body']['image_data'])

    # Initialize SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime')

    # Call the SageMaker endpoint
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType='image/png',
        Body=image
    )

    # Parse the prediction results
    result = response['Body'].read()
    inferences = json.loads(result)

    # Update the event with inference results
    return {
        'statusCode': 200,
        'body': json.dumps({
            "image_data": event["body"]['image_data'],
            "s3_bucket": event["body"]['s3_bucket'],
            "s3_key": event["body"]['s3_key'],
            "inferences": inferences
        })
    }

########################
# Function 3: filter
########################
# This function filters the results based on a confidence threshold

THRESHOLD = 0.9

def filter_handler(event, context):
    """
    A function to filter inference results.
    If the highest confidence score is below the threshold, it raises an exception.
    """
    # Parse event body if it's a JSON string
    body = event.get("body")
    if isinstance(body, str):
        body = json.loads(body)

    # Get the inference list
    inferences = body["inferences"]

    # Check if the maximum confidence score meets the threshold
    if max(inferences) > THRESHOLD:
        return {
            'statusCode': 200,
            'body': json.dumps(event)
        }
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")
