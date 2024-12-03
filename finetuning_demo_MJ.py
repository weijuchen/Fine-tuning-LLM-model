# Upload fine-tuning files
import time

from openai import  AzureOpenAI
import os

# Config Parse
from configparser import ConfigParser
config = ConfigParser()
config.read("config.ini")

key = config["Azure_OpenAI"]["API_KEY"]
# key = config["OPEN_AI"]["API_KEY"]
client = AzureOpenAI(
    azure_endpoint="https://chatbot-sc.openai.azure.com/",
    
    # azure_endpoint="https://neyahnorthcentralus.openai.azure.com/",
 
    api_version="2024-02-15-preview",
    # api_version="2023-09-15-preview",
    api_key=key

)
training_file_name = 'training_set.jsonl'
validation_file_name = 'validation_set.jsonl'

# Upload the training and validation dataset files to Azure OpenAI with the SDK.

training_response = client.files.create(
    file=open(training_file_name, "rb"), purpose="fine-tune"
)
training_file_id = training_response.id

validation_response = client.files.create(
    file=open(validation_file_name, "rb"), purpose="fine-tune"
)
validation_file_id = validation_response.id

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)

time.sleep(10)

response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-4o-2024-08-06"
    # model="gpt-35-turbo-0613",
)

job_id = response.id

# You can use the job ID to monitor the status of the fine-tuning job.
# The fine-tuning job will take some time to start and complete.

print("Job ID:", response.id)
print("Status:", response.status)
print(response)

#Retrieve training job ID

response = client.fine_tuning.jobs.retrieve(job_id)

print("Job ID:", response.id)
print("Status:", response.status)
print(response)