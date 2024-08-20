import requests
from retry import retry
import boto3


MAX_RETRY = 3


@retry(tries=MAX_RETRY)
def get_client(service_name, clients={}):
    if service_name in clients:
        return clients[service_name]

    try:
        clients[service_name] = boto3.client(service_name=service_name, region_name='us-east-1')
    except Exception as e:
        # get region from gateway
        response = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={
                "X-aws-ec2-metadata-token-ttl-seconds": "21600",
            },
        )
        token = response.text
        response = requests.get(
            "http://169.254.169.254/latest/meta-data/placement/region",
            headers={
                "X-aws-ec2-metadata-token": token,
            },
        )
        boto3.setup_default_session(region_name=response.text)
        print("Automatically set region to", response.text)
        clients[service_name] = boto3.client(service_name=service_name)
    return clients[service_name]
