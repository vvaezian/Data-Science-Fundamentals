### Modify EC2 from `AWSCLI` ([docs](https://docs.aws.amazon.com/cli/latest/reference/ec2/)):
- Stop an instance (docs): `aws ec2 stop-instances --instance-ids i-1234567890abcdef0`
- Start an instance (docs): `aws ec2 start-instances --instance-ids i-1234567890abcdef0`
- Change instance type (docs): `aws ec2 modify-instance-attribute --instance-id i-1234567890abcdef0 --instance-type "{\"Value\": \"m1.small\"}"`

### Scaling an EC2 instance using AWS Lambda:
- Trigger Lambda from EC2 using AWSCLI ([docs](https://aws.amazon.com/blogs/architecture/understanding-the-different-ways-to-invoke-lambda-functions/)): `aws lambda invoke —function-name MyLambdaFunction —invocation-type RequestResponse —payload  “[JSON string here]”`
- Lambda function to stop EC2 instance, change its type and start the instance ([docs](https://aws.amazon.com/premiumsupport/knowledge-center/start-stop-lambda-cloudwatch/)) ([status codes](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_InstanceState.html))
```python
import boto3
from pprint import pprint

region = 'us-west-1'
instances = ['i-12345cb6de4f78g9h']
ec2 = boto3.client('ec2', region_name=region)

def get_status():
    return ec2.describe_instance_status(InstanceIds=instances)['InstanceStatuses'][0]['InstanceState']['Name']

def lambda_handler(event, context):
    # if get_status() == 'running':
    #     ec2.stop_instances(InstanceIds=instances)
    #     print('Stopped the following instances: ' + str(instances))
    
    else:
        print('Cannot stop the instance because it is not running.')
```
