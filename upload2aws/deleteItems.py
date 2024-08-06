import boto3

# 初始化DynamoDB资源
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('prod-kairos-fe-stockPredict')

# 扫描表获取所有项目
response = table.scan()
data = response['Items']

# 删除每个项目
with table.batch_writer() as batch:
    for item in data:
        batch.delete_item(Key={'id': item['id']})

print("所有数据已删除")