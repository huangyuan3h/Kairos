import datetime
from decimal import Decimal

import boto3
import pandas as pd

from db import create_table, get_db_session
from db.predict_report import get_predict_report_by_date


def upload_df_to_dynamodb(df: pd.DataFrame, table_name: str, modelVersion = 'v1'):
    """
    将 Pandas DataFrame 中的数据上传到 DynamoDB 表

    Args:
        df (pd.DataFrame): 要上传的 DataFrame
        table_name (str): DynamoDB 表名
    """

    # 创建 DynamoDB 资源
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)

    # 将 DataFrame 转换为 DynamoDB item 格式
    items = df.to_dict('records')
    for item in items:
        # 将日期类型转换为字符串
        if 'report_date' in item:
            item['report_date'] = str(item['report_date'])

        # 将浮点数转换为 Decimal 类型，避免精度丢失
        for key, value in item.items():
            if isinstance(value, float):
                item[key] = Decimal(str(value))
        item["model"] = modelVersion
        # 上传数据到 DynamoDB
        table.put_item(Item=item)


def import_2_aws_process(report_date=None, report = None):
    if report is None:
        if report_date is None:
            report_date = datetime.datetime.today()
            report_date = report_date.strftime("%Y-%m-%d")
        with get_db_session() as db:
            report = get_predict_report_by_date(db, report_date)
    report['id'] = report['id'].astype(str)
    upload_df_to_dynamodb(report,"prod-kairos-fe-stockPredict")
    print("import to aws finished")