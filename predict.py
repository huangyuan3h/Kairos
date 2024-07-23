from db import get_db_session, create_table
from db.predict_report import bulk_insert_predict_report
from db.stock_list import get_all_stock_list_data, get_predict_stock_list_data
from src.crawl.sync_daily_all import sync_daily_all
from src.training.predict import predict_stock_list, process_predict
from upload2aws.upload_to_dynamodb import import_2_aws_process


def main():
    create_table()
    process_predict(report_date="2024-07-23", sync_all=False, version="simple_lstm_v2_1")
    # import_2_aws_process('2024-07-16')


if __name__ == "__main__":
    main()
