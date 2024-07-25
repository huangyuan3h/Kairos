from db import get_db_session, create_table
from src.training.predict import predict_stock_list, process_predict


def main():
    create_table()
    process_predict(report_date="2024-07-22", sync_all=False, version="simple_lstm_v2_1")
    process_predict(report_date="2024-07-23", sync_all=False, version="simple_lstm_v2_1")
    process_predict(report_date="2024-07-24", sync_all=False, version="simple_lstm_v2_1")
    # import_2_aws_process('2024-07-16')


if __name__ == "__main__":
    main()
