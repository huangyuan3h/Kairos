from db import get_db_session, create_table
from src.training.predict import predict_stock_list, process_predict


def main():
    create_table()
    process_predict(report_date="2024-07-25", sync_all=False, version="simple_lstm_v2_1")



if __name__ == "__main__":
    main()
