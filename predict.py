from db import create_table
from src.training.predict import process_predict


def main():
    create_table()
    process_predict(report_date="2024-08-13", sync_all=False)
    # process_predict_classify(report_date="2024-07-31", sync_all=False, version="simple_lstm_classify")


if __name__ == "__main__":
    main()
