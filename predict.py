from db import get_db_session, create_table
from db.predict_report import bulk_insert_predict_report
from db.stock_list import get_all_stock_list_data
from src.crawl.sync_daily_all import sync_daily_all
from src.measure_result.normalize_result import make_decision
from src.training.predict import predict_stock_list
from upload2aws.upload_to_dynamodb import import_2_aws_process


# from src.measure_result.normalize_result import make_decision
# from src.training.predict import predict_stock_list


def main():
    create_table()
    sync_daily_all()
    # risk_tolerance = 'moderate'
    with get_db_session() as db:
        stock_list = get_all_stock_list_data(db)
    stock_code_list = stock_list["code"].values
    df = predict_stock_list(stock_code_list)
    with get_db_session() as db:
        bulk_insert_predict_report(db, df)
    import_2_aws_process()
    # text_result = []
    # for idx in range(len(result)):
    #     predicted_returns = result[idx][0]
    #     decision = make_decision(predicted_returns, risk_tolerance)
    #     text_result.append(decision)
    #     print(predicted_returns)
    # print(text_result)


if __name__ == "__main__":
    main()
