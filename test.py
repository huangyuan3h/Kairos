import datetime

from data.data_merging import get_stock_total_data, get_random_valid_data
from data.data_preprocessing import merge_financial_data
from data.raw import get_sh_a_stock_list

from db import create_table, get_db_session
from db.predict_report import get_predict_report_by_date
from import_2_db.import_financial_data import import_single_financial_by_code
from import_2_db.import_stock_list import import_all_stock_list_data
from models.LSTMTransformer.RandomStockData import RandomStockData

from src.crawl.sync_fincial_report import sync_financial_report
from src.training.fix_standardlize import build_data, fit_target_scaler, fit_feature_scaler
from src.training.parameter import get_data_params
from upload2aws.upload_to_dynamodb import import_2_aws_process

start_date = '20190101'
end_date = '20200101'


def main():
    create_table()
    import_2_aws_process()

    # df, change_list = build_data()
    #
    # fit_feature_scaler(df)
    # fit_target_scaler(change_list)


if __name__ == "__main__":
    main()
