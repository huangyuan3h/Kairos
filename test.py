import datetime

from data.data_merging import get_stock_total_data, get_random_valid_data
from data.data_preprocessing import merge_financial_data
from data.raw import get_sh_a_stock_list

from db import create_table, get_db_session
from db.predict_report import get_predict_report_by_date
from db.stock_list import get_all_stock_list_data
from import_2_db.import_financial_data import import_single_financial_by_code
from import_2_db.import_stock_list import import_all_stock_list_data
from models.LSTMTransformer.RandomStockData import RandomStockData

from src.crawl.sync_fincial_report import sync_financial_report
from src.training.fix_standardlize import build_data, fit_target_scaler, fit_feature_scaler
from src.training.parameter import get_data_params
from upload2aws.upload_to_dynamodb import import_2_aws_process
from utils.export_df_2_json import df_to_json_with_key_value_pairs

start_date = '20190101'
end_date = '20200101'


def main():
    with get_db_session() as db:
        stock_list = get_all_stock_list_data(db)

    df = df_to_json_with_key_value_pairs(stock_list, "code", "name","stock_config.json")
    print(df)


if __name__ == "__main__":
    main()
