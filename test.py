from data.data_merging import get_stock_total_data, get_random_valid_data
from data.data_preprocessing import merge_financial_data
from data.raw import get_sh_a_stock_list

from db import create_table
from import_2_db.import_financial_data import import_single_financial_by_code
from import_2_db.import_stock_list import import_all_stock_list_data
from models.LSTMTransformer.RandomStockData import RandomStockData

from src.crawl.sync_fincial_report import sync_financial_report
from src.training.fix_standardlize import build_df, fit_standard_scaler
from src.training.parameter import get_data_params

start_date = '20190101'
end_date = '20200101'


def main():
    create_table()
    df = build_df()
    fit_standard_scaler(df)


if __name__ == "__main__":
    main()
