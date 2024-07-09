from data.data_merging import get_stock_total_data, get_random_valid_data
from data.data_preprocessing import merge_financial_data
from data.raw import get_sh_a_stock_list

from db import create_table
from import_2_db.import_financial_data import import_single_financial_by_code
from import_2_db.import_stock_list import import_all_stock_list_data
from models.LSTMTransformer.RandomStockData import RandomStockData

from src.crawl.sync_fincial_report import sync_financial_report
from src.training.parameter import get_data_params

start_date = '20190101'
end_date = '20200101'



def main():
    create_table()
    # feature_columns, target_column = get_data_params()
    # random = RandomStockData(feature_columns, target_column)
    # x,y = random.get_data()
    get_stock_total_data("600776","20191230","20201229")

    # print(x,y)



if __name__ == "__main__":
    main()
