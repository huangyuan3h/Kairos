from data.data_preprocessing import merge_financial_data

from db import create_table

from src.crawl.sync_fincial_report import sync_financial_report

start_date = '20190101'


def main():
    create_table()
    # import_sz_index_daily()
    # code = '300176'
    #
    # import_single_financial_by_code(code)
    sync_financial_report()



if __name__ == "__main__":
    main()
