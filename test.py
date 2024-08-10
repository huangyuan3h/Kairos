from data.data_merging.merge_data_v2 import get_random_v2_data
from operation.get_operation_data import get_xy_operation_data_from_df
from src.crawl.sync_daily_all import sync_daily_all

if __name__ == "__main__":
    df = get_random_v2_data()

    df = df.tail(70)
    get_xy_operation_data_from_df(df, [i for i in range(77)], 1)
