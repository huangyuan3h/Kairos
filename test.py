from data.data_merging import get_random_valid_data
from data.data_merging.column_choose import filter_features_correlation
from data.data_merging.merge_data import get_random_total_data


def main():
    stock_list = get_random_total_data()
    df, keep, drop = filter_features_correlation(stock_list, 0.95)
    print(df, keep, drop)


if __name__ == "__main__":
    main()
