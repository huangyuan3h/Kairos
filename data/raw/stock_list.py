import akshare as ak
import pandas as pd
import random

column_mapping = {
    '序号': 'index',
    '代码': 'code',
    '名称': 'name',
    '最新价': 'latest_price',
    '涨跌幅': 'price_change_percent',
    '涨跌额': 'price_change',
    '成交量': 'volume',
    '成交额': 'turnover',
    '振幅': 'amplitude',
    '最高': 'high',
    '最低': 'low',
    '今开': 'open',
    '昨收': 'previous_close',
    '量比': 'volume_ratio',
    '换手率': 'turnover_rate',
    '市盈率-动态': 'pe_ratio',
    '市净率': 'pb_ratio',
    '总市值': 'total_market_cap',
    '流通市值': 'circulating_market_cap',
    '涨速': 'price_change_rate',
    '5分钟涨跌': '5_min_change',
    '60日涨跌幅': '60_day_change',
    '年初至今涨跌幅': 'ytd_change'
}


def get_sh_a_stock_list() -> pd.DataFrame:
    """
    获取上海交易所 A 股列表，列名转换为英文。

    Returns:
        pandas.DataFrame: 包含上海交易所 A 股信息的 DataFrame，列名已转换为英文。
    """
    sh_stock_list = ak.stock_sh_a_spot_em()
    sh_stock_list.rename(columns=column_mapping, inplace=True)
    return sh_stock_list


def get_sz_a_stock_list() -> pd.DataFrame:
    """
    获取深圳交易所 A 股列表，列名转换为英文。

    Returns:
        pandas.DataFrame: 包含深圳交易所 A 股信息的 DataFrame，列名已转换为英文。
    """
    sz_stock_list = ak.stock_sz_a_spot_em()
    sz_stock_list.rename(columns=column_mapping, inplace=True)
    return sz_stock_list


def get_random_code_from_df(df: pd.DataFrame) -> str:
    random_index = random.randint(0, len(df) - 1)
    random_row = df.iloc[random_index]
    random_code = random_row["code"]

    return random_code


def get_sh_random_code() -> str:
    sh_list = get_sh_a_stock_list()
    return get_random_code_from_df(sh_list)


def get_sz_random_code() -> str:
    sz_list = get_sz_a_stock_list()
    return get_random_code_from_df(sz_list)


def random_bool():
    return random.random() < 0.5


def get_random_code() -> str:
    if random_bool():
        return get_sh_random_code()
    else:
        return get_sz_random_code()
