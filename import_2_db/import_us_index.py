from datetime import timedelta

import pandas as pd

from data.raw import get_us_stock_index_data
from db import get_db_session
from db.us_index_daily import get_last_us_index_daily_date, bulk_insert_us_index_daily_data
from import_2_db.utils import default_start_date


def calculate_daily_increase(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 DataFrame 中每日的涨幅。

    Args:
        df: 包含股票或指数数据的 DataFrame，必须包含 'close' 列，表示每日收盘价。

    Returns:
        pd.DataFrame: 包含每日涨幅的 DataFrame，新增一列 'increase'，表示涨幅。
    """
    df['increase'] = df['close'].pct_change() * 100
    df['increase'].fillna(0, inplace=True)
    return df


def combine_index_data(ixic_df: pd.DataFrame, dji_df: pd.DataFrame,
                       inx_df: pd.DataFrame, ndx_df: pd.DataFrame) -> pd.DataFrame:
    """
    合并四个美股指数的DataFrame，并计算每日涨幅。

    Args:
        ixic_df: 纳斯达克指数数据 DataFrame。
        dji_df: 道琼斯指数数据 DataFrame。
        inx_df: 标普500指数数据 DataFrame。
        ndx_df: 纳斯达克100指数数据 DataFrame。

    Returns:
        pd.DataFrame: 包含日期和四个指数每日涨幅的 DataFrame，如果合并失败则返回 None。
                      DataFrame 包含以下列：
                          - date: 日期
                          - IXIC_increase: 纳斯达克指数涨幅
                          - DJI_increase: 道琼斯指数涨幅
                          - INX_increase: 标普500指数涨幅
                          - NDX_increase: 纳斯达克100指数涨幅
    """

    try:
        # 先计算每日涨幅
        ixic_df = calculate_daily_increase(ixic_df)
        dji_df = calculate_daily_increase(dji_df)
        inx_df = calculate_daily_increase(inx_df)
        ndx_df = calculate_daily_increase(ndx_df)

        # 使用日期作为合并的键
        combined_df = pd.merge(ixic_df[['date', 'increase']], dji_df[['date', 'increase']], on='date',
                               suffixes=('_IXIC', '_DJI'))
        combined_df = pd.merge(combined_df, inx_df[['date', 'increase']], on='date')
        combined_df = pd.merge(combined_df, ndx_df[['date', 'increase']], on='date', suffixes=('_INX', '_NDX'))

        # 选择需要的列并重命名
        combined_df.columns = ['date', 'IXIC_increase', 'DJI_increase', 'INX_increase', 'NDX_increase']

        return combined_df
    except Exception as e:
        print(f"合并美股指数数据时发生错误：{e}")
        return None


currency_start_date = default_start_date


def import_us_index():
    cursor = currency_start_date.date()
    with get_db_session() as db:
        last_date = get_last_us_index_daily_date(db)
        if last_date is not None:
            cursor = last_date + timedelta(days=1)

    # 获取四个美股指数数据
    ixic_df = get_us_stock_index_data(symbol=".IXIC")
    dji_df = get_us_stock_index_data(symbol=".DJI")
    inx_df = get_us_stock_index_data(symbol=".INX")
    ndx_df = get_us_stock_index_data(symbol=".NDX")

    # 处理数据获取错误
    if any(df is None or df.empty for df in [ixic_df, dji_df, inx_df, ndx_df]):
        return

    # 合并美股指数数据并计算涨跌幅
    combined_df = combine_index_data(ixic_df, dji_df, inx_df, ndx_df)

    # 处理合并错误
    if combined_df is None or combined_df.empty:
        return

    # 转换日期格式
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df['date'] = combined_df['date'].dt.date

    # 筛选需要插入的数据
    to_insert = combined_df[combined_df["date"] >= cursor]

    # 批量插入数据到数据库
    with get_db_session() as db:
        bulk_insert_us_index_daily_data(db, to_insert)
