import pandas as pd

import datetime
from datetime import date, timedelta
import random

from data.data_preprocessing import (
    clean_stock_data,
    clean_index_data,
    clean_currency_exchange_rates,
    clean_financial_data,
)
from db import get_db_session
from db.exchange_rate_daily import get_exchange_rate_by_date_range
from db.sh_index_daily import get_sh_index_daily_by_date_range
from db.stock_daily import get_stock_data_by_date_range
from db.stock_financial_data import get_financial_data_by_date_range
from db.stock_list import get_all_stock_list_data
from db.sz_index_daily import get_sz_index_daily_by_date_range
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def interpolate_financial_data(df: pd.DataFrame, financial_data: pd.DataFrame) -> pd.DataFrame:
    """
    对财务数据进行线性插值，填充到日线数据中。

    Args:
        df (pd.DataFrame): 包含日线数据的 DataFrame，必须包含 'date' 列。
        financial_data (pd.DataFrame): 财务数据 DataFrame，必须包含 'report_date' 列。

    Returns:
        pd.DataFrame: 填充了插值后财务数据的 DataFrame。
    """
    # 将财务数据的 'report_date' 列设置为索引
    financial_data = financial_data.set_index('report_date')

    # 创建一个空 DataFrame 用于存储插值后的财务数据
    interpolated_data = pd.DataFrame(index=df.index, columns=financial_data.columns)

    # 遍历每条日线数据
    for i, row in df.iterrows():
        current_date = row['date']
        current_date = datetime.datetime.strptime(current_date, "%Y%m%d")
        # 找到当前日期所属的季度区间
        try:
            current_report_date = financial_data.index[financial_data.index < current_date][0]
        except IndexError:
            current_report_date = current_date

        quarter_data = financial_data.loc[current_report_date]

        interpolated_data.loc[i] = quarter_data

    merged_df = df.join(interpolated_data.reset_index(drop=True), how='left', lsuffix='_left', rsuffix='_right')
    return merged_df


def get_stock_total_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取指定股票代码的预测数据，包含股票日线数据、财务数据、汇率数据和指数数据。

    Args:
        stock_code (str): 股票代码。
        start_date (str): 开始日期，格式为 'YYYYMMDD'。
        end_date (str): 开始日期，格式为 'YYYYMMDD'。

    Returns:
        pd.DataFrame: 包含所有数据的 DataFrame，如果获取失败则返回 None。
    """
    try:
        # 获取股票日线数据
        with get_db_session() as db:
            stock_data = get_stock_data_by_date_range(db, stock_code, start_date, end_date)
        if stock_data is None:
            return None

        cleaned_stock_data = clean_stock_data(stock_data.copy())

        # 获取财务数据
        with get_db_session() as db:
            fin_data = get_financial_data_by_date_range(db, stock_code, start_date, end_date)
        if fin_data is None:
            return None

        cleaned_financial_data = clean_financial_data(fin_data.copy())

        # 获取汇率数据
        with get_db_session() as db:
            currency_data = get_exchange_rate_by_date_range(db, start_date, end_date)
        if currency_data is None:
            return None
        cleaned_currency_data = clean_currency_exchange_rates(currency_data.copy())

        # 获取指数数据
        with get_db_session() as db:
            sse_index_data = get_sh_index_daily_by_date_range(db, start_date, end_date)
        with get_db_session() as db:
            szse_index_data = get_sz_index_daily_by_date_range(db, start_date, end_date)
        if sse_index_data is None or szse_index_data is None:
            return None
        cleaned_sse_index_data = clean_index_data(sse_index_data.copy())
        cleaned_szse_index_data = clean_index_data(szse_index_data.copy())

        # 合并所有数据
        merged_data = pd.merge(cleaned_stock_data, cleaned_currency_data, on='date', how='left')
        merged_data = pd.merge(merged_data, cleaned_sse_index_data, on='date', how='left')
        merged_data = pd.merge(merged_data, cleaned_szse_index_data, on='date', how='left')
        merged_data = interpolate_financial_data(merged_data, cleaned_financial_data)

        merged_data = merged_data.ffill().bfill()
        final_df = drop_column_reset_type(merged_data)
        return final_df
    except Exception as e:
        print(f"获取股票预测数据时发生错误：{e}")
        return None


def drop_column_reset_type(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 删除不需要的列
    columns_to_remove = ['date', 'stock_code_left', 'stock_code_right']  # 将不需要的列名添加到这里
    df = df.drop(columns=columns_to_remove)

    # 2. 将剩余列转换为 float64 类型
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"无法将列 '{col}' 转换为数字：{e}")
    return df


def get_random_available_date() -> str:
    today = date.today()
    five_years_ago = today - timedelta(days=365 * 5)
    one_year_ago = today - timedelta(days=365)
    date_range = (five_years_ago, one_year_ago)
    random_days = random.randint(date_range[0].toordinal(), date_range[1].toordinal())
    random_date = date.fromordinal(random_days)
    return random_date.strftime("%Y%m%d")


def get_random_code_from_df(df: pd.DataFrame) -> str:
    random_index = random.randint(0, len(df) - 1)
    random_row = df.iloc[random_index]
    random_code = random_row["code"]

    return random_code


def get_random_code() -> str:
    with get_db_session() as db:
        stock_list = get_all_stock_list_data(db)
    return get_random_code_from_df(stock_list)


def get_one_year_later(dt):
    """
    获取指定日期一年后的日期

    Args:
        dt: datetime 对象

    Returns:
        datetime 对象，代表指定日期一年后的日期
    """
    if not isinstance(dt, datetime.datetime):
        raise TypeError("参数 dt 必须为 datetime 对象")

    if dt.month == 2 and dt.day == 29:
        # Check if the current year is a leap year
        if not is_leap_year(dt.year):
            # If it's not a leap year, set the day to March 1st of the following year
            new_year = dt.year + 1
            new_month = 3
            new_day = 1
        else:
            # If it's a leap year, set the day to February 29th of the following year
            new_year = dt.year + 1
            new_month = 2
            new_day = 29
    else:
        # For other dates, simply add one year
        new_year = dt.year + 1
        new_month = dt.month
        new_day = dt.day

    return dt.replace(year=new_year, month=new_month, day=new_day)


def is_leap_year(year):
    """
    判断给定年份是否为闰年

    Args:
        year: 年份

    Returns:
        True 如果给定年份是闰年，False 否则
    """
    if (year % 4 == 0) and (year % 100 != 0) or (year % 400 == 0):
        return True
    else:
        return False


def get_random_full_data() -> pd.DataFrame:
    result = None
    while result is None or len(result) <= 200:
        code = get_random_code()
        start_date = get_random_available_date()
        end_date = get_one_year_later(datetime.datetime.strptime(start_date, "%Y%m%d"))
        result = get_stock_total_data(stock_code=code, start_date=start_date, end_date=end_date.strftime("%Y%m%d"))
    return result


def get_random_valid_data() -> pd.DataFrame:
    df = get_random_full_data()

    cols_not_scale = ['stock_close', 'stock_change_percent']

    cols_to_scale = [x for x in df.columns if x not in cols_not_scale]

    # 从 DataFrame 中提取需要处理的列
    data_to_scale = df[cols_to_scale]

    # 使用 StandardScaler 对选定的列进行标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_scale)

    # 将标准化后的数据更新回原始 DataFrame
    df[cols_to_scale] = scaled_data

    return df

# stock_data = get_stock_total_data(stock_code='600000', start_date='20220101', n_days=200)
#
# removed_data = drop_columns_and_reset_index(stock_data)
# print(removed_data)
