from datetime import timedelta

from data.raw.option_qvix import get_50etf_qvix
from db import get_db_session
from db.option_qvix import get_last_etf_qvix_date, bulk_insert_etf_qvix_data

from import_2_db.utils import default_start_date
import pandas as pd

currency_start_date = default_start_date

def import_etf_qvix():
    """
    导入 50ETF 期权波动率指数 (QVIX) 数据到数据库
    """
    cursor = currency_start_date
    with get_db_session() as db:
        last_date = get_last_etf_qvix_date(db)
        if last_date is not None:
            cursor = last_date + timedelta(days=1)

    etf_qvix_df = get_50etf_qvix()

    # handle all the error here
    if etf_qvix_df is None or etf_qvix_df.empty:
        return

    etf_qvix_df['date'] = pd.to_datetime(etf_qvix_df['date'])
    to_insert = etf_qvix_df[etf_qvix_df["date"] > cursor]

    with get_db_session() as db:
        bulk_insert_etf_qvix_data(db, to_insert)