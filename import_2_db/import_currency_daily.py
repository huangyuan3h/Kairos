from data.raw import get_currency_exchange_rates
from db.database import get_db_session
from db.exchange_rate_daily import bulk_insert_exchange_rate_data, get_last_exchange_rate_date
from datetime import datetime

import pandas as pd

from import_2_db.utils import get_next_day

currency_start_date = datetime(2019, 1, 1)


def import_exchange_rates():
    cursor = currency_start_date
    with get_db_session() as db:
        last_date = get_last_exchange_rate_date(db)
        if last_date is not None:
            cursor = get_next_day(last_date)

    # 计算结束日期
    end_date_obj = datetime.now()
    end_date = end_date_obj.strftime('%Y%m%d')

    exchange_rate_df = get_currency_exchange_rates(cursor.strftime('%Y%m%d'), end_date)

    # handle all the error here
    if exchange_rate_df.empty or exchange_rate_df is None:
        return

    exchange_rate_df['date'] = pd.to_datetime(exchange_rate_df['date']).dt.strftime('%Y-%m-%d')

    with get_db_session() as db:
        bulk_insert_exchange_rate_data(db, exchange_rate_df)
