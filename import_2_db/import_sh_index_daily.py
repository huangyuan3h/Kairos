from data.raw import get_sse_composite_index
from db import get_db_session
from db.sh_index_daily import get_last_index_daily_date, bulk_insert_index_daily_data
from import_2_db.utils import default_start_date, get_next_day
from datetime import datetime
import pandas as pd

currency_start_date = default_start_date


def import_sh_index_daily():
    cursor = currency_start_date
    with get_db_session() as db:
        last_date = get_last_index_daily_date(db)
        if last_date is not None:
            cursor = get_next_day(last_date)

        # 计算结束日期
    end_date_obj = datetime.now()
    end_date = end_date_obj.strftime('%Y%m%d')

    sh_index_daily_df = get_sse_composite_index(cursor.strftime('%Y%m%d'), end_date)

    # handle all the error here
    if sh_index_daily_df.empty or sh_index_daily_df is None:
        return

    sh_index_daily_df['date'] = pd.to_datetime(sh_index_daily_df['date']).dt.strftime('%Y-%m-%d')

    with get_db_session() as db:
        bulk_insert_index_daily_data(db, sh_index_daily_df)
