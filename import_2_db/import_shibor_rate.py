from datetime import timedelta
from data.raw.interbank_rates import get_shibor_rate
from db import get_db_session
from db.shibor_rates import get_last_shibor_rate_date, bulk_insert_shibor_rate_data

from import_2_db.utils import default_start_date

currency_start_date = default_start_date


def import_shibor_rate():
    """
    导入银行间同业拆借利率（Shibor）数据到数据库
    """
    cursor = currency_start_date
    with get_db_session() as db:
        last_date = get_last_shibor_rate_date(db)
        if last_date is not None:
            cursor = last_date + timedelta(days=1)

    shibor_rate_df = get_shibor_rate()

    # handle all the error here
    if shibor_rate_df is None or shibor_rate_df.empty:
        return

    to_insert = shibor_rate_df[shibor_rate_df["date"] > cursor.date()]

    with get_db_session() as db:
        bulk_insert_shibor_rate_data(db, to_insert)
