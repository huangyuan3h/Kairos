from data.raw.interbank_rates import get_shibor_rate
from db import create_table, get_db_session
from db.shibor_rates import get_shibor_rate_by_date_range
from import_2_db.import_shibor_rate import import_shibor_rate

if __name__ == "__main__":
    create_table()
    with get_db_session() as db:
        df = get_shibor_rate_by_date_range(db, "20240101", "20240723")
    print(df)

