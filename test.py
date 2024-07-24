from data.data_merging.merge_data_v2 import get_random_data_all
from data.raw.interbank_rates import get_shibor_rate
from db import create_table, get_db_session
from db.shibor_rates import get_shibor_rate_by_date_range
from import_2_db.import_shibor_rate import import_shibor_rate

if __name__ == "__main__":
    create_table()
    df = get_random_data_all()
    print(df)

