from data.data_merging.merge_data_v2 import get_random_data_all, get_random_v2_data
from data.raw import get_us_stock_index_data
from data.raw.interbank_rates import get_shibor_rate
from db import create_table, get_db_session
from db.shibor_rates import get_shibor_rate_by_date_range
from import_2_db.import_shibor_rate import import_shibor_rate
import akshare as ak

from import_2_db.import_us_index import import_us_index
from import_2_db.import_vix_daily import import_etf_qvix
from models.standardize.FeatureStandardScaler import FeatureStandardScaler

if __name__ == "__main__":
    create_table()
    get_random_data_all()
