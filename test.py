from data.data_merging.merge_data_v2 import get_random_data_all, get_random_v2_data
from db import create_table, get_db_session
from import_2_db import import_shibor_rate
from import_2_db.import_us_index import import_us_index
from import_2_db.import_vix_daily import import_etf_qvix
from models.LSTMTransformer.WeightedSumLoss import get_weights

if __name__ == "__main__":
    create_table()
    import_shibor_rate()