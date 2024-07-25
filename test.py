from data.data_merging.merge_data_v2 import get_random_data_all, get_random_v2_data
from db import create_table, get_db_session

if __name__ == "__main__":
    create_table()
    get_random_data_all()
