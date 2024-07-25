import pandas as pd


def clean_us_index_data(us_index: pd.DataFrame) -> pd.DataFrame:
    us_index['date'] = pd.to_datetime(us_index['date'])
    us_index['date'] = us_index['date'].dt.strftime('%Y%m%d')

    return us_index