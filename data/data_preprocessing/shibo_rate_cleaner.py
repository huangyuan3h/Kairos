import pandas as pd


def clean_shibo_rate_data(shibo_rate_data: pd.DataFrame) -> pd.DataFrame:
    shibo_rate_data['date'] = pd.to_datetime(shibo_rate_data['date'])
    shibo_rate_data['date'] = shibo_rate_data['date'].dt.strftime('%Y%m%d')

    return shibo_rate_data