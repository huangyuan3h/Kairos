import pandas as pd


def clean_qvix_data(vix_data: pd.DataFrame) -> pd.DataFrame:
    vix_data['date'] = pd.to_datetime(vix_data['date'])
    vix_data['date'] = vix_data['date'].dt.strftime('%Y%m%d')

    return vix_data