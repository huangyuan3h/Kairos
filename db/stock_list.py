
from sqlalchemy import Column, Integer, Float, String, func, Date
from db import Base
from sqlalchemy.orm import Session
import pandas as pd
from datetime import datetime
from sqlalchemy import select

pd.set_option('future.no_silent_downcasting', True)


class StockListData(Base):
    __tablename__ = 'stock_list_data'

    code = Column(String(10), nullable=False, primary_key=True)
    name = Column(String(30), nullable=False)
    latest_price = Column(Float)
    price_change_percent = Column(Float)
    price_change = Column(Float)
    volume = Column(Float)
    turnover = Column(Float)
    amplitude = Column(Float)
    high = Column(Float)
    low = Column(Float)
    open = Column(Float)
    previous_close = Column(Float)
    volume_ratio = Column(Float)
    turnover_rate = Column(Float)
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    total_market_cap = Column(Float)
    circulating_market_cap = Column(Float)
    price_change_rate = Column(Float)
    five_min_change = Column(Float)
    sixty_day_change = Column(Float)
    ytd_change = Column(Float)


def bulk_insert_stock_daily_data(db: Session, df: pd.DataFrame):
    """
    This function takes a Pandas DataFrame and bulk inserts the data into the stock_data table.

    Args:
        db (Session): SQLAlchemy database session.
        df (pd.DataFrame): DataFrame containing stock data.
    """
    df =df.ffill().bfill().fillna(value=0)
    df.rename(
        columns={'5_min_change': 'five_min_change', '60_day_change': 'sixty_day_change', 'ytd_change': 'ytd_change'},
        inplace=True)
    db.bulk_insert_mappings(StockListData, df.to_dict(orient="records"))
    db.commit()


def get_all_stock_list_data(db: Session) -> pd.DataFrame:
    """
    Fetches all data from the stock_data table and returns it as a Pandas DataFrame.

    Args:
        db (Session): SQLAlchemy database session.

    Returns:
        pd.DataFrame: DataFrame containing all data from the stock_data table.
    """
    results = db.query(StockListData).all()
    df = pd.DataFrame([row.__dict__ for row in results])
    df = df.drop(columns=['_sa_instance_state'])
    return df
