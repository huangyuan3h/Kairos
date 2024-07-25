from db.database import Base
from sqlalchemy import Column, Float, Date, func
from sqlalchemy.orm import Session
import pandas as pd
from sqlalchemy import select
from datetime import datetime


class USIndexDaily(Base):
    __tablename__ = 'us_index_daily'

    date = Column(Date, primary_key=True)
    IXIC_increase = Column(Float)
    DJI_increase = Column(Float)
    INX_increase = Column(Float)
    NDX_increase = Column(Float)


def bulk_insert_us_index_daily_data(db: Session, df: pd.DataFrame):
    """
    将 Pandas DataFrame 中的数据批量插入到数据库

    Args:
        db (Session): 数据库会话对象
        df (pd.DataFrame): 包含美股指数数据的 DataFrame

    eg：
        with get_db_session() as db:
            bulk_insert_us_index_daily_data(db, df)
    """
    data_list = df.to_dict(orient='records')
    for data in data_list:
        db_data = USIndexDaily(**data)
        db.add(db_data)
    db.commit()


def get_last_us_index_daily_date(db: Session) -> datetime:
    """
    查询最近一条美股指数记录的时间

    Args:
        db (Session): 数据库会话对象

    Returns:
        datetime: 最近一条美股指数记录的时间，如果未找到则返回 None
    """

    stmt = select(func.max(USIndexDaily.date))
    result = db.execute(stmt).scalar()

    return result


def get_us_index_daily_by_date_range(db: Session, start_date: str, end_date: str) -> pd.DataFrame:
    """获取指定日期范围内的美股指数数据

    Args:
        db (Session): 数据库会话
        start_date (str): 开始日期，格式为 'YYYYMMDD'
        end_date (str): 结束日期，格式为 'YYYYMMDD'

    Returns:
        pd.DataFrame: 包含美股指数数据的DataFrame
    """
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')

    stmt = select("*").filter(
        USIndexDaily.date >= start_date,
        USIndexDaily.date <= end_date
    ).order_by(USIndexDaily.date.asc())

    result = db.execute(stmt).all()

    df = pd.DataFrame(result, columns=[col.key for col in USIndexDaily.__table__.columns])
    return df