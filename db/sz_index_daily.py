from db.database import Base
from sqlalchemy import Column, Integer, Float, Date,func
from sqlalchemy.orm import Session
import pandas as pd
from sqlalchemy import select
from datetime import datetime


class SZIndexDaily(Base):
    __tablename__ = 'sz_index_daily'

    date = Column(Date, primary_key=True)
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Integer)
    amount = Column(Float)
    amplitude = Column(Float)
    change_percent = Column(Float)
    change = Column(Float)
    turnover_rate = Column(Float)


def bulk_insert_index_daily_data(db: Session, df: pd.DataFrame):
    """
    将 Pandas DataFrame 中的数据批量插入到数据库

    Args:
        db (Session): 数据库会话对象
        df (pd.DataFrame): 包含股票数据的 DataFrame

    eg：
        with get_db_session() as db:
            bulk_insert_index_daily_data(db, df)
    """
    data_list = df.to_dict(orient='records')
    for data in data_list:
        sz_data = SZIndexDaily(**data)
        db.add(sz_data)
    db.commit()


def get_last_index_daily_date(db: Session) -> datetime:
    """
    查询最近一条记录的时间

    Args:
        db (Session): 数据库会话对象

    Returns:
        datetime: 最近一条记录的时间，如果未找到则返回 None
    """

    stmt = select(func.max(SZIndexDaily.date))
    result = db.execute(stmt).scalar()

    return result


def get_sz_index_daily_by_date_range(db: Session, start_date: str, end_date: str) -> pd.DataFrame:
    """获取指定日期范围内的股票数据

    Args:
        db (Session): 数据库会话
        start_date (str): 开始日期，格式为 'YYYYMMDD'
        end_date (str): 结束日期，格式为 'YYYYMMDD'

    Returns:
        pd.DataFrame: 包含股票数据的DataFrame
    """
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')

    stmt = select("*").filter(
        SZIndexDaily.date >= start_date,
        SZIndexDaily.date <= end_date
    ).order_by(SZIndexDaily.date.desc())

    result = db.execute(stmt).all()

    df = pd.DataFrame(result, columns=[col.key for col in SZIndexDaily.__table__.columns])
    return df