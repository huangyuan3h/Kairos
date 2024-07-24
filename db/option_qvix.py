from sqlalchemy import Column, Date, Float

from sqlalchemy import select, func
from sqlalchemy.orm import Session
import pandas as pd
from datetime import datetime

from db import Base


class ETFQVIX(Base):
    __tablename__ = 'etf_qvix'

    date = Column(Date, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)


def bulk_insert_etf_qvix_data(db: Session, df: pd.DataFrame):
    """
    将 Pandas DataFrame 中的 50ETF 期权波动率指数 (QVIX) 数据批量插入到数据库

    Args:
        db (Session): 数据库会话对象
        df (pd.DataFrame): 包含  50ETF 期权波动率指数 (QVIX) 数据的 DataFrame
    """
    data_list = df.to_dict(orient='records')
    for data in data_list:
        etf_qvix_data = ETFQVIX(**data)
        db.add(etf_qvix_data)
    db.commit()


def get_last_etf_qvix_date(db: Session) -> datetime:
    """
    查询最近一条 50ETF 期权波动率指数 (QVIX) 记录的时间

    Args:
        db (Session): 数据库会话对象

    Returns:
        datetime: 最近一条记录的时间，如果未找到则返回 None
    """

    stmt = select(func.max(ETFQVIX.date))
    result = db.execute(stmt).scalar()

    return result


def get_etf_qvix_by_date_range(db: Session, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取指定日期范围内的 50ETF 期权波动率指数 (QVIX) 数据

    Args:
        db (Session): 数据库会话
        start_date (str): 开始日期，格式为 'YYYYMMDD'
        end_date (str): 结束日期，格式为 'YYYYMMDD'

    Returns:
        pd.DataFrame: 包含  50ETF 期权波动率指数 (QVIX) 数据的 DataFrame
    """
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')

    stmt = select("*").filter(
        ETFQVIX.date >= start_date,
        ETFQVIX.date <= end_date
    ).order_by(ETFQVIX.date.asc())

    result = db.execute(stmt).all()

    df = pd.DataFrame(result, columns=[col.key for col in ETFQVIX.__table__.columns])
    return df
