from datetime import datetime

from sqlalchemy import Column, Date, Float
from sqlalchemy import select, func
from sqlalchemy.orm import Session
import pandas as pd

from db import Base


class ShiborRate(Base):
    __tablename__ = 'shibor_rate'

    date = Column(Date, primary_key=True)
    rate = Column(Float)
    change = Column(Float)


def bulk_insert_shibor_rate_data(db: Session, df: pd.DataFrame):
    """
    将 Pandas DataFrame 中的银行间同业拆借利率（Shibor）数据批量插入到数据库

    Args:
        db (Session): 数据库会话对象
        df (pd.DataFrame): 包含 Shibor 利率数据的 DataFrame
    """
    data_list = df.to_dict(orient='records')
    for data in data_list:
        shibor_rate_data = ShiborRate(**data)
        db.add(shibor_rate_data)
    db.commit()


def get_last_shibor_rate_date(db: Session) -> datetime:
    """
    查询最近一条银行间同业拆借利率（Shibor）记录的时间

    Args:
        db (Session): 数据库会话对象

    Returns:
        datetime: 最近一条记录的时间，如果未找到则返回 None
    """

    stmt = select(func.max(ShiborRate.date))
    result = db.execute(stmt).scalar()

    return result


def get_shibor_rate_by_date_range(db: Session, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取指定日期范围内的银行间同业拆借利率（Shibor）数据

    Args:
        db (Session): 数据库会话
        start_date (str): 开始日期，格式为 'YYYYMMDD'
        end_date (str): 结束日期，格式为 'YYYYMMDD'

    Returns:
        pd.DataFrame: 包含 Shibor 利率数据的 DataFrame
    """
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')

    stmt = select("*").filter(
        ShiborRate.date >= start_date,
        ShiborRate.date <= end_date
    ).order_by(ShiborRate.date.asc())

    result = db.execute(stmt).all()

    df = pd.DataFrame(result, columns=[col.key for col in ShiborRate.__table__.columns])
    return df
