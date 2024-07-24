from sqlalchemy import Column, Integer, Float, Date, func
from sqlalchemy.orm import Session
import pandas as pd
from sqlalchemy import select
from datetime import datetime
from db.database import Base


class ExchangeRate(Base):
    """汇率数据模型"""
    __tablename__ = "exchange_rate"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    USD_CNY = Column(Float)
    EUR_CNY = Column(Float)


def bulk_insert_exchange_rate_data(db: Session, df: pd.DataFrame):
    """
    将 Pandas DataFrame 中的汇率数据批量插入到数据库

    Args:
        db (Session): 数据库会话对象
        df (pd.DataFrame): 包含汇率数据的 DataFrame

    eg：
        with get_db_session() as db:
            bulk_insert_exchange_rate_data(db, df)
    """
    data_list = df.to_dict(orient='records')
    for data in data_list:
        db_exchange_rate = ExchangeRate(**data)
        db.add(db_exchange_rate)
    db.commit()


def get_exchange_rate_by_date_range(db: Session, start_date: str, end_date: str) -> pd.DataFrame:
    """
    查询一段时间的汇率数据

    Args:
        db (Session): 数据库会话对象
        start_date (str): 开始日期，格式为 'YYYYMMDD'
        end_date (str): 结束日期，格式为 'YYYYMMDD'

    Returns:
        pd.DataFrame: 包含汇率数据的 DataFrame，如果未找到则返回 None
                      DataFrame 包含以下列：
                          - date: 日期 (object)
                          - USD_CNY: 美元兑人民币汇率 (float64)
                          - EUR_CNY: 欧元兑人民币汇率 (float64)
    """
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')

    stmt = select("*").where(
        ExchangeRate.date >= start_date,
        ExchangeRate.date <= end_date
    ).order_by(ExchangeRate.date.asc())

    result = db.execute(stmt).all()
    df = pd.DataFrame(result, columns=[column.key for column in ExchangeRate.__table__.columns])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df = df.drop("id", axis=1)
    return df


def get_last_exchange_rate_date(db: Session) -> datetime:
    """
    查询数据库中最近一条汇率数据记录的日期

    Args:
        db (Session): 数据库会话对象

    Returns:
        datetime: 最近一条记录的日期，如果未找到则返回 None
    """

    stmt = select(func.max(ExchangeRate.date))
    result = db.execute(stmt).scalar()
    return result
