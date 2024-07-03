from datetime import datetime

from sqlalchemy import Column, Integer, Float, Date, String, func
from typing import List

from db.database import Base
from sqlalchemy.orm import Session
import pandas as pd
from sqlalchemy import select


class StockData(Base):
    """股票日线数据模型"""
    __tablename__ = "stock_daily_data"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    stock_code = Column(String(10), nullable=False)
    stock_open = Column(Float)
    stock_close = Column(Float)
    stock_high = Column(Float)
    stock_low = Column(Float)
    stock_volume = Column(Integer)
    stock_amount = Column(Float)
    stock_amplitude = Column(Float)
    stock_change_percent = Column(Float)
    stock_change = Column(Float)
    stock_turnover_rate = Column(Float)


# StockData 增删改查操作

def create_stock_daily_data(db: Session, stock_daily_data: dict) -> StockData:
    """
    创建新的股票数据记录
    """
    db_stock_daily_data = StockData(**stock_daily_data)
    db.add(db_stock_daily_data)
    db.commit()
    db.refresh(db_stock_daily_data)
    return db_stock_daily_data


def get_stock_data_by_date_range(db: Session, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    查询一段时间的股票数据

    Args:
        db (Session): 数据库会话对象
        stock_code (str): 股票代码
        start_date (str): 开始日期，格式为 'YYYYMMDD'
        end_date (str): 结束日期，格式为 'YYYYMMDD'

    Returns:
        pd.DataFrame: 包含股票数据的 DataFrame，如果未找到则返回 None
                      DataFrame 包含以下列：
                          - date: 日期 (object)
                          - stock_code: 股票代码 (object)
                          - stock_open: 开盘价 (float64)
                          - stock_close: 收盘价 (float64)
                          - stock_high: 最高价 (float64)
                          - stock_low: 最低价 (float64)
                          - stock_volume: 成交量 (int64)
                          - stock_amount: 成交额 (float64)
                          - stock_amplitude: 振幅 (float64)
                          - stock_change_percent: 涨跌幅 (float64)
                          - stock_change: 涨跌额 (float64)
                          - stock_turnover_rate: 换手率 (float64)
    """

    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')

    # 使用 SQLAlchemy Core API 构建查询语句
    stmt = select(StockData).where(
        StockData.stock_code == stock_code,
        StockData.date >= start_date,
        StockData.date <= end_date
    ).order_by(StockData.report_date.desc())

    # 执行查询并将结果转换为 DataFrame
    result = db.execute(stmt).all()
    df = pd.DataFrame(result, columns=[column.key for column in StockData.__table__.columns])

    # 转换数据类型
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['stock_volume'] = df['stock_volume'].astype('int64')

    return df


def get_stock_daily_data_by_id(db: Session, stock_daily_data_id: int) -> StockData:
    """
    根据 ID 获取股票数据
    """
    return db.query(StockData).filter(StockData.id == stock_daily_data_id).first()


def get_stock_daily_data(db: Session, skip: int = 0, limit: int = 100) -> List[StockData]:
    """
    获取股票数据列表
    """
    return db.query(StockData).offset(skip).limit(limit).all()


def delete_stock_daily_data(db: Session, stock_daily_data_id: int):
    """
    删除股票数据
    """
    db.query(StockData).filter(StockData.id == stock_daily_data_id).delete()
    db.commit()


def bulk_insert_stock_daily_data(db: Session, df: pd.DataFrame):
    """
    将 Pandas DataFrame 中的数据批量插入到数据库

    Args:
        db (Session): 数据库会话对象
        df (pd.DataFrame): 包含股票数据的 DataFrame

    eg：
        with get_db_session() as db:
            bulk_insert_stock_daily_data(db, df)
    """
    data_list = df.to_dict(orient='records')
    for data in data_list:
        db_stock_daily_data = StockData(**data)
        db.add(db_stock_daily_data)
    db.commit()


def get_last_stock_data_date(db: Session, stock_code: str) -> datetime:
    """
    查询一个股票的最近一条记录的时间

    Args:
        db (Session): 数据库会话对象
        stock_code (str): 股票代码

    Returns:
        datetime: 最近一条记录的时间，如果未找到则返回 None
    """

    stmt = select(func.max(StockData.date)).where(
        StockData.stock_code == stock_code
    )
    result = db.execute(stmt).scalar()

    return result
