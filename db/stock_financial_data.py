from sqlalchemy import Column, Integer, Float, String, func, Date
from db import Base
from sqlalchemy.orm import Session
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import select

pd.set_option('future.no_silent_downcasting', True)


class FinancialData(Base):
    __tablename__ = 'financial_data'

    id = Column(Integer, primary_key=True)
    report_date = Column(Date, nullable=False)
    stock_code = Column(String(10), nullable=False)
    revenue = Column(Float)
    total_operating_cost = Column(Float)
    operating_profit = Column(Float)
    gross_profit = Column(Float)
    net_profit = Column(Float)
    basic_eps = Column(Float)
    rd_expenses = Column(Float)
    interest_income = Column(Float)
    interest_expense = Column(Float)
    investment_income = Column(Float)
    cash_and_equivalents = Column(Float)
    accounts_receivable = Column(Float)
    inventory = Column(Float)
    net_fixed_assets = Column(Float)
    short_term_borrowings = Column(Float)
    long_term_borrowings = Column(Float)
    total_equity = Column(Float)
    total_assets = Column(Float)
    total_liabilities = Column(Float)
    net_cash_from_operating = Column(Float)
    net_cash_from_investing = Column(Float)
    net_cash_from_financing = Column(Float)
    net_increase_in_cce = Column(Float)
    end_cash_and_cash_equivalents = Column(Float)

    def __repr__(self):
        return f"<FinancialData(id={self.id}, stock_code='{self.stock_code}', report_date='{self.report_date}')>"


def bulk_insert_financial_data(db: Session, df: pd.DataFrame):
    """
    将 Pandas DataFrame 中的财务数据批量插入到数据库

    Args:
        db (Session): 数据库会话对象
        df (pd.DataFrame): 包含财务数据的 DataFrame

    eg：
        with get_db_session() as db:
            bulk_insert_financial_data(db, df)
    """
    df = df.bfill().ffill().fillna(value=0)

    data_list = df.to_dict(orient='records')
    for data in data_list:
        db_financial_data = FinancialData(**data)
        db.add(db_financial_data)
    db.commit()


def get_90_days_before(date: datetime) -> datetime:
    """
    获取指定日期的90天前的日期。

    Args:
        date (datetime): 指定的日期。

    Returns:
        datetime: 90天前的日期。
    """
    return date - timedelta(days=90)


def get_90_days_after(date: datetime) -> datetime:
    """
    获取指定日期的90天后的日期。

    Args:
        date (datetime): 指定的日期。

    Returns:
        datetime: 90天后的日期。
    """
    return date + timedelta(days=90)


def get_financial_data_by_date_range(db: Session, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
  根据日期范围获取股票的财务数据

  Args:
      db (Session): 数据库会话对象
      stock_code (str): 股票代码
      start_date (str): 开始日期，格式为 'YYYYMMDD'
      end_date (str): 结束日期，格式为 'YYYYMMDD'

  Returns:
      pd.DataFrame: 包含财务数据的 DataFrame
  """
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')

    stmt = select("*").filter(
        FinancialData.stock_code == stock_code,
        FinancialData.report_date >= get_90_days_before(start_date),
        FinancialData.report_date <= get_90_days_after(end_date)
    ).order_by(FinancialData.report_date.desc())

    result = db.execute(stmt).all()

    df = pd.DataFrame(result, columns=[col.key for col in FinancialData.__table__.columns])
    if '_sa_instance_state' in df.columns:
        df.drop('_sa_instance_state', axis=1, inplace=True)
    df = df.drop("id", axis=1)
    return df


def get_last_index_daily_date(db: Session, stock_code: str) -> datetime:
    """
    查询最近一条记录的时间

    Args:
        db (Session): 数据库会话对象

    Returns:
        datetime: 最近一条记录的时间，如果未找到则返回 None
    """

    stmt = select(func.max(FinancialData.report_date)).where(
        FinancialData.stock_code == stock_code
    )
    result = db.execute(stmt).scalar()

    return result
