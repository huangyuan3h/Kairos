from sqlalchemy import Column, Integer, String, DateTime, Float

from db import Base
from sqlalchemy.orm import Session
import pandas as pd
from datetime import datetime


class FinancialData(Base):
    __tablename__ = 'financial_data'

    id = Column(Integer, primary_key=True)
    report_date = Column(DateTime, nullable=False)
    stock_code = Column(String, nullable=False)
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
    data_list = df.to_dict(orient='records')
    for data in data_list:
        db_financial_data = FinancialData(**data)  # 使用你的模型类
        db.add(db_financial_data)
    db.commit()


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
    financial_data = db.query(FinancialData).filter(
        FinancialData.stock_code == stock_code,
        FinancialData.report_date >= start_date,
        FinancialData.report_date <= end_date
    ).all()

    df = pd.DataFrame([data.__dict__ for data in financial_data])
    if '_sa_instance_state' in df.columns:
        df.drop('_sa_instance_state', axis=1, inplace=True)
    return df
