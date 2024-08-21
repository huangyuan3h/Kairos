from datetime import datetime, date

from sqlalchemy import Column, Integer, Float, Date, String, func

from db.database import Base
from sqlalchemy.orm import Session
import pandas as pd
from sqlalchemy import select


class PredictReport(Base):
    """预测报告模型"""
    __tablename__ = "predict_report"

    id = Column(Integer, primary_key=True, index=True)
    report_date = Column(Date, nullable=False)
    stock_code = Column(String(10), nullable=False)
    change_1d = Column(Float)  # 1天涨幅
    change_2d = Column(Float)  # 2天涨幅
    change_3d = Column(Float)  # 3天涨幅
    trend = Column(Float)  # 4-10天涨幅
    operation_1d = Column(Float)  # 后一天买，1天卖的涨幅
    operation_2d = Column(Float)  # 后一天买，2天卖的涨幅


def bulk_insert_predict_report(db: Session, df: pd.DataFrame):
    """
    将 Pandas DataFrame 中的数据批量插入到数据库

    Args:
        db (Session): 数据库会话对象
        df (pd.DataFrame): 包含预测报告数据的 DataFrame
    """
    data_list = df.to_dict(orient='records')
    for data in data_list:
        db_predict_report = PredictReport(**data)
        db.add(db_predict_report)
    db.commit()


def get_predict_report_by_date(db: Session, report_date: str) -> pd.DataFrame:
    """
    从数据库中读取某一天的预测报告数据，并将其存储在 Pandas DataFrame 中

    Args:
        db (Session): 数据库会话对象
        report_date (date): 要查询的报告日期

    Returns:
        pd.DataFrame: 包含指定日期预测报告数据的 DataFrame
    """
    query = select("*").where(PredictReport.report_date == report_date)
    result = db.execute(query).all()

    df = pd.DataFrame(result, columns=[col.key for col in PredictReport.__table__.columns])

    return df
