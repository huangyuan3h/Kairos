
from datetime import datetime

from sqlalchemy import Column, Integer, Float, Date, String, func
from typing import List

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
    change_3d = Column(Float)  # 3天涨幅
    change_5d = Column(Float)  # 5天涨幅
    change_10d = Column(Float)  # 10天涨幅


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