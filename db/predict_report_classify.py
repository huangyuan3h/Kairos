from datetime import date

from sqlalchemy import Column, Integer, Date, String, Float

from db.database import Base
from sqlalchemy.orm import Session
import pandas as pd
from sqlalchemy import select


class PredictClassifyReport(Base):
    """预测报告模型"""
    __tablename__ = "predict_classify_report"

    id = Column(Integer, primary_key=True, index=True)
    report_date = Column(Date, nullable=False)
    stock_code = Column(String(10), nullable=False)
    rise = Column(Float(), nullable=False)
    jitter = Column(Float(), nullable=False)
    fall = Column(Float(), nullable=False)
    predict_class = Column(Integer)  # 0, 1, 2 暴跌，震荡， 暴涨
    model_version = Column(String(32), nullable=False)


def bulk_insert_predict_report(db: Session, df: pd.DataFrame):
    """
    将 Pandas DataFrame 中的数据批量插入到数据库

    Args:
        db (Session): 数据库会话对象
        df (pd.DataFrame): 包含预测报告数据的 DataFrame
    """
    data_list = df.to_dict(orient='records')
    for data in data_list:
        db_predict_report = PredictClassifyReport(**data)
        db.add(db_predict_report)
    db.commit()


def get_predict_classify_report_by_date(db: Session, report_date: str) -> pd.DataFrame:
    """
    从数据库中读取某一天的预测报告数据，并将其存储在 Pandas DataFrame 中

    Args:
        db (Session): 数据库会话对象
        report_date (date): 要查询的报告日期

    Returns:
        pd.DataFrame: 包含指定日期预测报告数据的 DataFrame
    """
    query = select("*").where(PredictClassifyReport.report_date == report_date)
    result = db.execute(query).all()

    df = pd.DataFrame(result, columns=[col.key for col in PredictClassifyReport.__table__.columns])

    return df
