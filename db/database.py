from decorator import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中获取数据库连接信息
DATABASE_URL = os.getenv("DATABASE_URL")

# 检查数据库连接字符串是否为空
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set.")

# 创建数据库引擎
engine = create_engine(DATABASE_URL)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基础类
Base = declarative_base()


@contextmanager
def get_db_session():
    """
    获取数据库会话

    Returns:
        Session: 数据库会话对象
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


#  基础方法
def create_table(engine=engine):
    """
    创建所有数据库表
    """
    Base.metadata.create_all(engine)


def drop_table(engine=engine):
    """
    删除所有数据库表
    """
    Base.metadata.drop_all(engine)