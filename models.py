from sqlalchemy import Column, Integer, Float, BigInteger, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class MarketData(Base):
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String)
    timeframe = Column(String)
    timestamp = Column(BigInteger, nullable=False)
    open = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)


class CandlestickImage(Base):
    __tablename__ = 'candlestick_images'
    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(Integer, nullable=False)
    collection_name = Column(String, nullable=False)
    timestamp = Column(Integer, nullable=False)
    image_data = Column(LargeBinary, nullable=False)
    label = Column(Integer, nullable=False)