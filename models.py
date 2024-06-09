from sqlalchemy import Column, Integer, Float, BigInteger, String, LargeBinary, ForeignKey, DateTime
from sqlalchemy.orm import relationship
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
    timestamp = Column(BigInteger, nullable=False)
    image_data = Column(LargeBinary, nullable=False)
    label = Column(Integer, nullable=False)


class Collection(Base):
    __tablename__ = 'collections'
    
    id = Column(Integer, primary_key=True)
    collection_name = Column(String, nullable=False)
    data_frames_metadata = relationship('DataFrameMetadata', back_populates='collection')


class DataFrameMetadata(Base):
    __tablename__ = 'data_frame_metadata'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(BigInteger, nullable=False)
    label = Column(Integer, nullable=False)
    collection_id = Column(Integer, ForeignKey('collections.id'))
    collection = relationship('Collection', back_populates='data_frames_metadata')
    data_frame_entries = relationship('DataFrameEntry', back_populates='data_frame_metadata')


class DataFrameEntry(Base):
    __tablename__ = 'data_frame_entries'
    
    id = Column(Integer, primary_key=True)
    data_frame_metadata_id = Column(Integer, ForeignKey('data_frame_metadata.id'))
    data_frame_metadata = relationship('DataFrameMetadata', back_populates='data_frame_entries')
    
    # Columns for BTC and ETH data
    open_btc = Column(Float)
    close_btc = Column(Float)
    high_btc = Column(Float)
    low_btc = Column(Float)
    volume_btc = Column(Float)
    amount_btc = Column(Float)
    open_eth = Column(Float)
    close_eth = Column(Float)
    high_eth = Column(Float)
    low_eth = Column(Float)
    volume_eth = Column(Float)
    amount_eth = Column(Float)
    
    # Scaled columns
    open_btc_robust = Column(Float)
    close_btc_robust = Column(Float)
    high_btc_robust = Column(Float)
    low_btc_robust = Column(Float)
    volume_btc_robust = Column(Float)
    amount_btc_robust = Column(Float)
    open_eth_robust = Column(Float)
    close_eth_robust = Column(Float)    
    high_eth_robust = Column(Float)
    low_eth_robust = Column(Float)
    volume_eth_robust = Column(Float)
    amount_eth_robust = Column(Float)
    
    open_btc_standard = Column(Float)
    close_btc_standard = Column(Float)
    high_btc_standard = Column(Float)
    low_btc_standard = Column(Float)
    volume_btc_standard = Column(Float)
    amount_btc_standard = Column(Float)
    open_eth_standard = Column(Float)
    close_eth_standard = Column(Float)
    high_eth_standard = Column(Float)
    low_eth_standard = Column(Float)
    volume_eth_standard = Column(Float)
    amount_eth_standard = Column(Float)
    
    open_btc_minmax = Column(Float)
    close_btc_minmax = Column(Float)
    high_btc_minmax = Column(Float)
    low_btc_minmax = Column(Float)
    volume_btc_minmax = Column(Float)
    amount_btc_minmax = Column(Float)
    open_eth_minmax = Column(Float)
    close_eth_minmax = Column(Float)
    high_eth_minmax = Column(Float)
    low_eth_minmax = Column(Float)
    volume_eth_minmax = Column(Float)
    amount_eth_minmax = Column(Float)