from models import Base, MarketData
from db_manager import DatabaseManager, session_management
from data_getter import DataGetter


class DataService:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.db_manager = DatabaseManager()
        self.data_getter = DataGetter(symbol, timeframe)
        engine = DatabaseManager.get_database_engine()
        Base.metadata.create_all(engine)

    @session_management
    def save_market_data(self, session, span, pull_delay=0.3):
        candles = self.data_getter.get_mass_candles(span, pull_delay)[::-1].reset_index(drop=True)

        # Add symbol and timeframe columns to the DataFrame
        candles['symbol'] = self.symbol
        candles['timeframe'] = self.timeframe

        # Convert DataFrame to a list of dictionaries
        records = candles.to_dict(orient='records')

        # Create MarketData objects from the list of dictionaries
        market_data_objects = [MarketData(**record) for record in records]

        # Add all MarketData objects to the session
        session.bulk_save_objects(market_data_objects)

        # Commit the session
        session.commit()