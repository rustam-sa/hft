import pandas as pd
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
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

    def load_market_data(self):
        """Load MarketData table into a pandas DataFrame filtered by symbol."""
        engine = self.db_manager.get_database_engine()
        with engine.connect() as connection:
            query = f"SELECT * FROM market_data WHERE symbol = '{self.symbol}'"
            df = pd.read_sql_query(query, con=connection)
        return df
    
    def ensure_schema_exists(self, schema):
        engine = self.db_manager.get_database_engine()
        with engine.connect() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))

    def save_dataframes_to_db(self, dataframes, schema, table_prefix='table_'):
        self.ensure_schema_exists(schema)
        engine = self.db_manager.get_database_engine()
        for i, df in enumerate(dataframes):
            table_name = f'{table_prefix}{i+1}'  # Create a unique table name
            try:
                df.to_sql(table_name, engine, index=False, if_exists='replace', schema=schema)  # Save the DataFrame to the database with schema
                print(f'Table {table_name} saved successfully in schema {schema}.')
            except SQLAlchemyError as e:
                print(f'Error saving table {table_name} in schema {schema}: {e}')

    def load_table_to_dataframe(self, table_name, schema=None):
        engine = self.db_manager.get_database_engine()
        df = pd.read_sql_table(table_name, engine, schema=schema)
        return df
    
    def load_all_tables_to_dataframes(self, table_names, schema=None):
        engine = self.db_manager.get_database_engine()
        dataframes = []
        for table_name in table_names:
            try:
                df = pd.read_sql_table(table_name, engine, schema=schema)
                dataframes.append(df)
                print(f'Table {table_name} loaded successfully from schema {schema}.')
            except SQLAlchemyError as e:
                print(f'Error loading table {table_name} from schema {schema}: {e}')
        return dataframes
    
    def load_all_tables_in_schema(self, schema):
        engine = self.db_manager.get_database_engine()
        inspector = inspect(engine)
        table_names = inspector.get_table_names(schema=schema)
        dataframes = []
        for table_name in table_names:
            try:
                df = pd.read_sql_table(table_name, engine, schema=schema)
                dataframes.append(df)
                print(f'Table {table_name} loaded successfully from schema {schema}.')
            except SQLAlchemyError as e:
                print(f'Error loading table {table_name} from schema {schema}: {e}')
        return dataframes
    
    @session_management
    def delete_schema(self, session, schema_name):
        session.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
        session.commit()
        print(f"Schema '{schema_name}' deleted successfully.")

