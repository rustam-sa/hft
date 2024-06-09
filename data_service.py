import pandas as pd
from datetime import datetime
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
from models import Base, MarketData, Collection, DataFrameMetadata, DataFrameEntry
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

    @session_management
    def save_samples_to_collection(self, session, samples_with_metadata, collection_name):
        """
        Save a list of samples, each containing a dataframe, timestamp, and label, to a collection in the database.

        Args:
            samples_with_metadata (list of dict): List of samples, each containing 'dataframe', 'timestamp', and 'label'.
            collection_name (str): The name of the collection.
        """

        # Create a new collection
        new_collection = Collection(collection_name=collection_name)
        session.add(new_collection)
        session.commit()
        
        # Add each sample to the collection
        for sample in samples_with_metadata:
            timestamp = sample['timestamp']
            label = sample['label']
            dataframe = sample['dataframe']
            
            # Create a new DataFrameMetadata
            new_metadata = DataFrameMetadata(timestamp=timestamp, label=label, collection=new_collection)
            session.add(new_metadata)
            session.commit()

            # Add each row of the dataframe as a DataFrameEntry
            for index, row in dataframe.iterrows():
                new_entry = DataFrameEntry(
                    data_frame_metadata=new_metadata,
                    open_btc=row.get('open (BTC)'),
                    close_btc=row.get('close (BTC)'),
                    high_btc=row.get('high (BTC)'),
                    low_btc=row.get('low (BTC)'),
                    volume_btc=row.get('volume (BTC)'),
                    amount_btc=row.get('amount (BTC)'),
                    open_eth=row.get('open (ETH)'),
                    close_eth=row.get('close (ETH)'),
                    high_eth=row.get('high (ETH)'),
                    low_eth=row.get('low (ETH)'),
                    volume_eth=row.get('volume (ETH)'),
                    amount_eth=row.get('amount (ETH)'),
                    open_btc_robust=row.get('open (BTC) robust'),
                    close_btc_robust=row.get('close (BTC) robust'),
                    high_btc_robust=row.get('high (BTC) robust'),
                    low_btc_robust=row.get('low (BTC) robust'),
                    volume_btc_robust=row.get('volume (BTC) robust'),
                    amount_btc_robust=row.get('amount (BTC) robust'),
                    open_eth_robust=row.get('open (ETH) robust'),
                    close_eth_robust=row.get('close (ETH) robust'),
                    high_eth_robust=row.get('high (ETH) robust'),
                    low_eth_robust=row.get('low (ETH) robust'),
                    volume_eth_robust=row.get('volume (ETH) robust'),
                    amount_eth_robust=row.get('amount (ETH) robust'),
                    open_btc_standard=row.get('open (BTC) standard'),
                    close_btc_standard=row.get('close (BTC) standard'),
                    high_btc_standard=row.get('high (BTC) standard'),
                    low_btc_standard=row.get('low (BTC) standard'),
                    volume_btc_standard=row.get('volume (BTC) standard'),
                    amount_btc_standard=row.get('amount (BTC) standard'),
                    open_eth_standard=row.get('open (ETH) standard'),
                    close_eth_standard=row.get('close (ETH) standard'),
                    high_eth_standard=row.get('high (ETH) standard'),
                    low_eth_standard=row.get('low (ETH) standard'),
                    volume_eth_standard=row.get('volume (ETH) standard'),
                    amount_eth_standard=row.get('amount (ETH) standard'),
                    open_btc_minmax=row.get('open (BTC) minmax'),
                    close_btc_minmax=row.get('close (BTC) minmax'),
                    high_btc_minmax=row.get('high (BTC) minmax'),
                    low_btc_minmax=row.get('low (BTC) minmax'),
                    volume_btc_minmax=row.get('volume (BTC) minmax'),
                    amount_btc_minmax=row.get('amount (BTC) minmax'),
                    open_eth_minmax=row.get('open (ETH) minmax'),
                    close_eth_minmax=row.get('close (ETH) minmax'),
                    high_eth_minmax=row.get('high (ETH) minmax'),
                    low_eth_minmax=row.get('low (ETH) minmax'),
                    volume_eth_minmax=row.get('volume (ETH) minmax'),
                    amount_eth_minmax=row.get('amount (ETH) minmax')
                )
                session.add(new_entry)
            session.commit()
        print(f"Collection '{collection_name}' saved successfully with {len(samples_with_metadata)} samples.")
            
    def retrieve_samples_by_scaler(self, collection_name, scaler_name):
        """
        Retrieve samples from a collection by scaler and return them as a list of dictionaries 
        containing the dataframe, timestamp, and label.

        Args:
            collection_name (str): The name of the collection.
            scaler_name (str): The name of the scaler to filter columns by.

        Returns:
            list of dict: List of dictionaries, each containing 'dataframe', 'timestamp', and 'label'.
        """
        engine = self.db_manager.get_database_engine()
        samples_with_metadata = []

        with engine.connect() as connection:
            query = text(f"""
                SELECT m.timestamp, m.label, e.* 
                FROM data_frame_metadata m
                JOIN data_frame_entries e ON m.id = e.data_frame_metadata_id
                JOIN collections c ON m.collection_id = c.id
                WHERE c.collection_name = :collection_name
            """)
            result = connection.execute(query, {'collection_name': collection_name})

            for row in result:
                # Filter columns based on scaler_name
                columns_to_include = [col for col in row.keys() if scaler_name in col]
                filtered_data = {col: row[col] for col in columns_to_include}
                
                # Create a DataFrame from the filtered data
                df = pd.DataFrame([filtered_data])
                
                samples_with_metadata.append({
                    'dataframe': df,
                    'timestamp': row['timestamp'],
                    'label': row['label']
                })

        return samples_with_metadata
    
    def replace_with_scaled_columns(self, list_of_dataframes, scaler_type):
        "options: 'robust', 'standard', 'minmax'"
        for df in list_of_dataframes:
            # Identify the scaler columns
            scaler_cols = [col for col in df.columns if scaler_type in col]
            
            # Extract the scaled columns
            scaled_cols = df[scaler_cols].copy()
            
            # Drop the scaler columns from the DataFrame
            df.drop(columns=scaler_cols, inplace=True)
            
            # Replace the unscaled columns with the scaled columns
            for col in df.columns:
                for scaler_col in scaler_cols:
                    if col in scaler_col:
                        df[col] = scaled_cols[scaler_col]
                        break
        return list_of_dataframes

    @session_management
    def get_dataframes_as_dicts(self, session, scaler_type=None):
        """
        Retrieve all dataframes and convert them into dictionaries containing 'label', 'timestamp', and 'data'.

        Args:
            scaler_type (str, optional): The type of scaled data to replace in the dataframes.

        Returns:
            list: A list of dictionaries with 'label', 'timestamp', and 'data' (pandas DataFrame).
        """
        collections = session.query(Collection).all()
        result = []

        for collection in collections:
            for metadata in collection.data_frame_metadata:
                entries = metadata.data_frame_entries

                data = pd.DataFrame([{
                    'open_btc': entry.open_btc,
                    'close_btc': entry.close_btc,
                    'high_btc': entry.high_btc,
                    'low_btc': entry.low_btc,
                    'volume_btc': entry.volume_btc,
                    'amount_btc': entry.amount_btc,
                    'open_eth': entry.open_eth,
                    'close_eth': entry.close_eth,
                    'high_eth': entry.high_eth,
                    'low_eth': entry.low_eth,
                    'volume_eth': entry.volume_eth,
                    'amount_eth': entry.amount_eth,
                    'open_btc_robust': entry.open_btc_robust,
                    'close_btc_robust': entry.close_btc_robust,
                    'high_btc_robust': entry.high_btc_robust,
                    'low_btc_robust': entry.low_btc_robust,
                    'volume_btc_robust': entry.volume_btc_robust,
                    'amount_btc_robust': entry.amount_btc_robust,
                    'open_eth_robust': entry.open_eth_robust,
                    'close_eth_robust': entry.close_eth_robust,
                    'high_eth_robust': entry.high_eth_robust,
                    'low_eth_robust': entry.low_eth_robust,
                    'volume_eth_robust': entry.volume_eth_robust,
                    'amount_eth_robust': entry.amount_eth_robust,
                    'open_btc_standard': entry.open_btc_standard,
                    'close_btc_standard': entry.close_btc_standard,
                    'high_btc_standard': entry.high_btc_standard,
                    'low_btc_standard': entry.low_btc_standard,
                    'volume_btc_standard': entry.volume_btc_standard,
                    'amount_btc_standard': entry.amount_btc_standard,
                    'open_eth_standard': entry.open_eth_standard,
                    'close_eth_standard': entry.close_eth_standard,
                    'high_eth_standard': entry.high_eth_standard,
                    'low_eth_standard': entry.low_eth_standard,
                    'volume_eth_standard': entry.volume_eth_standard,
                    'amount_eth_standard': entry.amount_eth_standard,
                    'open_btc_minmax': entry.open_btc_minmax,
                    'close_btc_minmax': entry.close_btc_minmax,
                    'high_btc_minmax': entry.high_btc_minmax,
                    'low_btc_minmax': entry.low_btc_minmax,
                    'volume_btc_minmax': entry.volume_btc_minmax,
                    'amount_btc_minmax': entry.amount_btc_minmax,
                    'open_eth_minmax': entry.open_eth_minmax,
                    'close_eth_minmax': entry.close_eth_minmax,
                    'high_eth_minmax': entry.high_eth_minmax,
                    'low_eth_minmax': entry.low_eth_minmax,
                    'volume_eth_minmax': entry.volume_eth_minmax,
                    'amount_eth_minmax': entry.amount_eth_minmax
                } for entry in entries])

                if scaler_type:
                    data = self.replace_with_scaled_columns([data], scaler_type)[0]

                result.append({
                    'label': metadata.label,
                    'timestamp': metadata.timestamp,
                    'data': data
                })

        return result
    
    def get_all_collection_names(self):
        """
        Retrieve all collection names from the database.

        Returns:
            list: A list of all collection names.
        """
        engine = self.db_manager.get_database_engine()
        collection_names = []
        with engine.connect() as connection:
            query = text("SELECT collection_name FROM collections")
            result = connection.execute(query)
            collection_names = [row['collection_name'] for row in result]
        return collection_names