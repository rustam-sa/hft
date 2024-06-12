import logging
import pandas as pd
from datetime import datetime
from sqlalchemy import text, inspect, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload
from models import Base, MarketData, Collection, DataFrameMetadata, DataFrameEntry
from db_manager import DatabaseManager, session_management
from data_getter import DataGetter


def check_timestamps(timestamps):
    if not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(timestamps)
    timestamps = pd.to_datetime(timestamps)
    
    ascending = timestamps.is_monotonic_increasing
    time_diffs = timestamps.diff().dt.total_seconds()
    gaps = time_diffs[1:] > 180
    has_gaps = gaps.any()
    
    checks = {
        'ascending': ascending,
        'has_gaps': has_gaps,
        'gaps_indices': gaps[gaps].index.tolist()
    }
    if not checks['ascending'] or checks['has_gaps']:
        print(checks)
        raise ValueError("Timestamps are not in ascending order or there are gaps larger than 3 minutes.")
    return checks

logging.basicConfig(level=logging.INFO)


class DataService:
    def __init__(self, symbol="ETH-USDT", timeframe="3min"):
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

        try:
            # Check if the collection already exists
            existing_collection = session.query(Collection).filter_by(collection_name=collection_name).first()
            if existing_collection:
                new_collection = existing_collection
            else:
                # Create a new collection
                new_collection = Collection(collection_name=collection_name)
                session.add(new_collection)
                session.commit()

            for sample in samples_with_metadata:
                timestamp = sample['timestamp']
                label = sample['label']
                dataframe = sample['dataframe']

                # Rename columns to match the DataFrameEntry schema
                dataframe = dataframe.rename(columns={
                    'open (ETH)': 'open_eth',
                    'close (ETH)': 'close_eth',
                    'high (ETH)': 'high_eth',
                    'low (ETH)': 'low_eth',
                    'amount (ETH)': 'amount_eth',
                    'volume (ETH)': 'volume_eth',
                    'open (BTC)': 'open_btc',
                    'close (BTC)': 'close_btc',
                    'high (BTC)': 'high_btc',
                    'low (BTC)': 'low_btc',
                    'amount (BTC)': 'amount_btc',
                    'volume (BTC)': 'volume_btc',
                    'open (ETH)_RobustScaler': 'open_eth_robust',
                    'close (ETH)_RobustScaler': 'close_eth_robust',
                    'high (ETH)_RobustScaler': 'high_eth_robust',
                    'low (ETH)_RobustScaler': 'low_eth_robust',
                    'amount (ETH)_RobustScaler': 'amount_eth_robust',
                    'volume (ETH)_RobustScaler': 'volume_eth_robust',
                    'open (BTC)_RobustScaler': 'open_btc_robust',
                    'close (BTC)_RobustScaler': 'close_btc_robust',
                    'high (BTC)_RobustScaler': 'high_btc_robust',
                    'low (BTC)_RobustScaler': 'low_btc_robust',
                    'amount (BTC)_RobustScaler': 'amount_btc_robust',
                    'volume (BTC)_RobustScaler': 'volume_btc_robust',
                    'open (ETH)_StandardScaler': 'open_eth_standard',
                    'close (ETH)_StandardScaler': 'close_eth_standard',
                    'high (ETH)_StandardScaler': 'high_eth_standard',
                    'low (ETH)_StandardScaler': 'low_eth_standard',
                    'amount (ETH)_StandardScaler': 'amount_eth_standard',
                    'volume (ETH)_StandardScaler': 'volume_eth_standard',
                    'open (BTC)_StandardScaler': 'open_btc_standard',
                    'close (BTC)_StandardScaler': 'close_btc_standard',
                    'high (BTC)_StandardScaler': 'high_btc_standard',
                    'low (BTC)_StandardScaler': 'low_btc_standard',
                    'amount (BTC)_StandardScaler': 'amount_btc_standard',
                    'volume (BTC)_StandardScaler': 'volume_btc_standard',
                    'open (ETH)_MinMaxScaler': 'open_eth_minmax',
                    'close (ETH)_MinMaxScaler': 'close_eth_minmax',
                    'high (ETH)_MinMaxScaler': 'high_eth_minmax',
                    'low (ETH)_MinMaxScaler': 'low_eth_minmax',
                    'amount (ETH)_MinMaxScaler': 'amount_eth_minmax',
                    'volume (ETH)_MinMaxScaler': 'volume_eth_minmax',
                    'open (BTC)_MinMaxScaler': 'open_btc_minmax',
                    'close (BTC)_MinMaxScaler': 'close_btc_minmax',
                    'high (BTC)_MinMaxScaler': 'high_btc_minmax',
                    'low (BTC)_MinMaxScaler': 'low_btc_minmax',
                    'amount (BTC)_MinMaxScaler': 'amount_btc_minmax',
                    'volume (BTC)_MinMaxScaler': 'volume_btc_minmax'
                })

                # Create a new DataFrameMetadata
                new_metadata = DataFrameMetadata(timestamp=timestamp, label=label, collection=new_collection)
                session.add(new_metadata)
                session.commit()

                # Add each row of the dataframe as a DataFrameEntry
                for index, row in dataframe.iterrows():
                    new_entry = DataFrameEntry(
                        data_frame_metadata=new_metadata,
                        timestamp = row.get('timestamp'),
                        open_btc=row.get('open_btc'),
                        close_btc=row.get('close_btc'),
                        high_btc=row.get('high_btc'),
                        low_btc=row.get('low_btc'),
                        volume_btc=row.get('volume_btc'),
                        amount_btc=row.get('amount_btc'),
                        open_eth=row.get('open_eth'),
                        close_eth=row.get('close_eth'),
                        high_eth=row.get('high_eth'),
                        low_eth=row.get('low_eth'),
                        volume_eth=row.get('volume_eth'),
                        amount_eth=row.get('amount_eth'),
                        open_btc_robust=row.get('open_btc_robust'),
                        close_btc_robust=row.get('close_btc_robust'),
                        high_btc_robust=row.get('high_btc_robust'),
                        low_btc_robust=row.get('low_btc_robust'),
                        volume_btc_robust=row.get('volume_btc_robust'),
                        amount_btc_robust=row.get('amount_btc_robust'),
                        open_eth_robust=row.get('open_eth_robust'),
                        close_eth_robust=row.get('close_eth_robust'),
                        high_eth_robust=row.get('high_eth_robust'),
                        low_eth_robust=row.get('low_eth_robust'),
                        volume_eth_robust=row.get('volume_eth_robust'),
                        amount_eth_robust=row.get('amount_eth_robust'),
                        open_btc_standard=row.get('open_btc_standard'),
                        close_btc_standard=row.get('close_btc_standard'),
                        high_btc_standard=row.get('high_btc_standard'),
                        low_btc_standard=row.get('low_btc_standard'),
                        volume_btc_standard=row.get('volume_btc_standard'),
                        amount_btc_standard=row.get('amount_btc_standard'),
                        open_eth_standard=row.get('open_eth_standard'),
                        close_eth_standard=row.get('close_eth_standard'),
                        high_eth_standard=row.get('high_eth_standard'),
                        low_eth_standard=row.get('low_eth_standard'),
                        volume_eth_standard=row.get('volume_eth_standard'),
                        amount_eth_standard=row.get('amount_eth_standard'),
                        open_btc_minmax=row.get('open_btc_minmax'),
                        close_btc_minmax=row.get('close_btc_minmax'),
                        high_btc_minmax=row.get('high_btc_minmax'),
                        low_btc_minmax=row.get('low_btc_minmax'),
                        volume_btc_minmax=row.get('volume_btc_minmax'),
                        amount_btc_minmax=row.get('amount_btc_minmax'),
                        open_eth_minmax=row.get('open_eth_minmax'),
                        close_eth_minmax=row.get('close_eth_minmax'),
                        high_eth_minmax=row.get('high_eth_minmax'),
                        low_eth_minmax=row.get('low_eth_minmax'),
                        volume_eth_minmax=row.get('volume_eth_minmax'),
                        amount_eth_minmax=row.get('amount_eth_minmax')
                    )
                    session.add(new_entry)

                session.commit()
            # logging.info(f"Metadata with timestamp {timestamp} and label {label} added to collection '{collection_name}'")
            # logging.info(f"Collection '{collection_name}' saved successfully with {len(samples_with_metadata)} samples.")

        except SQLAlchemyError as e:
            logging.error(f"Error saving samples to collection: {e}")
            session.rollback()
            raise

        finally:
            session.close()


    def get_all_collection_names(self):
        engine = self.db_manager.get_database_engine()
        with engine.connect() as connection:
            query = text("SELECT collection_name FROM collections")
            result = connection.execute(query)
            collection_names = [row[0] for row in result]
            print("Collection names retrieved:", collection_names)  # Debugging statement
        return collection_names

    def check_for_duplicate_collections(self):
        engine = self.db_manager.get_database_engine()
        with engine.connect() as connection:
            query = text("""
                SELECT collection_name, COUNT(*) 
                FROM collections 
                GROUP BY collection_name 
                HAVING COUNT(*) > 1
            """)
            result = connection.execute(query)
            duplicates = [row[0] for row in result]
            if duplicates:
                print("Duplicate collections found:", duplicates)
            else:
                print("No duplicate collections found.")
            return duplicates
        
    @session_management
    def remove_duplicate_collections(self, session):
        try:
            # Step 1: Find duplicate collection names
            duplicates_query = (
                session.query(Collection.collection_name)
                .group_by(Collection.collection_name)
                .having(func.count(Collection.id) > 1)
            )
            duplicate_names = [name for name, in duplicates_query]
            print(f"Duplicate collections: {duplicate_names}")

            if duplicate_names:
                for collection_name in duplicate_names:
                    # Step 2: Find all collections with the duplicate name
                    collections = session.query(Collection).filter_by(collection_name=collection_name).all()

                    if len(collections) > 1:
                        # Keep the first collection and delete the rest
                        primary_collection = collections[0]
                        duplicate_collections = collections[1:]

                        for collection in duplicate_collections:
                            # Step 3: Collect metadata IDs for batch deletion
                            metadata_ids = [
                                metadata.id for metadata in collection.data_frames_metadata
                            ]

                            if metadata_ids:
                                # Delete related DataFrameEntries in batches
                                session.query(DataFrameEntry).filter(
                                    DataFrameEntry.data_frame_metadata_id.in_(metadata_ids)
                                ).delete(synchronize_session='fetch')

                                # Delete DataFrameMetadata in batches
                                session.query(DataFrameMetadata).filter(
                                    DataFrameMetadata.id.in_(metadata_ids)
                                ).delete(synchronize_session='fetch')

                            # Delete the duplicate collection
                            session.delete(collection)

                session.commit()  # Commit the transaction
                print("Duplicate collections removed.")
            else:
                print("No duplicate collections found.")

        except SQLAlchemyError as e:
            session.rollback()  # Rollback the transaction in case of error
            print(f"An error occurred: {e}")

    @session_management
    def delete_all_collections(self, session):
        try:
            # Fetch all collections
            all_collections = session.query(Collection).all()

            for collection in all_collections:
                # Collect metadata IDs for batch deletion
                metadata_ids = [metadata.id for metadata in collection.data_frames_metadata]

                if metadata_ids:
                    # Delete related DataFrameEntries in batches
                    session.query(DataFrameEntry).filter(
                        DataFrameEntry.data_frame_metadata_id.in_(metadata_ids)
                    ).delete(synchronize_session='fetch')

                    # Delete DataFrameMetadata in batches
                    session.query(DataFrameMetadata).filter(
                        DataFrameMetadata.id.in_(metadata_ids)
                    ).delete(synchronize_session='fetch')

                # Delete the collection itself
                session.delete(collection)

            session.commit()  # Commit the transaction
            print("All collections removed successfully.")

        except SQLAlchemyError as e:
            session.rollback()  # Rollback the transaction in case of error
            print(f"An error occurred: {e}")
            raise

        finally:
            session.close()

    @session_management
    def merge_duplicate_market_data(self, session):
        try:
            # Step 1: Find duplicate MarketData entries based on unique fields (example: symbol, timeframe, and timestamp)
            duplicates_query = (
                session.query(
                    MarketData.symbol,
                    MarketData.timeframe,
                    MarketData.timestamp,
                    func.count(MarketData.id).label('count')
                )
                .group_by(MarketData.symbol, MarketData.timeframe, MarketData.timestamp)
                .having(func.count(MarketData.id) > 1)
            )

            duplicates = duplicates_query.all()
            print(f"Found {len(duplicates)} duplicate MarketData entries.")

            # Step 2: Process each set of duplicates
            for duplicate in duplicates:
                symbol, timeframe, timestamp, count = duplicate

                # Fetch all entries that are duplicates
                duplicate_entries = (
                    session.query(MarketData)
                    .filter_by(symbol=symbol, timeframe=timeframe, timestamp=timestamp)
                    .all()
                )

                if len(duplicate_entries) > 1:
                    primary_entry = duplicate_entries[0]
                    duplicate_entries_to_delete = duplicate_entries[1:]

                    # Example strategy: merge data from duplicates (here, just deleting duplicates)
                    for entry in duplicate_entries_to_delete:
                        session.delete(entry)

            session.commit()  # Commit the transaction
            print("Duplicate MarketData entries merged and duplicates removed.")

        except SQLAlchemyError as e:
            session.rollback()  # Rollback the transaction in case of error
            print(f"An error occurred: {e}")

    from sqlalchemy.orm import joinedload

    @session_management
    def get_dataframes_as_dicts(self, session, collection_name=None):
        """
        Retrieve dataframes for a specific collection and convert them into dictionaries containing 'label', 'timestamp', and 'data'.

        Args:
            scaler_type (str, optional): The type of scaled data to replace in the dataframes.
            collection_name (str, optional): The name of the collection to filter.

        Returns:
            list: A list of dictionaries with 'label', 'timestamp', and 'data' (pandas DataFrame).
        """
        query = session.query(Collection)
        
        if collection_name:
            query = query.filter(Collection.collection_name == collection_name)

        collections = query.all()
        result = []

        for collection in collections:
            for metadata in collection.data_frames_metadata:
                entries_query = session.query(DataFrameEntry).filter(DataFrameEntry.data_frame_metadata_id == metadata.id)
                entries = entries_query.all()

                data = pd.DataFrame([{
                    'timestamp': entry.timestamp,

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

                result.append({
                    'label': metadata.label,
                    'timestamp': metadata.timestamp,
                    'data': data
                })
        result.sort(key=lambda x: x['timestamp'])
        print("dataframe_get_gap")
        timestamps = [sample['timestamp'] for sample in result]
        check_timestamps(timestamps)

        return result
    # def get_dataframes_as_dicts(self, session, scaler_type=None, collection_name=None):
    #     """
    #     Retrieve dataframes for a specific collection and convert them into dictionaries containing 'label', 'timestamp', and 'data'.

    #     Args:
    #         scaler_type (str, optional): The type of scaled data to replace in the dataframes.
    #         collection_name (str, optional): The name of the collection to filter.

    #     Returns:
    #         list: A list of dictionaries with 'label', 'timestamp', and 'data' (pandas DataFrame).
    #     """
    #     query = session.query(Collection)
        
    #     if collection_name:
    #         query = query.filter(Collection.collection_name == collection_name)
            
    #     query = query.options(joinedload(Collection.data_frames_metadata).joinedload(DataFrameMetadata.data_frame_entries))

    #     collections = query.all()

    #     result = []
    #     for collection in collections:
    #         for metadata in collection.data_frames_metadata:
    #             entries = metadata.data_frame_entries

    #             data = pd.DataFrame([{
    #                 'open_btc': entry.open_btc,
    #                 'close_btc': entry.close_btc,
    #                 'high_btc': entry.high_btc,
    #                 'low_btc': entry.low_btc,
    #                 'volume_btc': entry.volume_btc,
    #                 'amount_btc': entry.amount_btc,
    #                 'open_eth': entry.open_eth,
    #                 'close_eth': entry.close_eth,
    #                 'high_eth': entry.high_eth,
    #                 'low_eth': entry.low_eth,
    #                 'volume_eth': entry.volume_eth,
    #                 'amount_eth': entry.amount_eth,
    #                 'open_btc_robust': entry.open_btc_robust,
    #                 'close_btc_robust': entry.close_btc_robust,
    #                 'high_btc_robust': entry.high_btc_robust,
    #                 'low_btc_robust': entry.low_btc_robust,
    #                 'volume_btc_robust': entry.volume_btc_robust,
    #                 'amount_btc_robust': entry.amount_btc_robust,
    #                 'open_eth_robust': entry.open_eth_robust,
    #                 'close_eth_robust': entry.close_eth_robust,
    #                 'high_eth_robust': entry.high_eth_robust,
    #                 'low_eth_robust': entry.low_eth_robust,
    #                 'volume_eth_robust': entry.volume_eth_robust,
    #                 'amount_eth_robust': entry.amount_eth_robust,
    #                 'open_btc_standard': entry.open_btc_standard,
    #                 'close_btc_standard': entry.close_btc_standard,
    #                 'high_btc_standard': entry.high_btc_standard,
    #                 'low_btc_standard': entry.low_btc_standard,
    #                 'volume_btc_standard': entry.volume_btc_standard,
    #                 'amount_btc_standard': entry.amount_btc_standard,
    #                 'open_eth_standard': entry.open_eth_standard,
    #                 'close_eth_standard': entry.close_eth_standard,
    #                 'high_eth_standard': entry.high_eth_standard,
    #                 'low_eth_standard': entry.low_eth_standard,
    #                 'volume_eth_standard': entry.volume_eth_standard,
    #                 'amount_eth_standard': entry.amount_eth_standard,
    #                 'open_btc_minmax': entry.open_btc_minmax,
    #                 'close_btc_minmax': entry.close_btc_minmax,
    #                 'high_btc_minmax': entry.high_btc_minmax,
    #                 'low_btc_minmax': entry.low_btc_minmax,
    #                 'volume_btc_minmax': entry.volume_btc_minmax,
    #                 'amount_btc_minmax': entry.amount_btc_minmax,
    #                 'open_eth_minmax': entry.open_eth_minmax,
    #                 'close_eth_minmax': entry.close_eth_minmax,
    #                 'high_eth_minmax': entry.high_eth_minmax,
    #                 'low_eth_minmax': entry.low_eth_minmax,
    #                 'volume_eth_minmax': entry.volume_eth_minmax,
    #                 'amount_eth_minmax': entry.amount_eth_minmax
    #             } for entry in entries])

    #             if scaler_type:
    #                 data = self.replace_with_scaled_columns([data], scaler_type)[0]

    #             result.append({
    #                 'label': metadata.label,
    #                 'timestamp': metadata.timestamp,
    #                 'data': data
    #             })

    #     return result
