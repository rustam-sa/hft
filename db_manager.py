import configparser
from functools import wraps
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def session_management(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        session = self.db_manager.get_session()
        try:
            return method(self, session, *args, **kwargs)
        finally:
            session.close()
    return wrapper


class DatabaseManager:

    @staticmethod
    def get_database_engine():
        config = configparser.ConfigParser()
        config.read('config.ini')
        database_url = config['hft']['url']
        engine = create_engine(database_url)
        return engine

    def get_session(self):
        engine = self.get_database_engine()
        Session = sessionmaker(engine)
        session = Session()
        return session
    

    

