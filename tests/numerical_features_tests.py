from sqlalchemy import create_engine, inspect as sql_inspect, Table, MetaData
from sqlalchemy.dialects.sqlite import insert
from scripts.utils.utils import Helpers
from scripts.dataprocessing.numerical_features import NumericalFeatures
import polars as pl

class FeatureTests():
    """
    
    """
    def __init__(self):
        self.test_features = NumericalFeatures(dev_mode=True, db_train=True)
        self.train_db, self.test_db = Helpers().get_dbs()
        pass

    def check_descriptors_table(self, db):
        """
        This helper function fetches all columns for the first 10,000 rows of data in the specified database and loads them into a polars dataframe

        Args:
        - db (str): The file path for the database you want to fetch from

        Returms:
        - df (DataFrame): a polars data frame containing the firsst 10,000 rows of data 
        """
        conn = create_engine(db).connect()
        query = "SELECT * FROM MoleculeDescriptors LIMIT 10000"
        df = pl.read_database(query = query, connection=conn)
        print(df.describe())
    
    def main_numftrs_test(self):
        self.check_descriptors_table(db = self.train_db)

if __name__ == "__main__":
    tester = FeatureTests()
    tester.main_numftrs_test()