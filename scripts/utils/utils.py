from dotenv import load_dotenv
import os
import polars as pl
from typing import Any, Dict
from sqlalchemy import create_engine, MetaData, Column, Integer, Float, String, Engine
from sqlalchemy.exc import SQLAlchemyError
import logging
class Helpers:

    def __init__(self):
        pass
    
    @staticmethod
    def get_dbs() -> tuple[str,str]:
        """
        This creates a tuple of the paths to the training SQLite database and testing SQLite database.

        Returns:
        - train_db (str): a string of the file path to the training database
        - test_db (str): a string of the file path to the training database
        """
        load_dotenv()
        train_db = os.getenv("TRAIN_DB")
        test_db = os.getenv("TEST_DB")
        return train_db, test_db
    
    @staticmethod
    def small_fetch(db:str) -> pl.DataFrame:
        """
        This helper function fetches all columns for the first 10,000 rows of data in the specified database and loads them into a polars dataframe

        Args:
        - db (str): The file path for the database you want to fetch from

        Returms:
        - df (DataFrame): a polars data frame containing the firsst 10,000 rows of data 
        """
        conn = create_engine(db).connect()
        query = "SELECT * FROM TrainSMILES LIMIT 10000"
        df = pl.read_database(query = query, connection=conn)
        conn.close()
        return df
    
    @staticmethod
    def create_table(df: pl.DataFrame, table_name: str, db:str) -> None:
        """
        Create a SQLite table based on the provided Polars DataFrame.

        Many tables have 100+ columns, such as the descriptors table and the MACCs keys table.
        This function programmatically creates tables by mapping Polars data types to SQLite equivalents.

        Args:
            df (pl.DataFrame): The DataFrame containing the data to load.
            table_name (str): Name of the table to create.
            db (str): The file path for the SQLite database where the table will be created.

        Raises:
            ValueError: If 'MoleculeID' column is missing from the DataFrame.
            SQLAlchemyError: If there is an issue executing the SQL statement.
            Exception: For any other unforeseen errors.
        """
        # Define mapping from Polars data types to SQLAlchemy types
        dtype_mapping: Dict[Any, Any] = {
            pl.Int8: Integer,
            pl.Int16: Integer,
            pl.Int32: Integer,
            pl.Int64: Integer,
            pl.UInt8: Integer,
            pl.UInt16: Integer,
            pl.UInt32: Integer,
            pl.UInt64: Integer,
            pl.Float32: Float,
            pl.Float64: Float,
            pl.Boolean: Integer, # SQLite does not have a boolean datatype
            pl.String: String
        }
        
        metadata = MetaData()

        try:
            # Ensure MoleculeID exists in the DataFrame
            if "MoleculeID" not in df.columns:
                logging.error("'MoleculeID' column is missing from the DataFrame.")
                raise ValueError("'MoleculeID' column is required for table creation.")

            # Prepare columns for SQL query
            columns = [
                Column("MoleculeID", Integer, primary_key=True)
            ]
            for column_name, polars_dtype in df.schema.items():
                if "MoleculeID" in column_name or "_Smiles" in column_name:
                    continue
                sqlalchemy_type = dtype_mapping.get(polars_dtype, String)
                columns.append(Column(column_name, sqlalchemy_type))
            
            # Create Table
            engine = create_engine(db)
            metadata.create_all(engine)
            logging.info(f"Table '{table_name}' created successfully in the database '{engine.name}'.")
        except SQLAlchemyError as e:
            logging.error(f"SQLAlchemy error occurred while creating table '{table_name}': {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise
        finally:
            engine.dispose()
    
    @staticmethod
    def get_chunks(i:int, db:str):
        """
        This gets chunks of data by Molecule ID in chunks of 500,000 observations
        """
        conn = create_engine(db).connect()
        try:
            query = f"SELECT * FROM TrainSMILES WHERE MoleculeID >= {i} AND MoleculeID < {i + 500000}"
            df = pl.read_database(query = query, connection=conn)
            conn.close()
            return df
        except Exception as e:
            logging.error(f"Error loading chunks into a data frame: {e}")
            try: 
                conn.close()
            except Exception as e:
                logging.error(f"Could not close connection: {e}")
            raise
