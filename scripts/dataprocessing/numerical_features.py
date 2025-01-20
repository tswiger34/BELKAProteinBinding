# Import data manipulation libraries
from ..utils.utils import Helpers
import polars as pl
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Crippen import _pyMolLogP
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional
import multiprocessing
from datetime import datetime
import inspect
from sqlalchemy import create_engine, inspect as sql_inspect, Table, MetaData
from sqlalchemy.dialects.sqlite import insert


logging.basicConfig(level=logging.INFO)

class NumericalFeatures:
    """
    Handles the creation of numerical features for molecular data.

    Args:
        dev_mode (bool): Enable development mode with a subset of data.
    """

    def __init__(self, dev_mode: bool, db_train:bool, num_workers:int = None, chunk_size:int = 500000):
        self.helper = Helpers()
        self.dev_mode = dev_mode
        # Set number of workers for parallel processing
        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count()
        else: self.num_workers = num_workers
        self.chunk_size = chunk_size
        # Choose the DB to connect to
        train_db, test_db = self.helper.get_dbs()
        if db_train:
            self.db = train_db
        else:
            self.db = test_db
    
    @staticmethod
    def calculate_maccs(smiles_batch: List[str]) -> Tuple[List[List[int]], List[str]]:
        """
        Generate MACCS keys for a batch of molecules.

        Args:
        smiles_batch (List[str]): List of molecule SMILEs

        Returns:
        Tuple[maccs_keys(List[List[int]]), key_names(List[str])]:
            - maccs_keys (List[List[int]]): A list of lists containing the MACCs keys values, i.e. each molecule has its MACCs keys values stored in a list, then these lists are stored in a master list
            - key_names (List[str]): A list of the names of the MACCs keys
        table_name (str): the name of the table the features will be inserted into
            
            Example:
            [
                [1,0,1,1,0],   # MACCs keys for molecule 1
                [1,0,0,0,0],   # MACCs keys for molecule 2
                [0,1,0,1,0]... # MACCs keys for molecule 3
            ], 
            [
                MACCsKey_1,
                MACCsKey_2,
                MACCsKey_3....
            ]
            "MACCsFeatures"

        """
        key_names = [f"MACCS_{i+1}" for i in range(166)]
        maccs_keys = []
        for smile in smiles_batch:
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol:
                    maccs = MACCSkeys.GenMACCSKeys(mol)
                    maccs_bits = list(maccs)[1:]
                    maccs_keys.append(maccs_bits)
                else:
                    maccs_keys.append([0] * 166)
            except Exception as e:
                logging.error(f"Error processing SMILEs '{smile}': {e}")
                maccs_keys.append([0] * 166)
        table_name = "MACCsFeatures"
        return maccs_keys, key_names, table_name
    
    @staticmethod
    def calculate_fingerprints(smiles_batch: List[str], radius: int = 2, n_bits: int = 2048) -> Tuple[List[List[int]], List[str]]:
        """
        Generate Morgan fingerprints for a batch of molecules using MorganGenerator.

        Args:
            smiles_batch (List[str]): List of molecule SMILES strings.
            radius (int, optional): Radius parameter for Morgan fingerprint. Defaults to 2.
            n_bits (int, optional): Number of bits in the fingerprint. Defaults to 2048.

        Returns:
            Tuple[List[List[int]], List[str]]:
                - morgan_fps (List[List[int]]): A list of lists containing the Morgan fingerprint bits.
                - key_names (List[str]): A list of the names of the Morgan fingerprint bits.
        """
        key_names = [f"Morgan_{i+1}" for i in range(n_bits)]
        morgan_fps = []
        
        try:
            generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
        except Exception as e:
            logging.error(f"Error initializing MorganGenerator: {e}")
            raise

        for smile in smiles_batch:
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol:
                    morgan = generator.GetFingerprint(mol)
                    # Convert fingerprint to a bit string and then to a list of integers
                    morgan_bits = [int(bit) for bit in morgan.ToBitString()]
                    morgan_fps.append(morgan_bits)
                else:
                    # Assign a fingerprint of all zeros for invalid SMILES
                    morgan_fps.append([0] * n_bits)
            except Exception as e:
                logging.error(f"Error processing SMILES '{smile}': {e}")
                morgan_fps.append([0] * n_bits)
        
        table_name = "MorganFPs"
        return morgan_fps, key_names, table_name
    
    @staticmethod
    def calculate_descriptors(smiles_batch: List[str]) -> tuple[List[List[Optional[float]]], List[str]]:
        """
        Calculate molecular descriptors for a batch of SMILES strings. 

        Args:
            smiles_batch (List[str]): Batch of SMILES strings.

        Returns:
            tuple[descriptors(List[List[Optional[float]]]), feature_names(str)]: Tuple of a List containing a list of descriptor values and a list containing the descriptor names
            - descriptors (List[List[Optional[float]]]): A list containing a list of descriptor values for an associated molecule
            - feature_names(str): a list of the descriptor names
        """
        descriptors = []
        for smile in smiles_batch:
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol:
                    desc = [func(mol) for _, func in Descriptors.descList]
                    descriptors.append(desc)
                else:
                    descriptors.append([None] * (len(Descriptors.descList)))
            except Exception as e:
                logging.error(f"Error processing SMILES '{smile}': {e}")
                descriptors.append([None] * (len(Descriptors.descList)))
        feature_names = [desc[0] for desc in Descriptors.descList]
        
        table_name = "MoleculeDescriptors"
        return descriptors, feature_names, table_name
    
    def create_features(self, df: pl.DataFrame, batch_size: int = 1000, feature_method = None) -> pl.DataFrame:
        """
        Create numerical features for molecules in the DataFrame using parallel processing and batching

        Args:
            df (pl.DataFrame): DataFrame containing 'FullMolecule_Smiles' column.
            num_workers (int, optional): Number of parallel workers. Defaults to number of CPU cores.
            batch_size (int, optional): Number of SMILES per batch. Defaults to 1000.

        Returns:
            pl.DataFrame: Original DataFrame with appended features.
        """

        try:
            smiles_list = df["FullMolecule_Smiles"].to_list()
        except KeyError as e:
            logging.error(f"Missing 'FullMolecule_Smiles' column: {e}")
            raise

        batches = [smiles_list[i:i + batch_size] for i in range(0, len(smiles_list), batch_size)]
        features = []

        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                for batch_features, feature_names, table_name in executor.map(feature_method, batches):
                    features.extend(batch_features)
                    feature_names = feature_names
                    table_name = table_name
        except Exception as e:
            logging.error(f"Error during parallel feature calculation: {e}")
            raise

        if len(features) != len(smiles_list):
            logging.error(f"Mismatch in features length. Length of Features, Smiles: {len(features)}, {len(smiles_list)}")
            raise ValueError("feature length mismatch.")

        try:
            features_df = pl.DataFrame(features, schema=feature_names)
        except Exception as e:
            logging.error(f"Error creating feature DataFrames: {e}")
            raise

        try:
            results_df = df.hstack(features_df)
        except Exception as e:
            logging.error(f"Error concatenating DataFrames: {e}")
            raise

        return results_df, table_name

    def insert_features(self, table_name:str, df: pl.DataFrame):
        """
        Insert descriptors into the database with upsert functionality.

        If the table exists, perform an upsert (insert or update on conflict).
        If the table does not exist, create the table and then insert the DataFrame.
        Any other errors are raised.

        Args:
            db (str): Database connection string or path.
            table_name (str): Name of the table to insert data into.
            df (pl.DataFrame): DataFrame containing descriptors.

        Raises:
            Exception: If writing to the database fails.
        """
        engine = create_engine(self.db)
        metadata = MetaData()
        try:
            inspector = sql_inspect(engine)
            table_exists = inspector.has_table(table_name)
            logging.info(f"Table '{table_name}' exists: {table_exists}")

            # Reflect the table if it exists
            if table_exists:
                table = Table(table_name, metadata, autoload_with=engine)
                logging.info(f"Reflecting existing table '{table_name}'.")
            else:
                # Create the table using the helper
                self.helper.create_table(df, table_name, self.db)
                table = Table(table_name, metadata, autoload_with=engine)
                logging.info(f"Created and reflected table '{table_name}'.")

            # Get the list of columns in the table
            table_columns = [c.name for c in table.c]
            # Restrict df to only columns in table_columns
            df_to_insert = df.select([col for col in table_columns if col in df.columns])

            with engine.connect() as conn:
                df_to_insert.write_database(table_name=table_name, connection=conn, if_table_exists="append")
                logging.info(f"Inserted records into '{table_name}'.")

        except Exception as e:
            logging.error(f"Failed to write descriptors to database: {e}")
            raise
    
    def main_numerical_features(self):
        logging.info("Starting numerical features creation process...")
        # Create list of featurization methods
        feature_methods = [
            name for name, member in inspect.getmembers(self, predicate=inspect.isfunction)
            if "calculate" in name
        ]

        # Setup chunk info
        num_chunks = self.helper.get_num_rows(self.db) // self.chunk_size
        start_time = datetime.now()
        chunk_num = 0
        logging.info("calculated num_chunks...")
        # Iterate over chunks
        for chunk in range(num_chunks):
            # Load chunk into polars dataframe
            start_row = chunk * self.chunk_size
            if self.dev_mode:
                df = self.helper.small_fetch(self.db)
            else:
                df = self.helper.load_chunk(i=start_row, db=self.db, chunk_size=self.chunk_size)[["MoleculeID", "FullMolecule_Smiles"]]
            chunk_num += 1
            logging.info(f"Starting chunk processing at {start_time}")
            
            # Create each of the feature sets 
            x = 1
            for method_name in feature_methods:
                logging.info(f"running method {x}")
                method = getattr(self, method_name, None)
                if method is None:
                    logging.error(f"Method '{method_name}' not found in NumericalFeatures.")
                    continue
                new_df, table_name = self.create_features(df=df, feature_method=method)
                logging.info(f"inserting features...")
                self.insert_features(table_name=table_name, df=new_df)
                x+=1
            logging.info(f"Finished chunk {chunk_num}/{num_chunks} at {datetime.now()}")
        logging.info(f"Finished chunk featurization at {datetime.now()}")

if __name__ == "__main__":
    featurization = NumericalFeatures(dev_mode=False, db_train=True)
    featurization.main_numerical_features()