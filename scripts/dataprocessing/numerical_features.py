# Import data manipulation libraries
from ..utils.utils import Helpers
import polars as pl
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, MACCSkeys, Descriptors3D
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Crippen import _pyMolLogP
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional
import multiprocessing
from datetime import datetime
import inspect
from sqlalchemy import create_engine, inspect as sql_inspect
from sqlalchemy.exc import SQLAlchemyError


logging.basicConfig(level=logging.INFO)
logging.info("Starting numerical features creation process...")

class NumericalFeatures:
    """
    Handles the creation of numerical features for molecular data.

    Args:
        dev_mode (bool): Enable development mode with a subset of data.
    """

    def __init__(self, dev_mode: bool, num_workers:int = None):
        self.helper = Helpers()
        self.dev_mode = dev_mode
        # Set number of workers for parallel processing
        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count()
        else: self.num_workers = num_workers
        
        # Get data from the database
        train_db, test_db = self.helper.get_dbs()    
        if dev_mode:
            self.db = train_db
            self.df = self.helper.small_fetch(self.db)[["MoleculeID", "FullMolecule_Smiles"]]
        # else: self.df = df
    
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
            smiles_list = self.df["FullMolecule_Smiles"].to_list()
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
        Insert descriptors into the database.

        Args:
            df (pl.DataFrame): DataFrame containing descriptors.

        Raises:
            Exception: If writing to the database fails.
        """
        engine = create_engine(self.db)
        try:
            inspector = sql_inspect(engine)
            if inspector.has_table(table_name):
                conn = engine.connect()
                df.write_database(table_name=table_name, connection=conn, if_table_exists="append")
                conn.close()
            else:
                self.helper.create_table(df, table_name=table_name, engine=engine)
                conn = engine.connect()
                df.write_database(table_name=table_name, connection=conn, if_table_exists="append")
                conn.close()
        except Exception as e:
            logging.error(f"Failed to write descriptors to database: {e}")
            conn.close()
            raise

if __name__ == "__main__":
    test = NumericalFeatures(dev_mode=True)
    feature_methods = [
        name for name, member in inspect.getmembers(NumericalFeatures, predicate=inspect.isfunction)
        if "calculate" in name
    ]
    start_time = datetime.now()
    logging.info(f"Starting new method at {start_time}")
    print(feature_methods)
    for method_name in feature_methods:
        method = getattr(test, method_name, None)
        if method is None:
            logging.error(f"Method '{method_name}' not found in NumericalFeatures.")
            continue
        new_df, table_name = test.create_features(df=test.df, feature_method=method)
        print(new_df.describe())
        test.insert_features(table_name=table_name, df=test.df)
    logging.info(f"Finished at {datetime.now()}, taking {datetime.now() - start_time}")