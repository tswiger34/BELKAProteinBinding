# Import data manipulation libraries
from ..utils.utils import Helpers
import polars as pl
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.Crippen import _pyMolLogP
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional
import multiprocessing
from datetime import datetime


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
            self.df = self.helper.small_fetch(train_db)[["MoleculeID", "FullMolecule_Smiles"]]
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
            
            Example:
            [
                [1,0,1,1,0],   # MACCs keys for molecule 1
                [1,0,0,0,0],   # MACCs keys for molecule 2
                [0,1,0,1,0]... # MACCs keys for molecule 3
            ]

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
            
        return maccs_keys, key_names
    
    @staticmethod
    def calculate_fingerprints():
        """
        Create Morgan fingerprints for molecules.

        TODO: Implement Morgan fingerprints generation.
        """
        pass
    
    @staticmethod
    def calculate_3d_descriptors():
        """
        Create 3D descriptors for molecules.

        TODO: Implement 3D descriptor generation.
        """
        pass

    @staticmethod
    def calculate_descriptors(smiles_batch: List[str]) -> tuple[List[List[Optional[float]]], List[str]]:
        """
        Calculate molecular descriptors for a batch of SMILES strings. 

        Args:
            smiles_batch (List[str]): Batch of SMILES strings.

        Returns:
            tuple[descriptors(List[List[Optional[float]]]), feature_names(str)]: List of descriptors lists and a 
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
        
        return descriptors, feature_names

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
                for batch_features, feature_names in executor.map(feature_method, batches):
                    features.extend(batch_features)
                    feature_names = feature_names

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

        return results_df

    def insert_features(self, db:str, table_name:str, df: pl.DataFrame):
        """
        Insert descriptors into the database.

        Args:
            df (pl.DataFrame): DataFrame containing descriptors.

        Raises:
            Exception: If writing to the database fails.
        """
        try:
            df.write_database(table_name="Train2dDescriptors")
        except Exception as e:
            logging.error(f"Failed to write descriptors to database: {e}")
            raise

if __name__ == "__main__":
    test = NumericalFeatures(dev_mode=True)
    start_time = datetime.now()
    logging.info(f"Starting new method at {start_time}")
    new_df = test.create_features(df=test.df, feature_method=test.calculate_maccs)
    print(new_df.describe())
    logging.info(f"Finished at {datetime.now()}, taking {datetime.now() - start_time}")