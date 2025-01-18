# Import data manipulation libraries
from ..utils.utils import Helpers
import polars as pl
import sqlite3 as sql
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

    def __init__(self, df:pl.DataFrame, dev_mode: bool, num_workers:int = None):
        helper = Helpers()
        self.dev_mode = dev_mode
        # Set number of workers for parallel processing
        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count()
        else: self.num_workers = num_workers
        
        # Get data from the database
        train_db, test_db = helper.get_dbs()
        if dev_mode:
            self.df = helper.small_fetch(train_db)
        else: self.df = df

    def create_maccs(self):
        """
        Create MACCS keys for molecules.

        TODO: Implement MACCS key generation.
        """
        pass

    @staticmethod
    def calculate_features(smiles_batch: List[str]) -> Tuple[List[List[Optional[float]]], List[Optional[float]]]:
        """
        Calculate molecular descriptors and LogP values for a batch of SMILES strings.

        Args:
            smiles_batch (List[str]): Batch of SMILES strings.

        Returns:
            Tuple[List[List[Optional[float]]], List[Optional[float]]]:
                - List of descriptor lists.
                - List of LogP values.
        """
        descriptors = []
        mol_logps = []
        for smile in smiles_batch:
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol:
                    desc = [func(mol) for _, func in Descriptors.descList]
                    descriptors.append(desc)
                    mol_logps.append(_pyMolLogP(mol))
                else:
                    descriptors.append([None] * len(Descriptors.descList))
                    mol_logps.append(None)
            except Exception as e:
                logging.error(f"Error processing SMILES '{smile}': {e}")
                descriptors.append([None] * len(Descriptors.descList))
                mol_logps.append(None)
        return descriptors, mol_logps

    def create_descriptors(self, df: pl.DataFrame, batch_size: int = 1000) -> pl.DataFrame:
        """
        Create numerical descriptors and LogP values for molecules in the DataFrame.

        Args:
            df (pl.DataFrame): DataFrame containing 'FullMolecule_Smiles' column.
            num_workers (int, optional): Number of parallel workers. Defaults to number of CPU cores.
            batch_size (int, optional): Number of SMILES per batch. Defaults to 1000.

        Returns:
            pl.DataFrame: Original DataFrame with appended descriptors and 'mol_logp'.
        """

        try:
            smiles_series = self.df["FullMolecule_Smiles"] if self.dev_mode else df["FullMolecule_Smiles"]
            smiles_list = smiles_series.to_list()
        except KeyError as e:
            logging.error(f"Missing 'FullMolecule_Smiles' column: {e}")
            raise

        batches = [smiles_list[i:i + batch_size] for i in range(0, len(smiles_list), batch_size)]
        descriptors = []
        mol_logps = []

        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                for batch_desc, batch_logp in executor.map(self.calculate_features, batches):
                    descriptors.extend(batch_desc)
                    mol_logps.extend(batch_logp)
        except Exception as e:
            logging.error(f"Error during parallel descriptor calculation: {e}")
            raise

        if len(descriptors) != len(smiles_list) or len(mol_logps) != len(smiles_list):
            logging.error("Mismatch in descriptors or LogP lengths.")
            raise ValueError("Descriptor or LogP length mismatch.")

        descriptor_names = [desc[0] for desc in Descriptors.descList]
        try:
            descriptors_df = pl.DataFrame(descriptors, schema=descriptor_names)
            mol_logp_df = pl.DataFrame({"mol_logp": mol_logps})
        except Exception as e:
            logging.error(f"Error creating descriptor DataFrames: {e}")
            raise

        try:
            results_df = df.hstack(descriptors_df).hstack(mol_logp_df)
        except Exception as e:
            logging.error(f"Error concatenating DataFrames: {e}")
            raise

        return results_df

    def insert_descriptors(self, df: pl.DataFrame):
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
    new_df = test.create_descriptors_optimized(test.dev_df)
    print(new_df.describe())
    logging.info(f"Finished at {datetime.now()}, taking {datetime.now() - start_time}")