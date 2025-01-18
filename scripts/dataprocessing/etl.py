import polars as pl
import sqlite3 as sql
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Starting ETL process...")

class DataETL:
    """
    This class manages the ETL 
    
    Steps:
    - Initialize file paths for the parque
    """
    def __init__(self):
        """
        Initializes the data_etl class, including the train/test connections, cursors, and file paths, and chunk size.
        After everything is initialized, it runs the main funciton.
        """
        logging.info("Initializing ETL Class")
        self.train_conn = sql.connect("D:/sqlite/BELKA/train_db.db")
        self.train_cursor = self.train_conn.cursor()
        self.train_path = 'C:/Users/Travis/Downloads/train.parquet'
        self.chunk_size = 500000
        self.test_conn = sql.connect("D:/sqlite/BELKA/test_db.db")
        self.test_cursor = self.test_conn.cursor()
        self.test_path = 'C:/Users/Travis/Downloads/test.parquet'
        self.main()

    def train_transform(self, df:pl.DataFrame) -> pl.DataFrame:
        """
        Transforms the train dataset:
        - Adds BindsEPH, BindsBRD, BindsALB columns.
        - Updates the columns based on TargetProtein and binds values.
        - Aggregates data by molecule_smiles.

        Args:
        - df: a polars dataframe containing the loaded chunk

        Returns:
        - df: a polars dataframe with the transformed chunk
        """

        ## Create new columns for BindsEPH, BindsBRD, BindsALB
        df = df.with_columns([
            pl.lit(0).alias("BindsEPH"),
            pl.lit(0).alias("BindsBRD"),
            pl.lit(0).alias("BindsALB")
        ])

        ## Identify the target protein and whether or not it binds, replace resepective column with value
        df = df.with_columns([
            (pl.when(pl.col("protein_name") == "sEH").then(pl.col("binds")).otherwise(pl.col("BindsEPH"))).alias("BindsEPH"),
            (pl.when(pl.col("protein_name") == "BRD4").then(pl.col("binds")).otherwise(pl.col("BindsBRD"))).alias("BindsBRD"),
            (pl.when(pl.col("protein_name") == "HSA").then(pl.col("binds")).otherwise(pl.col("BindsALB"))).alias("BindsALB")
        ])

        ## Aggregate by molecule_smiles
        df = df.group_by("molecule_smiles").agg(
            pl.col("buildingblock1_smiles").first(),
            pl.col("buildingblock2_smiles").first(),
            pl.col("buildingblock3_smiles").first(),
            pl.col("BindsEPH").max(),
            pl.col("BindsBRD").max(),
            pl.col("BindsALB").max()
            )

        return df

    def load_train_data(self, df: pl.DataFrame):
            """
            This loads the transformed data into the SQLite training database. The query is an upsert query where it inserts the rows if the FullMolecule_Smiles unique constraint 
            is not violated. If there is a conflict due to the the unique constraint being violated, it will update the molecule with the max of each of the bonding columns from the
            new data and the previously loaded data. This will avoid overwriting correct binding data while updating cases that are needed.           

            Args:
            - df: the transformed polars data frame from the train_transformation method
            - batch_size: an integer for the batch size for each cursor execution

            Returns:
            None
            """

            ## Create upsert query
            query = """
                INSERT INTO TrainSMILES (BuildingBlock1_Smiles, BuildingBlock2_Smiles, BuildingBlock3_Smiles, FullMolecule_Smiles, BindsEPH, BindsBRD, BindsALB)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (FullMolecule_Smiles) DO UPDATE SET
                    BindsEPH = MAX(BindsEPH, excluded.BindsEPH),
                    BindsBRD = MAX(BindsBRD, excluded.BindsBRD),
                    BindsALB = MAX(BindsALB, excluded.BindsALB);
                """
            
            ## Convert Polars DataFrame to a list of tuples
            data = df.select([
                    pl.col("buildingblock1_smiles"),
                    pl.col("buildingblock2_smiles"),
                    pl.col("buildingblock3_smiles"),
                    pl.col("molecule_smiles"),
                    pl.col("BindsEPH"),
                    pl.col("BindsBRD"),
                    pl.col("BindsALB")
                ]).to_numpy().tolist()
            try:
                self.train_cursor.executemany(query, data)
                self.train_conn.commit()
            except sql.Error as e:
                logging.error(f"Error inserting batch: {e}")
                self.train_conn.rollback()

    def test_load(self, df:pl.DataFrame):
        """
        This loads the transformed data into the SQLite testing database. The query is just an insert query because there is no unique constraint needed and the 
        test data is small enough to not need to be handled in batches. The test data does not have a binding indicator as it is just supposed to be used to test if the 
        models can be ran on the competition test data. This way the test data remains hidden. This also means no transformations will be performed.

        Args:
        - df: a polars data frame containing the test data

        Returns:
        None
        """
        ## Insert query
        query = """
                INSERT INTO TestSMILES (FullMolecule_Smiles, BuildingBlock1_Smiles, BuildingBlock2_Smiles, BuildingBlock3_Smiles, TargetProtein)
                VALUES (?, ?, ?, ?, ?)
                """
        
        ## Convert Polars DataFrame to a list of tuples
        data = df.select([
                pl.col("molecule_smiles"),
                pl.col("buildingblock1_smiles"),
                pl.col("buildingblock2_smiles"),
                pl.col("buildingblock3_smiles"),
                pl.col("protein_name"),
            ]).to_numpy().tolist()
        
        ## Insert data into the SQL database
        try:
            self.test_cursor.executemany(query, data)
            self.test_conn.commit()
        except sql.Error as e:
            logging.error(f"Error inserting batch: {e}")
            self.test_conn.rollback()


    def main(self):
        try:
            ## Train ETL
            logging.info("Starting Train ETL...")
            # Create a LazyFrame for the parquet file
            lazy_frame = pl.scan_parquet(self.train_path)
            
            # Create a list of chunks
            logging.info("Scan Complete, beginning chunk calculations")
            num_rows = lazy_frame.count().collect()["id"][0]
            batch_count =  num_rows // self.chunk_size + (1 if num_rows % self.chunk_size > 0 else 0)

            # Iterate through data
            logging.info("Chunk calculations complete, beginning iterations")
            for i in range(batch_count):
                chunk = lazy_frame.slice(i * self.chunk_size, self.chunk_size).collect()
                logging.info(f"Loaded chunk {i} of {batch_count}, beginning transformations")   
                # Apply transformations to the chunk
                chunk_transformed = self.train_transform(chunk)
                logging.info("Transformations complete, beginninging load process")
                # Load the transformed data 
                self.load_train_data(chunk_transformed)
                logging.info("Loading complete for this chunk")

            ## Test ETL
            logging.info("Starting Test ETL...")
            test_df = pl.read_parquet(self.test_path)
            self.test_load(test_df)

        except Exception as e:
            logging.error(f"ETL process failed: {e}")
        finally:
            self.train_conn.close()
            self.test_conn.close()

if __name__ == "__main__":
    DataETL()