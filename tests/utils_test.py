from scripts.utils.utils import Helpers
import logging
logging.basicConfig(level=logging.INFO)
class UtilsTest:

    def __init__(self):
        self.helper = Helpers()
        self.train_db, self.test_db = self.helper.get_dbs()
    
    def test_get_dbs(self):
        train_db, test_db = self.helper.get_dbs()
        assert train_db == "sqlite:///D:/sqlite/BELKA/train_db.db"
        assert test_db == "sqlite:///D:/sqlite/BELKA/test_db.db"

    def test_get_row_count(self):
        train_db, test_db = self.helper.get_dbs()
        max_id = self.helper.get_num_rows(train_db)
        assert type(max_id) == int, f"the get_row_count function returned type: {type(max_id)}"
        assert max_id == 98416004, f"the get_row_count function returned type: {max_id}"

    def test_create_table(self):
        df = self.helper.small_fetch(self.train_db)
        self.helper.create_table(df=df, table_name="TempTable", db=self.train_db)

    def test_load_chunk(self):
        # self.helper.load_chunk()
        pass

    def test_small_fetch(self):
        # self.helper.small_fetch()
        pass
    
    def main_test(self):
        logging.info("Beginning tests")
        self.test_get_dbs()
        self.test_get_row_count()
        self.test_small_fetch()
        self.test_load_chunk()
        self.test_create_table()

if __name__ == "__main__":
    tester = UtilsTest()
    tester.main_test()