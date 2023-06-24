from __future__ import annotations

import glob
import logging
import os.path
import sqlite3
from typing import Dict, List
from tqdm import tqdm

from src.DNALogging import DNALogging
from src.constant import BACKGROUND, COMPONENT, AUGMENTATION

# logging
DNALogging.config_logging()
logger = logging.getLogger(__name__)


class DatabaseManager:
    db_connection: sqlite3.Connection
    cursor: sqlite3.Cursor

    table_selected: str
    col_len: Dict[str, int] = dict()
    table_names: List[str]

    def __init__(self, db_name: str):
        db_path = os.path.join("../data", db_name)
        self.db_connection = sqlite3.connect(db_path)
        self.cursor = self.db_connection.cursor()

        # Initialise background table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS background (
                id INTEGER PRIMARY KEY,
                Background_name TEXT,
                Texture TEXT
            );
        """)

        self.col_len[BACKGROUND] = self.num_of_cols(BACKGROUND)

        # Initialise component table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS component (
                id INTEGER PRIMARY KEY,
                Raw_image TEXT,
                Sample TEXT,
                Texture TEXT 
            );
        """)

        self.col_len[COMPONENT] = self.num_of_cols(COMPONENT)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS augmentation (
                id INTEGER PRIMARY KEY,
                Component_id INTEGER,
                Background_id INTEGER,
                Component_scale REAL,
                Centre_x REAL,
                Centre_y REAL,
                Flip INTEGER,
                Rotate INTEGER,
                FOREIGN KEY(Component_id) REFERENCES component(id),
                FOREIGN KEY(Background_id) REFERENCES background(id)
            );
        """)

        self.col_len[AUGMENTATION] = self.num_of_cols(AUGMENTATION)

        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.table_names = [table_name[0] for table_name in self.cursor.fetchall()]

    def num_of_cols(self, table_name: str) -> int:
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = self.cursor.fetchall()

        return len(columns)

    def select_table(self, db_name: str):
        if db_name not in self.table_names:
            raise Exception(f"Error: Unknown table {db_name} is selected")

        self.table_selected = db_name

        return self

    def query_data(self, condition):
        self.__select_table_check()

        self.cursor.execute(f"SELECT * FROM {self.table_selected} WHERE {condition}")

        results = self.cursor.fetchall()
        if not results:
            logger.info(f"No results satisfy {condition} in table {self.table_selected}")
            return None

        self.table_selected = ""

        return results

    def insert_data(self, **kwargs):
        self.__select_table_check()

        column_names = ", ".join(kwargs.keys())
        if len(kwargs.keys()) != self.col_len[self.table_selected] - 1:
            raise Exception(
                f"Error: Not enough data provided to insert the row: expect {self.col_len[self.table_selected]} but "
                f"only {len(kwargs.keys())}")

        placeholder = ", ".join("?" for _ in kwargs)
        column_values = tuple(kwargs.values())

        try:
            self.cursor.execute(f"INSERT INTO {self.table_selected} ({column_names}) VALUES ({placeholder})",
                                column_values)
        except Exception as e:
            raise Exception(f"Error: Insert data into table {self.table_selected} with error {e}.")

        self.table_selected = ""
        self.db_connection.commit()

    def delete_row(self, condition: str):
        self.__select_table_check()

        self.cursor.execute(f"DELETE FROM {self.table_selected} WHERE {condition}")
        self.db_connection.commit()

    def delete_all_rows(self):
        self.__select_table_check()

        self.cursor.execute(f"DELETE FROM {self.table_selected}")
        self.db_connection.commit()

    def drop_all(self):
        for table in self.table_names:
            self.cursor.execute(f"DROP TABLE IF EXISTS {table};")

        self.table_names = []

        self.db_connection.commit()

    def close_connection(self):
        self.db_connection.close()

    def __select_table_check(self):
        if not self.table_selected:
            raise Exception("Error: Not select table first.")

    def scan_and_update(self, data_path: str):
        # scan provided path and compare with the database records
        # not matched image records should be removed from the database
        if not os.path.exists(data_path):
            logger.warning(f">>> Directory {data_path} cannot be found.")
            logger.info(f">>> Directory {data_path} is created.")
            os.mkdir(data_path)
            return
        else:
            if not os.path.exists(os.path.join(data_path, BACKGROUND)):
                logger.warning("Directory background does not exist")
            else:
                for img in tqdm(glob.glob(os.path.join(data_path, BACKGROUND, "*.png"))):
                    img_name_dir = img.split("/")[-1]
                    texture = img_name_dir.split("_")[1]
                    condition = f"Background_name = '{img_name_dir}' AND Texture = '{texture}'"

                    if not self.select_table(BACKGROUND).query_data(condition):
                        self.select_table(BACKGROUND).insert_data(Background_name=img_name_dir, Texture=texture)

            if not os.path.exists(os.path.join(data_path, "augmented")):
                logger.warning("Directory augmented does not exist")
            else:
                pass

        for img in glob.glob(os.path.join(data_path, "component")):
            pass

    def read_max_id(self) -> int:
        self.__select_table_check()

        self.cursor.execute(f"SELECT MAX(id) FROM {self.table_selected}")
        result = self.cursor.fetchone()[0]

        return result if result is not None else 0


if __name__ == "__main__":
    dbm = DatabaseManager("DNA_augmentation")
    dbm.select_table(BACKGROUND).delete_all_rows()
    dbm.scan_and_update("../test_dataset")
    dbm.close_connection()
