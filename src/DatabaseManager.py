from __future__ import annotations

import glob
import logging
import os.path
import re
import sqlite3
from typing import Dict, List, Union

import cv2
from tqdm import tqdm

from src.DNALogging import DNALogging
from src.constant import BACKGROUND, COMPONENT, AUGMENTATION

# logging
DNALogging.config_logging()
logger = logging.getLogger(__name__)


class DatabaseManager:
    __db_connection: sqlite3.Connection
    __cursor: sqlite3.Cursor

    table_selected: str
    col_len: Dict[str, int] = dict()
    table_names: List[str]

    def __init__(self, db_path: str):
        self.__db_connection = sqlite3.connect(db_path)
        self.__cursor = self.__db_connection.cursor()

        # Initialise background table
        self.__cursor.execute("""
            CREATE TABLE IF NOT EXISTS background (
                id INTEGER PRIMARY KEY,
                Background_name TEXT,
                Texture TEXT
            );
        """)

        self.col_len[BACKGROUND] = self.num_of_cols(BACKGROUND)

        # Initialise component table
        self.__cursor.execute("""
            CREATE TABLE IF NOT EXISTS component (
                id INTEGER PRIMARY KEY,
                Raw_image TEXT,
                Sample TEXT,
                Texture TEXT,
                Height INTEGER,
                Width INTEGER 
            );
        """)

        self.col_len[COMPONENT] = self.num_of_cols(COMPONENT)

        self.__cursor.execute("""
            CREATE TABLE IF NOT EXISTS augmentation (
                id INTEGER PRIMARY KEY,
                Image_name TEXT,
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

        self.__cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.table_names = [table_name[0] for table_name in self.__cursor.fetchall()]

    def num_of_cols(self, table_name: str) -> int:
        self.__cursor.execute(f"PRAGMA table_info({table_name})")
        columns = self.__cursor.fetchall()

        return len(columns)

    def select_table(self, db_name: str):
        if db_name not in self.table_names:
            raise Exception(f"Error: Unknown table {db_name} is selected")

        self.table_selected = db_name

        return self

    def query_data(self, condition, columns: List[str] = None) -> Union[List, None]:
        self.__select_table_check()

        if not columns:
            cols = "*"
        else:
            cols = ", ".join(columns)

        self.__cursor.execute(f"SELECT {cols} FROM {self.table_selected} WHERE {condition}")

        results = self.__cursor.fetchall()
        if not results:
            logger.info(f"No results satisfy {condition} in table {self.table_selected}")
            return None

        if len(results) == 1:
            results = list(results[0])
        elif columns and len(columns) == 1:
            results = [s[0] for s in results]

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
            self.__cursor.execute(f"INSERT INTO {self.table_selected} ({column_names}) VALUES ({placeholder})",
                                  column_values)
        except Exception as e:
            raise Exception(f"Error: Insert data into table {self.table_selected} with error {e}.")

        logger.info(f"New record is added")

        self.table_selected = ""
        self.__db_connection.commit()

    def delete_row(self, condition: str):
        self.__select_table_check()

        self.__cursor.execute(f"DELETE FROM {self.table_selected} WHERE {condition}")

        self.table_selected = ""
        self.__db_connection.commit()

    def delete_all_rows(self):
        self.__select_table_check()

        self.__cursor.execute(f"DELETE FROM {self.table_selected}")

        self.table_selected = ""
        self.__db_connection.commit()

    def clean_all_tables(self):
        logger.warning(f"Clean all data in {len(self.table_names)} tables")

        for tn in self.table_names:
            self.select_table(tn).delete_all_rows()

    def drop_all(self):
        for table in self.table_names:
            self.__cursor.execute(f"DROP TABLE IF EXISTS {table};")

        self.table_names = []

        self.__db_connection.commit()

    def get_unique_values(self, column_name: str) -> List:
        self.__select_table_check()

        self.__cursor.execute(f"SELECT DISTINCT {column_name} FROM {self.table_selected}")

        results = self.__cursor.fetchall()
        unique_values = [result[0] for result in results]

        self.table_selected = ""

        return unique_values

    def group_by_column(self, on_column: str, by_column: str) -> Dict:
        self.__select_table_check()

        self.__cursor.execute(
            f"SELECT {by_column}, GROUP_CONCAT({on_column}) FROM {self.table_selected} GROUP BY {by_column}")

        temp_results = self.__cursor.fetchall()
        res_dict = dict()

        for one_class in temp_results:
            res_dict[one_class[0]] = [int(s) for s in one_class[1].split(",")]

        self.table_selected = ""

        return res_dict

    def close_connection(self):
        self.__db_connection.close()

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

            # remove all records to all tables
            self.clean_all_tables()

            return
        else:
            # backgrounds
            logger.info(f"Scan and update table {BACKGROUND}")
            self.__scan_helper(BACKGROUND, data_path)

            # component
            logger.info(f"Scan and update table {COMPONENT}")
            self.__scan_helper(COMPONENT, data_path)

            # augmentation
            logger.info(f"Scan and update table {AUGMENTATION}")
            self.__scan_helper(AUGMENTATION, data_path)

    def __scan_helper(self,
                      table_name: str,
                      data_path: str):
        # condition for different images
        # condition is defined only based on the name of each existing image
        if table_name == BACKGROUND:
            condition_template = "Background_name = '{}' AND Texture = '{}'"
            query_columns = self.select_table(BACKGROUND).__get_column_names(drop=True)
            imgs_path = os.path.join(data_path, table_name)
        elif table_name == COMPONENT:
            condition_template = "Sample = '{}' AND Texture = '{}'"
            query_columns = self.select_table(COMPONENT).__get_column_names(drop=True)
            imgs_path = os.path.join(data_path, table_name, "images")
        else:
            # augmentation TODO
            condition_template = ""
            query_columns = []
            imgs_path = ""

        if not os.path.exists(imgs_path):
            logger.warning(f"Directory {imgs_path} does not exist")
            self.select_table(table_name).delete_all_rows()
        else:
            records_kept = []  # local existing images

            for img in tqdm(glob.glob(os.path.join(imgs_path, "*.png"))):
                column_values = []
                img_name_dir = img.split("/")[-1]

                if table_name in [COMPONENT, BACKGROUND]:
                    column_values.append(img_name_dir)
                    column_values.append(img_name_dir.split("_")[1])
                else:
                    # augmented
                    # TODO
                    pass

                condition = condition_template.format(*column_values)
                records_kept.append(img_name_dir)

                if not self.select_table(table_name).query_data(condition):
                    column_values.insert(0, "N/A")  # Unknown raw image
                    temp_img = cv2.imread(img)
                    column_values += list(temp_img.shape[: 2])  # height and width
                    self.select_table(table_name).insert_data(**dict(zip(query_columns, column_values)))

            records_kept = ', '.join(f"'{value}'" for value in records_kept)
            if table_name == BACKGROUND:
                idx = 0
            elif table_name == COMPONENT:
                idx = 1
            else:
                idx = -1

            self.select_table(table_name).delete_row(f"{query_columns[idx]} NOT IN ({records_kept})")

    @staticmethod
    def __insert_helper(columns: List[str],
                        values: List[str]) -> Dict:
        return dict(zip(columns, values))

    def __get_column_names(self, drop: bool = True) -> List[str]:
        self.__select_table_check()

        self.__cursor.execute(f"PRAGMA table_info({self.table_selected})")
        columns = self.__cursor.fetchall()
        column_names = [column[1] for column in columns]

        self.table_selected = ""

        return column_names if not drop else column_names[1:]

    def read_max_id(self) -> int:
        self.__select_table_check()

        self.__cursor.execute(f"SELECT MAX(id) FROM {self.table_selected}")
        result = self.__cursor.fetchone()[0]

        self.table_selected = ""

        return result if result is not None else 0


if __name__ == "__main__":
    dbm = DatabaseManager("../data/DNA_augmentation")
    dbm.scan_and_update("../dataset")
    dbm.close_connection()
