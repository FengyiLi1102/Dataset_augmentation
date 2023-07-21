from __future__ import annotations

import glob
import logging
import os.path
import pickle
import re
import sqlite3
from copy import copy
from typing import Dict, List, Union, Tuple

import cv2
from rich.progress import track

from src.AugmentedImage import AugmentedImage
from src.DNALogging import DNALogging
from src.constant import BACKGROUND, COMPONENT, AUGMENTED, MOSAICS, CROPPED, RAW
from src.utils import mkdir_if_not_exists

# logging
DNALogging.config_logging()
logger = logging.getLogger(__name__)


class DatabaseManager:
    __db_connection: sqlite3.Connection
    __cursor: sqlite3.Cursor

    table_selected: str
    col_len: Dict[str, int] = dict()
    table_names: List[str]
    training_dataset_name: str = AUGMENTED

    prefix_name: Dict[str, str] = dict()
    condition_templates: Dict[str, str] = dict()

    fixed_tables = [BACKGROUND, MOSAICS, RAW, CROPPED]
    current_all_tables: List[str]

    def __init__(self, db_path: str, training_dataset_name: str = AUGMENTED):
        self.__db_connection = sqlite3.connect(db_path)
        self.__cursor = self.__db_connection.cursor()
        self.training_dataset_name = training_dataset_name

        logger.info(f">>> Connect to the database {db_path}")

        # Initialise mosaic table
        self.__cursor.execute("""
            CREATE TABLE IF NOT EXISTS mosaics (
                id INTEGER PRIMARY KEY,
                Mosaic_name TEXT,
                Texture TEXT,
                Height INTEGER,
                Width INTEGER
            );
        """)

        # self.col_len[MOSAICS] = self.num_of_cols(MOSAICS)

        # Initialise background table
        self.__cursor.execute("""
            CREATE TABLE IF NOT EXISTS background (
                id INTEGER PRIMARY KEY,
                Background_name TEXT,
                Texture TEXT
            );
        """)

        # self.col_len[BACKGROUND] = self.num_of_cols(BACKGROUND)

        # Initialise component table
        self.__cursor.execute("""
            CREATE TABLE IF NOT EXISTS cropped (
                id INTEGER PRIMARY KEY,
                Sample TEXT,
                Raw_image TEXT,
                Morphology TEXT,
                Height INTEGER,
                Width INTEGER 
            );
        """)

        # self.col_len[COMPONENT] = self.num_of_cols(COMPONENT)

        # Initialise raw component table
        self.__cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw (
                id INTEGER PRIMARY KEY,
                Image_name TEXT,
                Height INTEGER,
                Width INTEGER 
            );
        """)

        # self.col_len[RAW] = self.num_of_cols(RAW)

        # Initialise augmented dataset with given dataset name
        self.__cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {training_dataset_name} (
                id INTEGER PRIMARY KEY,
                Image_name TEXT,
                Category TEXT,
                Component_id INTEGER,
                Background_id INTEGER,
                Component_scale REAL,
                Flip TEXT,
                Rotate INTEGER,
                LabelTxt TEXT,
                FOREIGN KEY(Component_id) REFERENCES component(id),
                FOREIGN KEY(Background_id) REFERENCES mosaics(id)
            );
        """)

        # self.col_len[training_dataset_name] = self.num_of_cols(training_dataset_name)

        self.__cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.table_names = [table_name[0] for table_name in self.__cursor.fetchall()]

        self.prefix_name = {
            MOSAICS: "Mosaic_name",
            BACKGROUND: "Background_name",
            RAW: "Image_name",
            CROPPED: "Sample",
            training_dataset_name: "Image_name"
        }

        # for directory-based querying condition template
        self.current_all_tables = self.fixed_tables + [training_dataset_name]

        for table_name in self.current_all_tables:
            self.condition_templates[table_name] = self.prefix_name[table_name] + " = '{}'"

            # get number of columns for each current working tables
            self.col_len[table_name] = self.num_of_cols(table_name)

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

        # logger.info(f"New record is added")

        self.table_selected = ""
        self.__db_connection.commit()

    def delete_row(self, condition: str):
        self.__select_table_check()

        self.__cursor.execute(f"DELETE FROM {self.table_selected} WHERE {condition}")

        # TODO: (improve) display what is deleted; add a query step
        logger.info(f"One row is deleted")

        self.table_selected = ""
        self.__db_connection.commit()

    def delete_all_rows(self):
        self.__select_table_check()

        logger.warning(f"All records in {self.table_selected} are deleted")
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
        logger.info(f">>> Close the connection to the database")
        self.__db_connection.close()

    def __select_table_check(self):
        if not self.table_selected:
            raise Exception("Error: Not select table first.")

    def scan_and_update(self,
                        dataset_root_path: str,
                        data_path: str,
                        training_flag: bool = True,
                        load_cache: bool = True,
                        cache_dir: str = "cache"):
        # TODO: double load the cache in order to update the table (pref)

        # check if provided background mosaics and raw images for cropping exist
        raw_data_exist = self.__directory_to_table(data_path, [MOSAICS, RAW])

        # scan provided dataset root directory
        processed_data_exist = self.__directory_to_table(dataset_root_path,
                                                         [CROPPED, BACKGROUND, self.training_dataset_name])

        # if both directories exist, update records
        if raw_data_exist or processed_data_exist:
            # extract cache files in the root directory
            all_tables = copy(self.current_all_tables)

            # update based on the cache files
            if load_cache:
                logger.info(f">>> Scan based on the cache files")
                caches_paths = glob.glob(os.path.join(dataset_root_path, cache_dir, "*.pkl")) + \
                               glob.glob(os.path.join(data_path, cache_dir, "*.pkl"))

                self.__check_cache_files(caches_paths)

                for cache_path in caches_paths:
                    cache_type = re.match(r"(.*?)_\d{4}_.*", cache_path.split("/")[-1]).group(1)

                    # update the corresponding table
                    logger.info(f">>> Scan and update table {cache_type} from {cache_path}")
                    self.__scan_cache(cache_type, cache_path)

                    # finish updating and remove
                    all_tables.remove(cache_type)

                # clean tables which do not update
                for table_name in all_tables:
                    self.select_table(table_name).delete_all_rows()
            else:
                logger.info(f">>> Scan based on existing images in directories")

                # based on the existing images in the directory
                for table_name in all_tables:
                    if table_name in [MOSAICS, RAW]:
                        load_path = data_path
                    else:
                        load_path = dataset_root_path

                    logger.info(f">>> Scan and update table {table_name} from {load_path}")

                    if table_name == self.training_dataset_name:
                        self.__scan_directory(table_name, os.path.join(load_path, table_name),
                                              training_flag=training_flag)
                    else:
                        self.__scan_directory(table_name, os.path.join(load_path, table_name))
        else:
            # both do noe exist, delete all records
            self.drop_all()

    def __scan_cache(self, cache_type: str, cache_path: str):
        with open(cache_path, "rb") as f:
            # name [str] : Background / Component / Image / AugmentedImage
            dataset = pickle.load(f)

        name_tag = self.prefix_name[cache_type]

        for name, image in dataset.items():
            if not self.select_table(cache_type).query_data(f"{name_tag} = '{name}'"):
                # not in the table and update it
                added_data = self.__scan_cache_helper(cache_type, image)
                self.select_table(cache_type).insert_data(**added_data)

        # delete invalid records in the database
        cached_images = ', '.join(f"'{value}'" for value in list(dataset.keys()))
        self.select_table(cache_type).delete_row(f"{name_tag} NOT IN ({cached_images})")

    def __scan_cache_helper(self,
                            table_name: str,
                            image: Union[BACKGROUND, COMPONENT, AugmentedImage]) -> Dict[str, Union[int, float, str]]:
        table_cols = self.select_table(table_name).__get_column_names(drop=True)

        if table_name == BACKGROUND:
            # Background_name, Texture
            table_values = [image.img_name, image.texture]
        elif table_name == MOSAICS:
            # Mosaic_name, Texture, Height, Width
            table_values = [image.img_name, image.texture, image.img_size[0], image.img_size[1]]
        elif table_name == RAW:
            # Image_name, Height, Width
            table_values = [image.img_name, image.img_size[0], image.img_size[1]]
        elif table_name == CROPPED:
            # Raw_image, Sample, Texture, Height, Width
            table_values = [image.img_name, "N/A", image.morphology, image.img_size[0], image.img_size[1]]
        else:
            # Image_name, Category, Component_id, Background_id, Component_scale, Flip, Rotate, LabelTxt
            table_values = [image.img_name, image.category, image.component_id, image.background_id, image.scale,
                            image.flip, image.rotate, image.label_name]

        return dict(zip(table_cols, table_values))

    def __directory_to_table(self, dir_path: str, tables: List[str]) -> bool:
        if mkdir_if_not_exists(dir_path):
            logger.warning(f"Provided {dir_path} does not exist")
            logger.info(f">>> Directory {dir_path} is created.")

            # remove table records
            for table_name in tables:
                self.select_table(table_name).delete_all_rows()

            return False

        return True

    def __scan_directory(self,
                         table_name: str,
                         dataset_path: str,
                         training_flag: bool = False):
        # TODO: add label txt check

        imgs_paths, labels_paths = self.__img_label_directory(table_name, dataset_path, training_flag)
        query_columns = self.select_table(table_name).__get_column_names(drop=True)
        condition_template = self.condition_templates[table_name]

        for imgs_path in imgs_paths:
            if not os.path.exists(imgs_path):
                logger.warning(f"Directory {imgs_path} does not exist")
                self.select_table(table_name).delete_all_rows()
            else:
                logger.info(f">>> Scan in {imgs_path}")

                records_kept = []  # local existing images
                category = None  # only for ML dataset

                if table_name == self.training_dataset_name:
                    category = imgs_path.split("/")[-2]

                for img_path in track(glob.glob(os.path.join(imgs_path, "*.png"))):
                    img_name_ext = img_path.split("/")[-1]

                    # for query the record in the database
                    condition = condition_template.format(img_name_ext)

                    # later on to delete records in the table not corresponding to the existing files
                    records_kept.append(img_name_ext)

                    # query the database and if the image is not there add it
                    if not self.select_table(table_name).query_data(condition):
                        new_record = self.__form_img_dir_query_con(table_name, img_path, query_columns,
                                                                   category=category)

                        # insert new records
                        self.select_table(table_name).insert_data(**new_record)

                # delete records that no corresponding images exist in the directory
                records_kept = ', '.join(f"'{value}'" for value in records_kept)

                if table_name != self.training_dataset_name:
                    self.select_table(table_name).delete_row(f"{query_columns[0]} NOT IN ({records_kept})")
                else:
                    # augmented
                    self.select_table(table_name).delete_row(
                        f"Category = '{category}' AND {query_columns[0]} NOT IN ({records_kept})")

    @staticmethod
    def __form_img_dir_query_con(table_name: str,
                                 img_path: str,
                                 query_columns: List[str],
                                 category: str = None) -> Dict:
        column_values = []
        img_name_ext = img_path.split("/")[-1]
        height, width = cv2.imread(img_path).shape[: 2]

        column_values.append(img_name_ext)  # Background_name / Mosaic name / Image name / Sample

        if table_name in [BACKGROUND, MOSAICS]:
            column_values.append(img_name_ext.split("_")[1])  # Texture / Morphology

            if table_name == MOSAICS:
                column_values.append(height)  # height
                column_values.append(width)  # width
        elif table_name in [CROPPED, RAW]:
            column_values.append(height)
            column_values.append(width)

            if table_name == CROPPED:
                # for existing image extraction we do not know the raw image where it comes from
                column_values.insert(1, "N/A")  # Raw image
                column_values.insert(2, img_name_ext.split("_")[1])  # Morphology
        else:
            infos = img_name_ext[: -4].split("_")

            # component id, mosaic id, scale, flip, rotation, labelTxt
            column_values.append(category)
            column_values.append(int(infos[1]))
            column_values.append(int(infos[2]))
            column_values.append(float(infos[3]))
            column_values.append(infos[4])
            column_values.append(int(infos[5]))
            column_values.append(img_name_ext[:-4] + ".txt")

        return dict(zip(query_columns, column_values))

    @staticmethod
    def __img_label_directory(table_name: str,
                              dataset_path: str,
                              flag: bool = False) -> Tuple[List[str], List[str]]:
        labels_paths = []

        if table_name in [MOSAICS, RAW, BACKGROUND]:
            imgs_paths = [dataset_path]
        elif table_name == CROPPED:
            imgs_paths = [os.path.join(dataset_path, "images")]
            labels_paths = [os.path.join(dataset_path, "labels")]
        else:
            # augmentation
            # e.g. augmented_2.20_42_18_n_5.png
            #        type   scale bg com flip rotation
            if not flag:
                # just for simple augmentation
                imgs_paths = [os.path.join(dataset_path, "images")]
            else:
                split_list = ["train", "val", "test"]
                imgs_paths = [os.path.join(dataset_path, category, "images") for category in split_list]
                labels_paths = [os.path.join(dataset_path, category, "labelTxt") for category in split_list]

        return imgs_paths, labels_paths

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

    def __check_cache_files(self, caches_paths):
        table_check_list = copy(self.current_all_tables)

        for cache_path in caches_paths:
            cache_type = re.match(r"(.*?)_\d{4}_.*", cache_path.split("/")[-1]).group(1)

            try:
                table_check_list.remove(cache_type)
            except Exception as e:
                raise Exception(
                    f"Given {e}, more than one {cache_type} is given in the directory {cache_path.split('/')[-2]}")


if __name__ == "__main__":
    dbm = DatabaseManager("../data/DNA_augmentation", training_dataset_name="one_chip_dataset")
    dbm.scan_and_update("../test_dataset", "../data")
    # dbm.scan_and_update(dataset_root_path="../test_dataset", data_path="../data", load_cache=False)
    # dbm.drop_all()
    dbm.close_connection()
    #
    # dbm = DatabaseManager("../data/test_dbm", training_dataset_name="one_chip_dataset")
    # dbm.scan_and_update(dataset_root_path="../test_dataset", data_path="../data")
    # dbm.close_connection()
