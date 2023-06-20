from __future__ import annotations

import os.path
import sqlite3


class DatabaseManager:
    db_connection: sqlite3.Connection
    cursor: sqlite3.Cursor

    table_selected: str
    db_table_names = ["background", "component", "augmentation"]

    def __init__(self, db_name: str):
        db_path = os.path.join("../data", db_name)
        self.db_connection = sqlite3.connect(db_path)
        self.cursor = self.db_connection.cursor()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS background (
                id INTEGER PRIMARY KEY,
                Raw_image TEXT,
                Sample TEXT
            );
        """)

        # Initialise component table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS component (
                id INTEGER PRIMARY KEY,
                Raw_image TEXT,
                Sample TEXT 
            );
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS augmentation (
                id INTEGER PRIMARY KEY,
                Background TEXT,
                Component TEXT,
                Centre_x REAL,
                Centre_y REAL,
                Flip INTEGER,
                Rotate INTEGER
            );
        """)

    def select_db(self, db_name: str):
        if db_name not in self.db_table_names:
            raise Exception(f"Error: Unknown table {db_name} is selected")

        self.table_selected = db_name

        return self

    def insert_data(self, **kwargs):
        self.__select_table_check()

        if self.table_selected in ["background", "component"]:
            self.cursor.execute(f"INSERT INTO {self.table_selected} (Raw_image, Sample) VALUES (?, ?)",
                                (kwargs["Raw_image"], kwargs["Sample"]))
        elif self.table_selected == "augmentation":
            self.cursor.execute(f"INSERT INTO {self.table_selected} "
                                f"(Background, Component, Centre_x, Centre_y, Flip, Rotate) VALUES "
                                f"(?, ?, ?, ?, ?, ?)",
                                (kwargs["Background"], kwargs["Componet"], kwargs["Centre_x"], kwargs["Centre_y"],
                                 kwargs["Flip"], kwargs["Rotate"]))
        else:
            raise Exception(f"Error: Insert data into table {self.table_selected} is incorrect in the format.")

        self.table_selected = None
        self.db_connection.commit()

    def delete_row(self, condition: str):
        self.__select_table_check()

        self.cursor.execute(f"DELETE FROM {self.table_selected} WHERE {condition}")
        self.db_connection.commit()

    def delete_all_rows(self):
        self.__select_table_check()

        self.cursor.execute(f"DELETE FROM {self.table_selected}")
        self.db_connection.commit()

    def __select_table_check(self):
        if not self.table_selected:
            raise Exception("Error: Not select table first.")

    def close_connection(self):
        self.db_connection.close()


if __name__ == "__main__":
    dbm = DatabaseManager("DNA_augmentation")
    dbm.select_db("background").insert_data(Raw_image="test1", Sample="test2")




    dbm.close_connection()
