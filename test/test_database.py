import unittest

from src.DatabaseManager import DatabaseManager


class MyTestCase(unittest.TestCase):

    def test_table_create(self):
        test_db = DatabaseManager("test_db")
        test_db.select_table("background")
        test_db.select_table("component")
        test_db.select_table("augmentation")
        test_db.close_connection()

    def test_insert_data(self):
        test_db = DatabaseManager("test_db")

        test_db.select_table("background").insert_data(Background_name="test1", Texture="test2")
        test_db.select_table("component").insert_data(Raw_image="test1", Sample="test2", Texture="test3")
        augmentation_row = {"Component_id": 1, "Background_id": 1, "Centre_x": 1.1, "Centre_y": 1.2, "Flip": 0,
                            "Rotate": 110}
        test_db.select_table("augmentation").insert_data(**augmentation_row)

        self.assertEqual(test_db.select_table("background").query_data("Background_name = 'test1'")[0][0], 1)
        self.assertEqual(test_db.select_table("background").query_data("Texture = 'test2'")[0][0], 1)
        self.assertEqual(test_db.select_table("augmentation").query_data("Centre_x = 1.1")[0][0], 1)
        self.assertEqual(test_db.select_table("augmentation").query_data("Rotate = 110")[0][0], 1)

        test_db.select_table("background").delete_row("id = 1")
        test_db.select_table("component").delete_row("id = 1")
        test_db.select_table("augmentation").delete_row("id = 1")

        test_db.close_connection()


if __name__ == '__main__':
    unittest.main()

    test_db = DatabaseManager("test_db")
    test_db.drop_all()
    test_db.close_connection()
