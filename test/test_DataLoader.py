import unittest

from src.DataLoader import DataLoader


class MyTestCase(unittest.TestCase):
    def test_loading(self):
        images = DataLoader.load_mosaics_labels("../data", 320)

        self.assertEqual(images.background_img["clean"][0].img_name, "background_0_clean")
        self.assertEqual(images.component_img[0].img_name, "component_0")

        print(images.component_img[0].centre)


if __name__ == '__main__':
    unittest.main()
