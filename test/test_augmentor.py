import unittest

from src.ArgumentParser import ArgumentParser
from src.Augmentor import Augmentor
from src.DataLoader import DataLoader
from src.JobAssigner import JobAssigner


class MyTestCase(unittest.TestCase):
    data_loader = DataLoader.load_mosaics_labels("../data/background/", 320)
    job_assigner = JobAssigner.background_job(data_loader, ArgumentParser.arg_background())

    def test_augment_backgrounds(self):
        Augmentor.produce_backgrounds(self.data_loader, self.job_assigner)


if __name__ == '__main__':
    unittest.main()
