import argparse
import unittest

from src.ArgumentParser import ArgumentParser
from src.DataLoader import DataLoader
from src.JobAssigner import JobAssigner


class MyTestCase(unittest.TestCase):
    def test_background_generation(self):
        data_loader = DataLoader.load_mosaics_labels("data/background/", 320)
        job_assigner = JobAssigner.background_job(data_loader, ArgumentParser.arg_background())

        self.assertEqual(len(job_assigner.background_job_pipeline["clean"]), 8)
        self.assertEqual(len(job_assigner.background_job_pipeline["messy"][0]), 64)
        self.assertTrue(job_assigner.background_job_pipeline["noisyL"][0][0] in job_assigner.operations_backgrounds)


if __name__ == '__main__':
    unittest.main()
