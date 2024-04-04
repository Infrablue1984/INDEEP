import time
import unittest

import numpy as np
from numpy import testing

# import Activities
from data_factory import synchronizer as sync


class TestActivitiesCase1(unittest.TestCase):
    def setUp(self):
        # self.region = [['01001'],['06533'],['05162'],['03101'],['07111'],['08335']]
        t4 = time.time()
        self.region = [5162, 1002]
        self.scale = 100
        self.cal = sync.Controller(self.region, self.scale)
        t6 = time.time_ns()
        # print("time for people: {}, time for workplaces {}".format(t5-t4, t6-t5))

    def test_init(self):
        self.cal.calculate_contacts_per_age_group()


def suite1():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestActivitiesCase1)
    suite.addTests(tests)
    return suite


def suite2():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestActivitiesCase1)
    suite.addTests(tests)
    return suite


if __name__ == "__main__":
    unittest.main()
