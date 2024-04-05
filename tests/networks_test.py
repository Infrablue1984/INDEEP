#!/venv/bin/python3
# -*- coding: utf-8 -*-

__author__ = "firstname lastname"
__created__ = "2020"
__date_modified__ = "yyyy/mm/dd"
__version__ = "1.0"

import unittest

from calculation import interventions, networks, people
from synchronizer.synchronizer import PathManager as PM


class TestNetworks(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        self.people = None

    def test_init(self):
        pass


class TestPublicLocation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        self.people = None

    def test_init(self):
        pass


def suite2():
    suite = unittest.TestSuite()
    suite.addTest(TestNetworks("test_init"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite2())
    # unittest.main()
