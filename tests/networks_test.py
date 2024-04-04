#!/venv/bin/python3
# -*- coding: utf-8 -*-

__author__ = "firstname lastname"
__created__ = "2020"
__date_modified__ = "yyyy/mm/dd"
__version__ = "1.0"

import unittest

from calculation import interventions, networks, people
from synchronizer import synchronizer


class TestNetworks(unittest.TestCase):
    def setUp(self):
        self.regions = [1002, 1004]  # 5162,
        self.scale = 100
        self.weekday = 1
        intervention_data = synchronizer.PM.get_path_interventions()
        inter_maker = interventions.InterventionMaker()
        self.interventions = inter_maker.initialize_interventions(intervention_data)
        self.people = people.People(self.regions, scale=self.scale)
        self.timestep = 0
        self.members = [1002655, 10025643, 10023478, 10028899]
        self.id = 1002444
        self.net = networks.Network(self.members, self.id)

    def tearDown(self):
        self.people = None

    def test_init(self):
        pass
        # TODO: write test for every single method in Network class


class TestPublicLocation(unittest.TestCase):
    def setUp(self):
        pass
        # TODO: initialize data for testing

    def tearDown(self):
        self.people = None

    def test_init(self):
        pass
        # TODO: write test for every single method in PublicLocation class


# TODO: write TestClasses for any other subclass of Network class


def suite2():
    suite = unittest.TestSuite()
    suite.addTest(TestNetworks("test_init"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite2())
    # unittest.main()
