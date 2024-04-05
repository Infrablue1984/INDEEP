#!/venv/bin/python3
# -*- coding: utf-8 -*-

__author__ = "Inga Franzen and firstname lastname"
__created__ = "2020"
__date_modified__ = "yyyy/mm/dd"
__version__ = "1.0"

import unittest
from datetime import date

from calculation import interventions
from synchronizer import constants as cs
from synchronizer import synchronizer

# TODO: write further tests


class TestInterventionMaker(unittest.TestCase):
    def setUp(self):
        self.start_date = date(2020, 9, 1)
        self.end_date = date(2020, 12, 31)
        intervention_data = synchronizer.PM.get_path_interventions()
        inter_maker = interventions.InterventionMaker()
        self.interventions = inter_maker.initialize_interventions(intervention_data)
        self.region1 = [1001]


class TestIntervention(unittest.TestCase):
    # TODO: write further tests
    def setUp(self):
        self.start_date = date(2020, 9, 1)
        self.end_date = date(2020, 12, 31)
        self.intervention = interventions.Intervention(
            cs.HOME_DOING, self.start_date, self.end_date
        )

    def test_init(self):
        self.assertFalse(self.intervention.is_activated())

    def test_update_activated(self):
        pass


class TestSubclasses(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_organizer(self):
        pass


if __name__ == "__main__":
    unittest.main()
