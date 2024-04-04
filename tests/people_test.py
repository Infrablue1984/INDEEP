#!/venv/bin/python3
# -*- coding: utf-8 -*-

__author__ = "Felix Rieth and firstname lastname"
__created__ = "2022"
__date_modified__ = "yyyy/mm/dd"
__version__ = "1.0"

import pathlib
import unittest

import numpy as np

from calculation.people import People


class TestPeople(unittest.TestCase):
    def setUp(self):
        my_scale = 100
        my_region_1 = 1002  # pop_size = 246794
        my_region_2 = 5162  # pop_size = 451730
        self.people_1 = People([my_region_1, my_region_2])
        self.people_2 = People([my_region_2, my_region_1], my_scale)

    def tearDown(self):
        pass

    def test_get_pop_size(self):
        self.assertEqual(self.people_1.get_pop_size(), 4517 + 2467)

    def test_filter_data_by_z_code(self):
        my_array = self.people_1.filter_data_by_z_code([5162, 1002], 100)
        np.testing.assert_array_equal(my_array, np.arange(0, 6984))

    def test_update_epidemic_data(self):
        self.people_1._epidemic_data = {
            "infected": np.full(6984, False, dtype=bool),
            "exposed": np.full(6984, False, dtype=bool),
            "infectious": np.full(6984, False, dtype=bool),
        }
        new_epi_data_1 = {"tested": np.full(6984, True, dtype=bool)}
        new_epi_data_2 = {
            "infected": np.full(6984, True, dtype=bool),
            "exposed": np.full(6984, True, dtype=bool),
            "infectious": np.full(6984, True, dtype=bool),
        }
        empty_epi_data = {}
        self.people_1.update_epidemic_data(new_epi_data_1)
        self.assertEqual(self.people_1._epidemic_data, new_epi_data_1)

    def test_get_data_for(self):
        self.people_1._epidemic_data = {
            "infected": np.full(6984, False, dtype=bool),
            "exposed": np.full(6984, False, dtype=bool),
            "infectious": np.full(6984, False, dtype=bool),
        }
        abs_path = pathlib.Path(__file__).parent.absolute()
        idx = self.people_1.filter_data_by_z_code([1002, 5162], 100, abs_path)

        age_array = self.people_1.filter_data_by_idx(
            idx,
            np.load(f"{abs_path}/data_formatted/agents_ages_" + str(100) + ".npy"),
        )

        np.testing.assert_array_equal(self.people_1.get_data_for("age"), age_array)
        np.testing.assert_array_equal(
            self.people_1.get_data_for("infected"), np.full(6984, False, dtype=bool)
        )
        np.testing.assert_array_equal(
            self.people_1.get_data_for("quarantined"), np.zeros(6984, dtype=bool)
        )

        with self.assertRaises(AssertionError):
            self.people_1.get_data_for("dog")

    def test_get_agents_data(self):
        np.testing.assert_equal(
            self.people_1.get_agents_data(), self.people_1._agents_data
        )

    def test_get_public_data(self):
        np.testing.assert_equal(
            self.people_1.get_public_data(), self.people_1._public_data
        )

    def test_get_epidemic_data(self):
        with self.assertRaises(AssertionError):
            self.people_1.get_epidemic_data()
        epi_data = {
            "infected": np.full(6984, False, dtype=bool),
            "exposed": np.full(6984, False, dtype=bool),
            "infectious": np.full(6984, False, dtype=bool),
        }
        self.people_1.update_epidemic_data(epi_data)
        np.testing.assert_equal(self.people_1.get_epidemic_data(), epi_data)

    def test_get_id_from_index(self):
        self.assertEqual(self.people_1.get_id_from_index(2466), 100202466)
        self.assertEqual(self.people_1.get_id_from_index(2467), 516200000)
        self.assertEqual(self.people_1.get_id_from_index(6983), 516204516)

    def test_get_information_about_agent(self):
        epi_data = {
            "infected": np.full(6984, False, dtype=bool),
            "exposed": np.full(6984, False, dtype=bool),
            "infectious": np.full(6984, False, dtype=bool),
        }
        self.people_1.update_epidemic_data(epi_data)
        id = 100200000  # the first agent for self.people_1
        self.assertEqual(self.people_1.get_information_about_agent(id, "infected"), 0)

        new_epi_data = {
            "infected": np.concatenate(
                (np.array([True]), np.full(6983, False, dtype=bool)), axis=None
            ),
            "exposed": np.full(6984, False, dtype=bool),
            "infectious": np.full(6984, False, dtype=bool),
        }
        self.people_1.update_epidemic_data(new_epi_data)
        self.assertEqual(self.people_1.get_information_about_agent(id, "infected"), 1)

        abs_path = pathlib.Path(__file__).parent.absolute()
        idx = self.people_1.filter_data_by_z_code([1002, 5162], 100, abs_path)
        age_array = self.people_1.filter_data_by_idx(
            idx,
            np.load(f"{abs_path}/data_formatted/agents_ages_" + str(100) + ".npy"),
        )

        self.assertEqual(
            self.people_1.get_information_about_agent(id, "age"), age_array[0]
        )
        self.assertEqual(
            self.people_1.get_information_about_agent(id, "quarantined"), 0
        )

        with self.assertRaises(AssertionError):
            self.people_1.get_information_about_agent(id, "dog")

    def test_get_id_from_array(self):
        bool_array = np.concatenate(
            (np.array([True]), np.full(6983, False, dtype=bool)), axis=None
        )
        self.assertEqual(self.people_1.get_id_from_array(bool_array), {"100200000"})

        with self.assertRaises(IndexError):
            self.people_1.get_id_from_array(np.full(6, False, dtype=bool))

        with self.assertRaises(TypeError):
            self.people_1.get_id_from_array(np.full(6984, 1, dtype=int))

        with self.assertRaises(TypeError):
            self.people_1.get_id_from_array("Dog")


if __name__ == "__main__":
    unittest.main()
    # suite1 = unittest.defaultTestLoader.loadTestsFromTestCase(TestPeople)
    # unittest.TextTestRunner().run(suite1)

#
# def suite1():
#     suite = unittest.TestSuite()
#     suite.addTest(TestPeopleCase1())
#     #suite.addTest(TestPeopleCase1('test_make_sexes'))
#     return suite

#
# def suite2():
#     suite = unittest.TestSuite()
#     tests = ['test_make_ages', 'test_make_sexes']
#     suite(map(TestPeopleCase1, tests))
#     return suite()
#
#
# def suite3():
#     suite = unittest.TestSuite()
#     tests = ['test_make_ids']
#     suite(map(TestPeopleCase1, tests))
#     return suite()
#
