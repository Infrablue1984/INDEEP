import unittest

import make_people as p
import network_functions as net
import numpy as np
import pandas as pd
from numpy import testing

from synchronizer import constants as cs


class TestNetworksCase1(unittest.TestCase):
    def setUp(self):
        # self.region = [['01001'],['06533'],['05162'],['03101'],['07111'],['08335']]
        self.regions = ["01001", "05162", "06533"]
        # self.regions = ['05162']
        self.scale = 100
        self.people = p.People(self.regions, scale=self.scale)
        self.net = net.Institution(self.people, self.scale)
        self.age_gr_multi = pd.read_excel(
            "../inputs/households_ages.xls", "Mehrpersonen"
        )
        self.agents_data = {}
        self.agents_data["ids"] = np.arange(12)
        self.agents_data[cs.AGE] = np.array(
            [37, 45, 29, 68, 23, 56, 59, 62, 43, 64, 28, 39]
        )
        self.agents_data[cs.SEX] = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        self.agents_data[cs.Z_CODE] = np.full_like(
            np.arange(12), fill_value="05621", dtype="U5"
        )
        self.agents_data[cs.Z_CODE][:2] = "01001"
        self.agents_data[cs.Z_CODE][8] = "01001"
        self.agents_data["landscapes"] = np.full_like(np.arange(12), fill_value=30)
        self.agents_data[cs.OCCUPATION] = np.array(
            ["p", "e", "e", "e", "p", "p", "e", "e", "p", "p", "e", "e"]
        )

    def test_init(self):
        pass

    def test_set_possible_age_groups_for_other_members(self):
        age_dist_orig = np.array(
            [0.025, 0.025, 0.100, 0.225, 0.25, 0.225, 0.100, 0.025, 0.025]
        )
        age_dist1, age_gr1 = self.net.set_possible_age_groups_for_other_members(
            4, 4, 19, 4, 4, age_dist_orig, self.age_gr_multi
        )
        testing.assert_array_equal(age_gr1, np.array([4, 5, 6, 7, 8]), verbose=True)
        testing.assert_array_equal(
            age_dist1,
            np.array(
                [
                    0.25 / 0.625,
                    0.225 / 0.625,
                    0.100 / 0.625,
                    0.025 / 0.625,
                    0.025 / 0.625,
                ]
            ),
            verbose=True,
        )
        age_dist2, age_gr2 = self.net.set_possible_age_groups_for_other_members(
            19, 4, 19, 4, 4, age_dist_orig, self.age_gr_multi
        )
        testing.assert_array_equal(age_gr2, np.array([15, 16, 17, 18, 19]))
        testing.assert_array_equal(
            age_dist2,
            np.array(
                [
                    0.025 / 0.625,
                    0.025 / 0.625,
                    0.100 / 0.625,
                    0.225 / 0.625,
                    0.25 / 0.625,
                ]
            ),
        )
        age_dist3, age_gr3 = self.net.set_possible_age_groups_for_other_members(
            10, 4, 19, 4, 4, age_dist_orig, self.age_gr_multi
        )
        testing.assert_array_equal(age_gr3, np.array([6, 7, 8, 9, 10, 11, 12, 13, 14]))
        testing.assert_array_equal(
            age_dist3,
            np.array([0.025, 0.025, 0.100, 0.225, 0.25, 0.225, 0.100, 0.025, 0.025]),
        )
        age_dist_orig2 = np.array([0.025, 0.100, 0.225, 0.25, 0.225, 0.100, 0.025])
        age_dist4, age_gr4 = self.net.set_possible_age_groups_for_other_members(
            3, 4, 19, 4, 4, age_dist_orig, self.age_gr_multi
        )
        testing.assert_array_equal(age_gr4, np.array([4, 5, 6, 7]), verbose=True)
        testing.assert_array_equal(
            age_dist4,
            np.array([0.225 / 0.375, 0.100 / 0.375, 0.025 / 0.375, 0.025 / 0.375]),
            verbose=True,
        )
        age_dist5, age_gr5 = self.net.set_possible_age_groups_for_other_members(
            16, 4, 19, 4, 4, age_dist_orig, self.age_gr_multi
        )
        testing.assert_array_equal(
            age_gr5, np.array([12, 13, 14, 15, 16, 17, 18, 19]), verbose=True
        )
        testing.assert_array_equal(
            age_dist5,
            np.array(
                [
                    0.025 / 0.975,
                    0.025 / 0.975,
                    0.100 / 0.975,
                    0.225 / 0.975,
                    0.25 / 0.975,
                    0.225 / 0.975,
                    0.100 / 0.975,
                    0.025 / 0.975,
                ]
            ),
            verbose=True,
        )
        orig_age_dist2 = np.array([0.10, 0.25, 0.30, 0.25, 0.10])
        age_dist6, age_gr6 = self.net.set_possible_age_groups_for_other_members(
            6, 0, 6, 2, 2, orig_age_dist2, self.age_gr_multi
        )
        testing.assert_array_equal(age_gr6, np.array([4, 5, 6]), verbose=True)
        testing.assert_array_almost_equal(
            age_dist6,
            np.array([0.10 / 0.65, 0.25 / 0.65, 0.30 / 0.65]),
            verbose=True,
        )  # test Exceptions  # with self.assertRaisesRegex(AssertionError, "Range must be greater zero, Age group min must be greater age group max and Range of age groups around age must be of same length as age distribution"):  #     self.net.set_possible_age_groups_for_other_members(17, 4, 19, -10, 10, age_dist_orig)

    def test_extract_regions_from_people_class(self):
        regions = self.net.extract_regions_from_people_class(self.people)
        self.assertEqual(regions, [1001, 5162, 6533])
        # self.assertEqual(regions, [5162])

    def test_split_sampled_data_from_agents_data(self):
        # create fantasy agents data
        data = self.agents_data["ids"]
        sub_data = np.array([3, 6, 4])
        (new_agents_data, sampled_data,) = self.net.split_sampled_data_from_agents_data(
            data, sub_data, self.agents_data
        )
        testing.assert_array_equal(sampled_data["ids"], np.array([3, 4, 6]))
        testing.assert_array_equal(sampled_data[cs.AGE], np.array([68, 23, 59]))
        testing.assert_array_equal(
            new_agents_data["ids"], np.array([0, 1, 2, 5, 7, 8, 9, 10, 11])
        )
        testing.assert_array_equal(
            new_agents_data[cs.AGE],
            np.array([37, 45, 29, 56, 62, 43, 64, 28, 39]),
        )

    def test_extract_data_of_specified_indices(self):
        indices = [7, 11, 3, 4]
        new_data = self.net.extract_data_of_specified_indices(indices, self.agents_data)
        testing.assert_array_equal(new_data[cs.SEX], np.array([1, 1, 0, 0]))
        testing.assert_array_equal(new_data[cs.AGE], np.array([68, 23, 62, 39]))

    def test_filter_agents_data_by_subdata(self):
        filtered_data = self.net.filter_agents_data_by_subdata(
            cs.Z_CODE, "05621", self.agents_data
        )
        testing.assert_array_equal(
            filtered_data[cs.AGE], [29, 68, 23, 56, 59, 62, 64, 28, 39]
        )


def suite1():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestNetworksCase1)
    suite.addTests(tests)
    return suite


# def suite2():
#     suite = unittest.TestSuite()
#     suite.addTest(TestSchoolsCase1('test_get_data_from_ids'))
#     return suite


def suite3():
    suite = unittest.TestSuite()
    suite.addTest(TestNetworksCase1("test_extract_regions_from_people_class"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite1())

# unittest.main()
