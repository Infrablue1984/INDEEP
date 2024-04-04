import time
import unittest

import make_people as p
import networks as net
import numpy as np
import pandas as pd
from make_schools import Schools
from numpy import testing

from synchronizer import constants as cs


class TestSchoolsCase1(unittest.TestCase):
    def setUp(self):
        # self.region = [['01001'],['06533'],['05162'],['03101'],['07111'],['08335']]
        self.region = [5162]
        self.people1 = p.People(self.region, scale=100)
        self.s1 = Schools(self.people1, 100)
        self.test_ids = ["05162_U_5", "05162_U_40", "05162_U_52", "05162_U_76"]
        self.test_ages = [14, 38, 5, 54]
        self.test_occupations = ["p", "p", "k", "e"]

    def test_init(self):
        # all pupil in one class must have same age
        for cl_id in self.s1.sub_members:
            if cl_id < 300000000:
                ids = self.s1.sub_members[cl_id]
                ages = self.people1.get_data_from_ids(ids, cs.AGE)
                ages = set(ages)
                self.assertLessEqual(
                    len(ages), 4
                )  # as three teachers per class wth different age

    def test_determine_num_of_schools_and_classes_per_year(self):
        # test1
        cl_size = 23
        num_p = 253
        sc_size = 143
        years = 4
        (dist, probs, num_s,) = self.s1.determine_num_of_schools_and_classes_per_year(
            cl_size, num_p, sc_size, years
        )
        self.assertEqual(dist, [1, 2])
        self.assertEqual(probs, [0.45, 0.55])
        self.assertEqual(num_s, 2)

        # test2
        cl_size = 21
        num_p = 1553
        sc_size = 252
        years = 4
        (dist, probs, num_s,) = self.s1.determine_num_of_schools_and_classes_per_year(
            cl_size, num_p, sc_size, years
        )
        self.assertEqual(dist, [3])
        self.assertEqual(probs, [1])
        self.assertEqual(num_s, 7)

        # test3
        cl_size = 21
        num_p = 163
        sc_size = 284
        years = 4
        (dist, probs, num_s,) = self.s1.determine_num_of_schools_and_classes_per_year(
            cl_size, num_p, sc_size, years
        )
        self.assertEqual(dist, [3, 4])
        self.assertEqual(probs, [0.62, 0.38])
        self.assertEqual(num_s, 1)

    def test_add_teachers(self):
        sc_id = 20
        num_t = 3
        self.s1.members[sc_id] = np.arange(0)
        for cl_id in range(50):
            self.s1.sub_members[sc_id + cl_id] = np.arange(0)
        # create fantasy agents data
        agents_data = {}
        agents_data["ids"] = np.arange(12)
        agents_data[cs.AGE] = np.array([37, 45, 29, 68, 23, 56, 59, 62, 43, 64, 28, 39])
        agents_data[cs.SEX] = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        agents_data[cs.Z_CODE] = np.full_like(
            np.arange(12), fill_value="05621", dtype="U5"
        )
        agents_data["landscapes"] = np.full_like(np.arange(12), fill_value=30)
        agents_data[cs.OCCUPATION] = np.array(
            ["p", "e", "e", "e", "p", "p", "e", "e", "p", "p", "e", "e"]
        )

        agents_data = self.s1.add_teachers(sc_id, num_t, agents_data)
        sampled_t = self.s1.members[sc_id]
        all_t = np.hstack((agents_data["ids"], sampled_t))
        all_t = np.sort(all_t)
        testing.assert_array_equal(len(agents_data["ids"]), 9)
        testing.assert_array_equal(all_t, np.arange(12))

        # test Exceptions
        num_t = 10
        with self.assertRaisesRegex(
            AssertionError, "Not enough teachers in the system"
        ):
            self.s1.add_teachers(sc_id, num_t, agents_data)

    def test_adopt_class_structure(self):
        cl_size, cl_per_y = self.s1.adopt_class_structure(21, 3, 100, last=True)
        self.assertEqual(cl_size, 25)
        self.assertEqual(cl_per_y, 4)

        cl_size, cl_per_y = self.s1.adopt_class_structure(21, 3, 106, last=True)
        self.assertEqual(cl_size, 22)
        self.assertEqual(cl_per_y, 5)

        cl_size, cl_per_y = self.s1.adopt_class_structure(21, 3, 106, last=False)
        self.assertEqual(cl_size, 21)
        self.assertEqual(cl_per_y, 3)

        cl_size, cl_per_y = self.s1.adopt_class_structure(15, 3, 100, last=True)
        self.assertEqual(cl_size, 17)
        self.assertEqual(cl_per_y, 6)

        cl_size, cl_per_y = self.s1.adopt_class_structure(26, 3, 60, last=True)
        self.assertEqual(cl_size, 30)
        self.assertEqual(cl_per_y, 2)

        cl_size, cl_per_y = self.s1.adopt_class_structure(15, 3, 43, last=True)
        self.assertEqual(cl_size, 15)
        self.assertEqual(cl_per_y, 3)

        cl_size, cl_per_y = self.s1.adopt_class_structure(25, 3, 43, last=True)
        self.assertEqual(cl_size, 22)
        self.assertEqual(cl_per_y, 2)

        cl_size, cl_per_y = self.s1.adopt_class_structure(25, 3, 15, last=True)
        self.assertEqual(cl_size, 15)
        self.assertEqual(cl_per_y, 1)

        cl_size, cl_per_y = self.s1.adopt_class_structure(25, 3, 35, last=True)
        self.assertEqual(cl_size, 35)
        self.assertEqual(cl_per_y, 1)

        cl_size, cl_per_y = self.s1.adopt_class_structure(25, 3, 44, last=True)
        self.assertEqual(cl_size, 22)
        self.assertEqual(cl_per_y, 2)

        cl_size, cl_per_y = self.s1.adopt_class_structure(15, 2, 46, last=True)
        self.assertEqual(cl_size, 16)
        self.assertEqual(cl_per_y, 3)

    def test_extract_school_data_from_people_class(self):
        all_agents_data = self.s1.extract_pupils_and_edus_from_people_class(
            self.people1
        )
        idx = np.argwhere(
            np.where(np.isin(all_agents_data["ids"], self.test_ids), 1, 0)
        )
        ages = all_agents_data[cs.AGE][idx].reshape(len(idx))
        testing.assert_array_equal(ages, self.test_ages)


class TestSchoolsCase2(unittest.TestCase):
    def setUp(self):
        # self.region = [['01001'],['06533'],['05162'],['03101'],['07111'],['08335']]
        self.region = [1002, 5162]
        self.people2 = p.People(self.region, scale=10)
        self.s2 = Schools(self.people2, 10)
        test = 1

    def test_init(self):
        # all pupil in one class must have same age
        for cl_id in self.s2.sub_members:
            if cl_id < 3000000000:
                ids = self.s2.sub_members[cl_id]
                ages = self.people2.get_data_from_ids(ids, cs.AGE)
                ages = set(ages)
                self.assertLessEqual(
                    len(ages), 4
                )  # as three teachers per class wth different age

        # test 1 ensure school sizes
        filt = (self.s2.df_networks["type"] == "primary") & (
            self.s2.df_networks["z_code"] == 5162
        )
        sum_p = self.s2.df_networks.loc[filt, "size"][:-1].sum()
        num_s = (
            len(
                self.s2.df_networks.loc[
                    filt,
                ]
            )
            - 1
        )
        mid_size = sum_p / num_s
        self.assertLessEqual(mid_size, 180 * 350 / 250 + 30)
        self.assertGreaterEqual(mid_size, 180 * 350 / 250 - 30)

        # test 2 ensure school sizes
        filt = (self.s2.df_networks["type"] == "primary") & (
            self.s2.df_networks["z_code"] == 1002
        )
        sum_p = self.s2.df_networks.loc[filt, "size"][:-1].sum()
        num_s = (
            len(
                self.s2.df_networks.loc[
                    filt,
                ]
            )
            - 1
        )
        mid_size = sum_p / num_s
        self.assertLessEqual(mid_size, 180 * 230 / 250 + 30)
        self.assertGreaterEqual(mid_size, 180 * 230 / 250 - 30)


class TestSchoolsCase3(unittest.TestCase):
    def setUp(self):
        self.region = [2000]
        t1 = time.time()
        self.people3 = p.People(self.region, scale=10)
        t2 = time.time()
        print("time for people class region 2000: " + str(t2 - t1))
        self.s3 = Schools(self.people3, 10)

    def test_init(self):
        # all pupil in one class must have same age
        for cl_id in self.s3.sub_members:
            if cl_id < 3000000000:
                ids = self.s3.sub_members[cl_id]
                ages = self.people3.get_data_from_ids(ids, cs.AGE)
                ages = set(ages)
                self.assertLessEqual(
                    len(ages), 4
                )  # as three teachers per class wth different age

        # test 1 ensure school sizes
        filt = (self.s3.df_networks["type"] == "primary") & (
            self.s3.df_networks["z_code"] == 2000
        )
        sum_p = self.s3.df_networks.loc[filt, "size"][:-1].sum()
        num_s = (
            len(
                self.s3.df_networks.loc[
                    filt,
                ]
            )
            - 1
        )
        mid_size = sum_p / num_s
        self.assertLessEqual(mid_size, 180 * 290 / 250 + 30)
        self.assertGreaterEqual(mid_size, 180 * 290 / 250 - 30)

        # # test 1 ensure school sizes for all regions
        # for region in self.region:
        #     filt = (self.s2.df_networks['type'] == 'primary') & (self.s2.df_networks['z_code'] == int(region))
        #     sc_size = self.s2.fed_sc_size[int(region/1000)]
        #     sum_p = self.s2.df_networks.loc[filt, 'size'][:-1].sum()
        #     num_s = len(self.s2.df_networks.loc[filt,]) - 1
        #     mid_size = sum_p / num_s
        #     self.assertLessEqual(mid_size, 180 * sc_size / 250 + 30)
        #     self.assertGreaterEqual(mid_size, 180 * sc_size / 250 - 30)


def suite1():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestSchoolsCase1)
    suite.addTests(tests)
    return suite


def suite2():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestSchoolsCase2)
    suite.addTests(tests)
    return suite


def suite3():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestSchoolsCase3)
    suite.addTests(tests)
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite1())

    # unittest.main()
