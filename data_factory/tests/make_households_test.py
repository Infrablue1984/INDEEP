import unittest

import make_people as p
import networks as net
import numpy as np
import pandas as pd
from make_households import Households
from numpy import testing

from synchronizer import constants as cs


class TestHouseholdsCase1(unittest.TestCase):
    def setUp(self):
        # self.region = [['01001'],['06533'],['05162'],['03101'],['07111'],['08335']]
        self.scale = 100
        self.region = [5162]
        self.people = p.People(self.region, scale=self.scale)
        self.hh = Households(self.people, self.scale)
        self.age_gr_multi = pd.read_excel(
            "../inputs/households_ages.xls", "Mehrpersonen"
        )

    def test_create_and_save_households_for_all_german_regions(self):
        pass

    def test_init(self):
        # ensure that none of the ids is taken twice or never
        ids_1 = set(self.people.get_data_for("ids"))
        ids_2 = set(
            self.hh.contacts.reshape(
                self.hh.contacts.shape[0] * self.hh.contacts.shape[1]
            )
        )
        ids_2.remove("")
        self.assertEqual(ids_1, ids_2)

    def test_initialize_1_size_hh(self):
        one_size_hh = self.hh.contacts[self.hh.sizes == 1]
        ids = one_size_hh[:, 0]
        ages = self.people.get_data_from_ids(ids, cs.AGE)
        sexes = self.people.get_data_from_ids(ids, cs.SEX)
        fem = np.where(sexes == 1, 1, 0).sum()
        male = np.where(sexes == 0, 1, 0).sum()
        f_0 = np.where((sexes == 1) & ((ages == 18) | (ages == 19)), 1, 0).sum()
        m_0 = np.where((sexes == 0) & ((ages == 18) | (ages == 19)), 1, 0).sum()
        f_1 = np.where((sexes == 1) & (ages >= 20) & (ages <= 24), 1, 0).sum()
        m_1 = np.where((sexes == 0) & (ages >= 20) & (ages <= 24), 1, 0).sum()
        f_2 = np.where((sexes == 1) & (ages >= 25) & (ages <= 29), 1, 0).sum()
        m_2 = np.where((sexes == 0) & (ages >= 25) & (ages <= 29), 1, 0).sum()
        f_3 = np.where((sexes == 1) & (ages >= 30) & (ages <= 34), 1, 0).sum()
        m_3 = np.where((sexes == 0) & (ages >= 30) & (ages <= 34), 1, 0).sum()
        f_4 = np.where((sexes == 1) & ((ages >= 35) & (ages <= 39)), 1, 0).sum()
        m_4 = np.where((sexes == 0) & ((ages >= 35) & (ages <= 39)), 1, 0).sum()
        f_5 = np.where((sexes == 1) & (ages >= 40) & (ages <= 44), 1, 0).sum()
        m_5 = np.where((sexes == 0) & (ages >= 40) & (ages <= 44), 1, 0).sum()
        f_6 = np.where((sexes == 1) & (ages >= 45) & (ages <= 49), 1, 0).sum()
        m_6 = np.where((sexes == 0) & (ages >= 45) & (ages <= 49), 1, 0).sum()
        f_7 = np.where((sexes == 1) & (ages >= 50) & (ages <= 54), 1, 0).sum()
        m_7 = np.where((sexes == 0) & (ages >= 50) & (ages <= 54), 1, 0).sum()
        f_8 = np.where((sexes == 1) & (ages >= 55) & (ages <= 59), 1, 0).sum()
        m_8 = np.where((sexes == 0) & (ages >= 55) & (ages <= 59), 1, 0).sum()
        f_9 = np.where((sexes == 1) & ((ages >= 60) & (ages <= 64)), 1, 0).sum()
        m_9 = np.where((sexes == 0) & ((ages >= 60) & (ages <= 64)), 1, 0).sum()
        f_10 = np.where((sexes == 1) & (ages >= 65) & (ages <= 69), 1, 0).sum()
        m_10 = np.where((sexes == 0) & (ages >= 65) & (ages <= 69), 1, 0).sum()
        f_11 = np.where((sexes == 1) & (ages >= 70) & (ages <= 74), 1, 0).sum()
        m_11 = np.where((sexes == 0) & (ages >= 70) & (ages <= 74), 1, 0).sum()
        f_12 = np.where((sexes == 1) & (ages >= 75) & (ages <= 99), 1, 0).sum()
        m_12 = np.where((sexes == 0) & (ages >= 75) & (ages <= 99), 1, 0).sum()
        fac1 = 2.0
        fac2 = 0.5
        self.assertLessEqual(f_0 / fem, 0.0069 * fac1)
        self.assertLessEqual(m_0 / male, 0.0094 * fac1)
        self.assertLessEqual(f_1 / fem, 0.063 * fac1)
        self.assertLessEqual(f_2 / fem, 0.0696 * fac1)
        self.assertLessEqual(m_2 / male * 100, 11.2 * fac1)
        self.assertLessEqual(f_3 / fem, 0.0523 * fac1)
        self.assertLessEqual(m_3 / male, 0.105 * fac1)
        self.assertLessEqual(f_4 / fem, 0.0374 * fac1)
        self.assertLessEqual(m_4 / male, 0.0845 * fac1)
        self.assertLessEqual(f_5 / fem, 0.0306 * fac1)
        self.assertLessEqual(m_5 / male * 100, 7.3 * fac1)
        self.assertLessEqual(f_6 / fem, 0.0399 * fac1)
        self.assertLessEqual(m_6 / male, 0.0751 * fac1)
        self.assertLessEqual(f_8 / fem, 0.0818 * fac1)
        self.assertLessEqual(m_8 / male, 0.0940 * fac1)
        self.assertLessEqual(f_10 / fem, 0.0830 * fac1)
        self.assertLessEqual(m_10 / male, 0.0539 * fac1)
        self.assertLessEqual(f_12 / fem, 0.3146 * fac1)
        self.assertLessEqual(m_12 / male, 0.1042 * fac1)

        self.assertGreaterEqual(f_0 / fem, 0.0069 * fac2)
        self.assertGreaterEqual(m_1 / male, 0.079 * fac2)
        self.assertGreaterEqual(f_2 / fem, 0.0696 * fac2)
        self.assertGreaterEqual(m_2 / male * 100, 11.2 * fac2)
        self.assertGreaterEqual(f_3 / fem, 0.0523 * fac2)
        self.assertGreaterEqual(m_3 / male, 0.105 * fac2)
        self.assertGreaterEqual(f_4 / fem, 0.0374 * fac2)
        self.assertGreaterEqual(m_4 / male, 0.0845 * fac2)
        self.assertGreaterEqual(f_5 / fem, 0.0306 * fac2)
        self.assertGreaterEqual(m_5 / male * 100, 7.3 * fac2)
        self.assertGreaterEqual(f_6 / fem, 0.0399 * fac2)
        self.assertGreaterEqual(m_6 / male, 0.0751 * fac2)
        self.assertGreaterEqual(f_8 / fem, 0.0818 * fac2)
        self.assertGreaterEqual(m_8 / male, 0.0940 * fac2)
        self.assertGreaterEqual(f_10 / fem, 0.0830 * fac2)
        self.assertGreaterEqual(m_10 / male, 0.0539 * fac2)
        self.assertGreaterEqual(f_12 / fem, 0.3146 * fac2)
        self.assertGreaterEqual(m_12 / male, 0.1042 * fac2)

    def test_initialize_household_contacts(self):
        # contacts = self.hh2.contacts
        z_code = self.people.get_data_for(cs.Z_CODE)
        landsc = self.people.get_data_for("landscapes")
        ages = self.people.get_data_for(cs.AGE)
        sexes = self.people.get_data_for(cs.SEX)

        sum_hh_mbr_05162 = (
            ((self.hh.sizes == 1) & (self.hh.z_codes == 5162)).sum()
            + 2 * ((self.hh.sizes == 2) & (self.hh.z_codes == 5162)).sum()
            + 3 * ((self.hh.sizes == 3) & (self.hh.z_codes == 5162)).sum()
            + 4 * ((self.hh.sizes == 4) & (self.hh.z_codes == 5162)).sum()
            + 5 * ((self.hh.sizes == 5) & (self.hh.z_codes == 5162)).sum()
        )
        print("Requested household members_institutions: " + str(sum_hh_mbr_05162))
        sum_people_05162 = np.where(z_code == 5162, 1, 0).sum()
        print("Existing household members_institutions: " + str(sum_people_05162))

        sum_hh_mbr_05162_r = (
            1
            * (
                (self.hh.sizes == 1)
                & (self.hh.z_codes == 5162)
                & (self.hh.landsc == 60)
            ).sum()
            + 2
            * (
                (self.hh.sizes == 2)
                & (self.hh.z_codes == 5162)
                & (self.hh.landsc == 60)
            ).sum()
            + 3
            * (
                (self.hh.sizes == 3)
                & (self.hh.z_codes == 5162)
                & (self.hh.landsc == 60)
            ).sum()
            + 4
            * (
                (self.hh.sizes == 4)
                & (self.hh.z_codes == 5162)
                & (self.hh.landsc == 60)
            ).sum()
            + 5
            * (
                (self.hh.sizes == 5)
                & (self.hh.z_codes == 5162)
                & (self.hh.landsc == 60)
            ).sum()
        )
        print("Requested household members_institutions: " + str(sum_hh_mbr_05162_r))
        sum_people_05162 = ((z_code == 5162) & (landsc == 60)).sum()
        print(
            "Existing household members_institutions rural area: "
            + str(sum_people_05162)
        )

    def test_get_data_from_ids(self):
        ids = np.zeros((10, 2), dtype="U20")
        ids[:, 0] = np.array(
            [
                "05162_U_36",
                "05162_U_72",
                "05162_U_201",
                "05162_U_2",
                "05162_U_16",
                "05162_R_36",
                "05162_R_25",
                "05162_R_72",
                "05162_R_93",
                "05162_R_85",
            ]
        )
        ids[:, 1] = np.array(
            [
                "05162_U_32",
                "05162_U_62",
                "05162_U_1",
                "05162_U_5",
                "05162_U_56",
                "05162_R_13",
                "05162_R_12",
                "05162_R_11",
                "05162_R_15",
                "05162_R_80",
            ]
        )
        self.people.get_data_from_ids(ids, cs.AGE)


class TestHouseholdsCase2(unittest.TestCase):
    def setUp(self):
        # self.region = [['01001'],['06533'],['05162'],['03101'],['07111'],['08335']]
        self.scale = 100
        self.region = ["11000"]
        self.people = p.People(self.region, scale=self.scale)
        print("begin hh")
        self.hh = Households(self.people, self.scale)
        print("end households")
        self.age_gr_multi = pd.read_excel(
            "../inputs/households_ages.xls", "Mehrpersonen"
        )

    def test_save_households_to_file(self):
        pass

    def test_init(self):
        print("test_init")
        # ensure that none of the ids is taken twice or never
        ids_1 = set(self.people.get_data_for("ids"))
        ids_2 = set(
            self.hh.contacts.reshape(
                self.hh.contacts.shape[0] * self.hh.contacts.shape[1]
            )
        )
        ids_2.remove("")
        self.assertEqual(ids_1, ids_2)

    def test_initialize_1_size_hh(self):
        print("test_initialize_1_size_hh")
        one_size_hh = self.hh.contacts[self.hh.sizes == 1]
        ids = one_size_hh[:, 0]
        ages = self.people.get_data_from_ids(ids, cs.AGE)
        sexes = self.people.get_data_from_ids(ids, cs.SEX)
        fem = np.where(sexes == 1, 1, 0).sum()
        male = np.where(sexes == 0, 1, 0).sum()
        f_0 = np.where((sexes == 1) & ((ages == 18) | (ages == 19)), 1, 0).sum()
        m_0 = np.where((sexes == 0) & ((ages == 18) | (ages == 19)), 1, 0).sum()
        f_1 = np.where((sexes == 1) & (ages >= 20) & (ages <= 24), 1, 0).sum()
        m_1 = np.where((sexes == 0) & (ages >= 20) & (ages <= 24), 1, 0).sum()
        f_2 = np.where((sexes == 1) & (ages >= 25) & (ages <= 29), 1, 0).sum()
        m_2 = np.where((sexes == 0) & (ages >= 25) & (ages <= 29), 1, 0).sum()
        f_3 = np.where((sexes == 1) & (ages >= 30) & (ages <= 34), 1, 0).sum()
        m_3 = np.where((sexes == 0) & (ages >= 30) & (ages <= 34), 1, 0).sum()
        f_4 = np.where((sexes == 1) & ((ages >= 35) & (ages <= 39)), 1, 0).sum()
        m_4 = np.where((sexes == 0) & ((ages >= 35) & (ages <= 39)), 1, 0).sum()
        f_5 = np.where((sexes == 1) & (ages >= 40) & (ages <= 44), 1, 0).sum()
        m_5 = np.where((sexes == 0) & (ages >= 40) & (ages <= 44), 1, 0).sum()
        f_6 = np.where((sexes == 1) & (ages >= 45) & (ages <= 49), 1, 0).sum()
        m_6 = np.where((sexes == 0) & (ages >= 45) & (ages <= 49), 1, 0).sum()
        f_7 = np.where((sexes == 1) & (ages >= 50) & (ages <= 54), 1, 0).sum()
        m_7 = np.where((sexes == 0) & (ages >= 50) & (ages <= 54), 1, 0).sum()
        f_8 = np.where((sexes == 1) & (ages >= 55) & (ages <= 59), 1, 0).sum()
        m_8 = np.where((sexes == 0) & (ages >= 55) & (ages <= 59), 1, 0).sum()
        f_9 = np.where((sexes == 1) & ((ages >= 60) & (ages <= 64)), 1, 0).sum()
        m_9 = np.where((sexes == 0) & ((ages >= 60) & (ages <= 64)), 1, 0).sum()
        f_10 = np.where((sexes == 1) & (ages >= 65) & (ages <= 69), 1, 0).sum()
        m_10 = np.where((sexes == 0) & (ages >= 65) & (ages <= 69), 1, 0).sum()
        f_11 = np.where((sexes == 1) & (ages >= 70) & (ages <= 74), 1, 0).sum()
        m_11 = np.where((sexes == 0) & (ages >= 70) & (ages <= 74), 1, 0).sum()
        f_12 = np.where((sexes == 1) & (ages >= 75) & (ages <= 99), 1, 0).sum()
        m_12 = np.where((sexes == 0) & (ages >= 75) & (ages <= 99), 1, 0).sum()

        self.assertAlmostEqual(f_0 / fem, 0.0069, places=2)
        self.assertAlmostEqual(m_0 / male, 0.0094, places=2)
        self.assertAlmostEqual(f_1 / fem, 0.063, places=2)
        self.assertAlmostEqual(m_1 / male, 0.079, places=2)
        self.assertAlmostEqual(f_2 / fem, 0.0696, places=2)
        self.assertAlmostEqual(m_2 / male * 100, 11.2, places=-2)
        self.assertAlmostEqual(f_3 / fem, 0.0523, places=2)
        self.assertAlmostEqual(m_3 / male, 0.105, places=2)
        self.assertAlmostEqual(f_4 / fem, 0.0374, places=2)
        self.assertAlmostEqual(m_4 / male, 0.0845, places=2)
        self.assertAlmostEqual(f_5 / fem, 0.0306, places=2)
        self.assertAlmostEqual(m_5 / male * 100, 7.3, places=-2)
        self.assertAlmostEqual(f_6 / fem, 0.0399, places=2)
        self.assertAlmostEqual(m_6 / male, 0.0751, places=2)
        self.assertAlmostEqual(f_8 / fem, 0.0818, places=2)
        self.assertAlmostEqual(m_8 / male, 0.0940, places=2)
        self.assertAlmostEqual(f_10 / fem, 0.0830, places=2)
        self.assertAlmostEqual(m_10 / male, 0.0539, places=2)
        self.assertAlmostEqual(f_12 / fem, 0.3146, places=2)
        self.assertAlmostEqual(m_12 / male, 0.1042, places=2)

    def test_initialize_2_size_hh(self):
        print("test_initialize_2_size_hh")
        two_size_hh = self.hh.contacts[self.hh.sizes == 2]
        hh_with_ids = two_size_hh[:, 0:2]
        hh_with_ages = np.zeros(hh_with_ids.shape, dtype=int)
        hh_with_sexes = np.zeros(hh_with_ids.shape, dtype=int)
        for hh in range(len(hh_with_ids)):
            hh_with_ages[hh] = self.people.get_data_from_ids(hh_with_ids[hh], cs.AGE)
            hh_with_sexes[hh] = self.people.get_data_from_ids(hh_with_ids[hh], cs.SEX)
        ages_min = hh_with_ages.min(1)
        ages_max = hh_with_ages.max(1)
        age_diff = abs(hh_with_ages[:, 0] - hh_with_ages[:, 1])
        age_diff_single_p = age_diff[ages_min < 18]
        age_diff_couple = age_diff[ages_min >= 25]
        test = np.where(age_diff_single_p > 40, 1, 0).sum()
        # self.assertLess(np.where(age_diff > 40, 1, 0).sum(), len(hh_with_ages) * 0.05)
        self.assertLess(
            np.where(age_diff_single_p > 40, 1, 0).sum(),
            len(age_diff_single_p) * 0.075,
        )
        self.assertLess(
            np.where(age_diff_single_p > 45, 1, 0).sum(),
            len(age_diff_single_p) * 0.025,
        )
        self.assertLess(
            np.where(age_diff_single_p < 20, 1, 0).sum(),
            len(age_diff_single_p) * 0.075,
        )
        self.assertLess(
            np.where(age_diff_couple > 15, 1, 0).sum(),
            len(age_diff_couple) * 0.125,
        )
        self.assertLess(
            np.where(age_diff_couple > 20, 1, 0).sum(),
            len(age_diff_couple) * 0.075,
        )
        # because 84% couples
        self.assertLess(
            (hh_with_sexes[:, 0] == hh_with_sexes[:, 1]).sum(),
            len(hh_with_sexes) * 0.20,
        )

    def test_initialize_3_size_hh(self):
        print("test_initialize_3_size_hh")
        three_size_hh = self.hh.contacts[self.hh.sizes == 3]
        hh_with_ids = three_size_hh[:, 0:3]
        hh_with_ages = np.zeros(hh_with_ids.shape, dtype=int)
        hh_with_sexes = np.zeros(hh_with_ids.shape, dtype=int)
        for hh in range(len(hh_with_ids)):
            hh_with_ages[hh] = self.people.get_data_from_ids(hh_with_ids[hh], cs.AGE)
            hh_with_sexes[hh] = self.people.get_data_from_ids(hh_with_ids[hh], cs.SEX)
        ages_min = hh_with_ages.min(1)
        ages_max = hh_with_ages.max(1)
        age_diff = abs(ages_min - ages_max)

        self.assertLess(np.where(age_diff > 40, 1, 0).sum(), len(hh_with_ages) * 0.05)
        self.assertLess(np.where(age_diff < 20, 1, 0).sum(), len(hh_with_ages) * 0.05)

    def test_initialize_4_size_hh(self):
        print("test_initialize_4_size_hh")
        four_size_hh = self.hh.contacts[self.hh.sizes == 4]
        hh_with_ids = four_size_hh[:, 0:4]
        hh_with_ages = np.zeros(hh_with_ids.shape, dtype=int)
        hh_with_sexes = np.zeros(hh_with_ids.shape, dtype=int)
        for hh in range(len(hh_with_ids)):
            hh_with_ages[hh] = self.people.get_data_from_ids(hh_with_ids[hh], cs.AGE)
            hh_with_sexes[hh] = self.people.get_data_from_ids(hh_with_ids[hh], cs.SEX)
        ages_min = hh_with_ages.min(1)
        ages_max = hh_with_ages.max(1)
        age_diff = abs(ages_min - ages_max)

        self.assertLess(np.where(age_diff > 40, 1, 0).sum(), len(hh_with_ages) * 0.075)
        self.assertLess(np.where(age_diff < 20, 1, 0).sum(), len(hh_with_ages) * 0.075)

    def test_initialize_5_size_hh(self):
        print("test_initialize_5_size_hh")
        five_size_hh = self.hh.contacts[self.hh.sizes == 5]
        hh_with_ids = five_size_hh[:, 0:5]
        hh_with_ages = np.zeros(hh_with_ids.shape, dtype=int)
        hh_with_sexes = np.zeros(hh_with_ids.shape, dtype=int)
        for hh in range(len(hh_with_ids)):
            hh_with_ages[hh] = self.people.get_data_from_ids(hh_with_ids[hh], cs.AGE)
            hh_with_sexes[hh] = self.people.get_data_from_ids(hh_with_ids[hh], cs.SEX)
        ages_min = hh_with_ages.min(1)
        ages_max = hh_with_ages.max(1)
        age_diff = abs(ages_min - ages_max)
        test = age_diff.sum() / len(hh_with_ages)

        # self.assertLess(np.where(age_diff > 40, 1, 0).sum(), len(hh_with_ages) * 0.50)
        self.assertLess(np.where(age_diff > 40, 1, 0).sum(), len(hh_with_ages) * 0.35)
        self.assertLess(np.where(age_diff < 20, 1, 0).sum(), len(hh_with_ages) * 0.35)


def suite1():
    suite = unittest.TestSuite()
    suite.addTest(TestHouseholdsCase1("test_set_possible_age_groups_for_other_members"))
    return suite


def suite2():
    suite = unittest.TestSuite()
    suite.addTest(TestHouseholdsCase1("test_get_data_from_ids"))
    return suite


def suite3():
    suite = unittest.TestSuite()
    suite.addTest(TestHouseholdsCase2("test_initialize_1_size_hh"))
    return suite


def suite4():
    suite = unittest.TestSuite()
    suite.addTest(TestHouseholdsCase1("test_initialize_2_size_hh"))
    suite.addTest(TestHouseholdsCase1("test_initialize_3_size_hh"))
    suite.addTest(TestHouseholdsCase1("test_initialize_4_size_hh"))
    suite.addTest(TestHouseholdsCase1("test_initialize_5_size_hh"))
    suite.addTest(TestHouseholdsCase1("test_init"))
    return suite


def suite6():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestHouseholdsCase1)
    suite.addTests(tests)
    return suite


def suite7():
    suite = unittest.TestSuite()
    print("begin loader")
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestHouseholdsCase2)
    print("end loader")
    suite.addTests(tests)
    return suite


if __name__ == "__main__":
    print("begin runner")
    runner = unittest.TextTestRunner()
    runner.run(suite6())
    print("end runner")

# unittest.main()
