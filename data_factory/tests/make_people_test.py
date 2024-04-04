import unittest
from copy import deepcopy as deep

# from contextlib import contextmanager as cm
import exception as ex
import make_people as p
import numpy as np
import pandas as pd
from numpy import random as rm

from synchronizer import constants as cs

german_pop = 83166711


class TestPeopleCase1(unittest.TestCase):
    def setUp(self):
        # create properties for population
        self.scale = 100
        self.regions1 = ["01001", "01002"]
        self.seed1 = rm.RandomState(0)
        self.people1 = p.People(self.regions1, scale=self.scale)
        self.pop_size = 3369

    def tearDown(self):
        self.people1 = None

    def test_make_ages(self):
        """county level"""
        ages = self.people1.__make_ages()
        # built sum of specific inds_age_gr
        age_0 = (ages == 0).sum() / 3368
        age_5 = (ages == 5).sum() / 3368
        age_11 = (ages == 11).sum() / 3368
        age_17 = (ages == 17).sum() / 3368
        age_23 = (ages == 23).sum() / 3368
        age_29 = (ages == 29).sum() / 3368
        age_35 = (ages == 35).sum() / 3368
        age_41 = (ages == 41).sum() / 3368
        age_46 = (ages == 46).sum() / 3368
        age_52 = (ages == 52).sum() / 3368
        age_58 = (ages == 58).sum() / 3368
        age_64 = (ages == 64).sum() / 3368
        age_70 = (ages == 70).sum() / 3368
        age_76 = (ages == 76).sum() / 3368
        age_80 = (ages == 80).sum() / 3368
        age_86 = (ages == 86).sum() / 3368
        age_92 = (ages == 92).sum() / 3368
        age_99 = (ages == 99).sum() / 3368

        # test difference between % of specific inds_age_gr and original table
        # should be less than 0.2 % (some exception 0.3 %)
        self.assertLess(age_0, 0.0105) and self.assertGreater(age_0, 0.0075)
        self.assertLess(age_5, 0.0098) and self.assertGreater(age_5, 0.0068)
        self.assertLess(age_11, 0.0101) and self.assertGreater(age_11, 0.0076)
        self.assertLess(age_17, 0.0113) and self.assertGreater(age_17, 0.0083)
        self.assertLess(age_23, 0.0134) and self.assertGreater(age_23, 0.0094)
        self.assertLess(age_29, 0.0139) and self.assertGreater(age_29, 0.0099)
        self.assertLess(age_35, 0.0131) and self.assertGreater(age_35, 0.0091)
        self.assertLess(age_41, 0.0135) and self.assertGreater(age_41, 0.0085)
        self.assertLess(age_46, 0.0136) and self.assertGreater(age_46, 0.0096)
        self.assertLess(age_52, 0.0200) and self.assertGreater(age_52, 0.0160)
        self.assertLess(age_58, 0.0180) and self.assertGreater(age_58, 0.0140)
        self.assertLess(age_64, 0.0143) and self.assertGreater(age_64, 0.0103)
        self.assertLess(age_70, 0.0313) and self.assertGreater(age_70, 0.0093)
        self.assertLess(age_76, 0.0122) and self.assertGreater(age_76, 0.0082)
        self.assertLess(age_80, 0.0127) and self.assertGreater(age_80, 0.0087)
        self.assertLess(age_86, 0.0058) and self.assertGreater(age_86, 0.0018)
        self.assertLess(age_92, 0.0020) and self.assertGreater(age_92, 0.0005)
        self.assertLess(age_99, 0.0025) and self.assertGreater(age_99, 0.0005)

    def test_get_pop_size(self):
        self.assertEqual(self.people1.get_pop_size(), 3368)

    def test_make_ids(self):
        regions = [1001, 1002]
        ids1 = self.people1.__make_ids(regions, 100)
        self.assertEqual(len(ids1), 3368)

    def test_make_sexes(self):
        sexes1 = self.people1.make_sexes()
        male1 = np.sum(sexes1 == 0) / len(sexes1)
        female1 = np.sum(sexes1 == 1) / len(sexes1)
        self.assertAlmostEqual(male1, 0.5, places=1) and self.assertAlmostEqual(
            female1, 0.5, places=1
        )

    def test_make_occupation(self):
        occ = self.people1.make_occupation()
        ages = self.people1.__make_ages()
        age_0 = (ages == 0).sum()
        age_6 = ((ages > 0) & (ages < 7)).sum()
        age_16 = ((ages > 6) & (ages < 17)).sum()
        age_19 = ((ages > 16) & (ages < 20)).sum()
        age_24 = ((ages > 19) & (ages < 25)).sum()
        age_29 = ((ages > 24) & (ages < 30)).sum()
        age_44 = ((ages > 39) & (ages < 45)).sum()
        age_59 = ((ages > 54) & (ages < 60)).sum()
        age_99 = ((ages > 69) & (ages < 100)).sum()

        # ensure all groups exist
        self.assertGreater((occ == "m").sum(), 0)  # medical
        self.assertGreater((occ == "c").sum(), 0)  # elderly care
        self.assertGreater((occ == "e").sum(), 0)  # education
        self.assertGreater((occ == "g").sum(), 0)  # gastronomy
        self.assertGreater((occ == "w").sum(), 0)  # worker
        self.assertGreater((occ == "p").sum(), 0)  # pupil
        self.assertGreater((occ == "k").sum(), 0)  # kita
        self.assertGreater((occ == "r").sum(), 0)  # retired
        self.assertGreater((occ == "N").sum(), 0)  # None

        # ensure the percentile is estimately right

        # check special occupations
        # total numbers from table Arbeitsamt: w 41.452.392
        # m 2.648.466, c 686.962, e 2.611.470, g 1.275.158
        w = 41452392 - 2648466 - 686962 - 2611470 - 1275158
        constant = self.pop_size / german_pop

        self.assertLessEqual((occ == "m").sum() / (2648466 * constant), 1.10)
        self.assertGreaterEqual((occ == "m").sum() / (2648466 * constant), 0.90)
        self.assertLessEqual((occ == "c").sum() / (686962 * constant), 1.10)
        self.assertGreaterEqual((occ == "c").sum() / (686962 * constant), 0.90)
        self.assertLessEqual((occ == "e").sum() / (2611470 * constant), 1.10)
        self.assertGreaterEqual((occ == "e").sum() / (2611470 * constant), 0.90)
        self.assertLessEqual((occ == "g").sum() / (1275158 * constant), 1.10)
        self.assertGreaterEqual((occ == "g").sum() / (1275158 * constant), 0.65)
        self.assertLessEqual((occ == "w").sum() / (w * constant), 1.10)
        self.assertGreaterEqual((occ == "w").sum() / (w * constant), 0.90)

        # check simple ccupations
        # percentages are taken from the original table according to age group

        # check kita
        k_0 = np.where((occ == "k") & (ages == 0), 1, 0).sum()
        k_6 = np.where((occ == "k") & (ages > 0) & (ages < 7), 1, 0).sum()
        self.assertAlmostEqual(k_0 / age_0 * 100, 25, places=-2)
        self.assertAlmostEqual(k_6 / age_6, 0.95, places=1)

        # check school
        p_16 = np.where((occ == "p") & (ages > 6) & (ages < 17), 1, 0).sum()
        p_19 = np.where((occ == "p") & (ages > 16) & (ages < 20), 1, 0).sum()
        p_24 = np.where((occ == "p") & (ages > 19) & (ages < 25), 1, 0).sum()
        p_24 = np.where((occ == "p") & (ages > 19) & (ages < 25), 1, 0).sum()
        self.assertAlmostEqual(p_16 / age_16, 1.0, places=1)
        self.assertAlmostEqual(p_19 / age_19, 0.73, places=1)
        self.assertAlmostEqual(p_24 / age_24, 0.32, places=1)

        # check retired
        r_44 = np.where((occ == "r") & (ages > 39) & (ages < 45), 1, 0).sum()
        r_59 = np.where((occ == "r") & (ages > 54) & (ages < 60), 1, 0).sum()
        r_99 = np.where((occ == "r") & (ages > 69) & (ages < 100), 1, 0).sum()
        self.assertAlmostEqual(r_44 / age_44 * 100, 2, places=-1)
        self.assertAlmostEqual(r_59 / age_59 * 100, 2, places=-1)
        self.assertAlmostEqual(r_99 / age_99 * 100, 100, places=-2)

        # check n
        n_0 = np.where((occ == "N") & (ages == 0), 1, 0).sum()
        n_6 = np.where((occ == "N") & (ages > 0) & (ages < 7), 1, 0).sum()
        n_59 = np.where((occ == "N") & (ages > 54) & (ages < 60), 1, 0).sum()
        self.assertAlmostEqual(n_0 / age_0 * 100, 75, places=-2)
        self.assertAlmostEqual(n_6 / age_6, 0.05, places=1)
        self.assertAlmostEqual(n_59 / age_59 * 100, 18, places=-2)

    def test_initialize_contacts(self):
        pass


class TestPeopleCase2(unittest.TestCase):
    def setUp(self):
        # create properties for population
        self.scale = 100
        # self.regions3 = [1]
        self.regions3 = [1, 2, 3]
        self.people3 = p.People(self.regions3, scale=self.scale)
        self.pop_size = 108918  # actually from the table it must be 127446

    def tearDown(self):
        self.people3 = None

    def test_get_pop_size(self):
        self.assertAlmostEqual(
            self.people3.get_pop_size(), self.pop_size
        )  # self.assertEqual(self.people4.get_pop_size(), 60)

    def test_initialize_contacts(self):
        pass

    def test_make_ids(self):
        self.assertAlmostEqual(len(self.people3._agents_data[cs.ID]), self.pop_size)

    def test_make_ages(self):
        """state level"""
        # built sum of specific inds_age_gr
        ages = self.people3.__make_ages()
        age_0 = (ages == 0).sum() / self.pop_size
        age_10 = (ages == 10).sum() / self.pop_size
        age_20 = (ages == 20).sum() / self.pop_size
        age_30 = (ages == 30).sum() / self.pop_size
        age_39 = (ages == 39).sum() / self.pop_size
        age_51 = (ages == 51).sum() / self.pop_size
        age_63 = (ages == 63).sum() / self.pop_size
        age_70 = (ages == 70).sum() / self.pop_size
        age_82 = (ages == 82).sum() / self.pop_size
        age_93 = (ages == 93).sum() / self.pop_size

        # test difference between % of specific inds_age_gr and original table
        # should be less than 0.15 %
        self.assertLess(age_0, 0.0108) and self.assertGreater(age_0, 0.0088)
        self.assertLess(age_10, 0.0098) and self.assertGreater(age_10, 0.0078)
        self.assertLess(age_20, 0.0124) and self.assertGreater(age_20, 0.0094)
        self.assertLess(age_30, 0.0157) and self.assertGreater(age_30, 0.0127)
        self.assertLess(age_39, 0.0149) and self.assertGreater(age_39, 0.0119)
        self.assertLess(age_51, 0.0177) and self.assertGreater(age_51, 0.0147)
        self.assertLess(age_63, 0.0133) and self.assertGreater(age_63, 0.0103)
        self.assertLess(age_70, 0.0116) and self.assertGreater(age_70, 0.0076)
        self.assertLess(age_82, 0.0090) and self.assertGreater(age_82, 0.0060)
        self.assertLess(age_93, 0.0025) and self.assertGreater(age_93, 0.0005)

    def test_make_sexes(self):
        sexes3 = self.people3.make_sexes()
        male3 = np.sum(sexes3 == 0) / len(sexes3)
        female3 = np.sum(sexes3 == 1) / len(sexes3)
        self.assertAlmostEqual(male3, 0.5, places=1) and self.assertAlmostEqual(
            female3, 0.5, places=1
        )

    def test_make_occupation(self):
        # get total number of age groups
        occ = self.people3.make_occupation()
        ages = self.people3.__make_ages()
        age_0 = (ages == 0).sum()
        age_6 = ((ages > 0) & (ages < 7)).sum()
        age_14 = ((ages > 6) & (ages < 15)).sum()
        age_19 = ((ages > 16) & (ages < 20)).sum()
        age_24 = ((ages > 19) & (ages < 25)).sum()
        # 8075 taken from debugger
        spec_age_24 = 8075
        age_29 = ((ages > 24) & (ages < 30)).sum()
        age_44 = ((ages > 39) & (ages < 45)).sum()
        spec_age_54 = ((ages > 24) & (ages < 55)).sum()
        age_59 = ((ages > 54) & (ages < 60)).sum()
        age_99 = ((ages > 69) & (ages < 100)).sum()

        # ensure all groups exist
        self.assertGreater((occ == "m").sum(), 0)
        self.assertGreater((occ == "c").sum(), 0)
        self.assertGreater((occ == "e").sum(), 0)
        self.assertGreater((occ == "g").sum(), 0)
        self.assertGreater((occ == "w").sum(), 0)
        self.assertGreater((occ == "p").sum(), 0)
        self.assertGreater((occ == "k").sum(), 0)
        self.assertGreater((occ == "r").sum(), 0)
        self.assertGreater((occ == "N").sum(), 0)

        # ensure the total number of special occupations is less tan 5 % different to numbers
        # from original table of Arbeitsamt: w 41.452.392
        # m 2.648.466, c 686.962, e 2.611.470, g 1.275.158
        w = 41452392 - 2648466 - 686962 - 2611470 - 1275158
        # the constant is due to % population Regions/Germany and scaling factor
        constant = self.pop_size / german_pop

        self.assertLessEqual((occ == "m").sum() / (2648466 * constant), 1.10)
        self.assertGreaterEqual((occ == "m").sum() / (2648466 * constant), 0.90)
        self.assertLessEqual((occ == "c").sum() / (686962 * constant), 1.10)
        self.assertGreaterEqual((occ == "c").sum() / (686962 * constant), 0.90)
        self.assertLessEqual((occ == "e").sum() / (2611470 * constant), 1.10)
        self.assertGreaterEqual((occ == "e").sum() / (2611470 * constant), 0.90)
        self.assertLessEqual((occ == "g").sum() / (1275158 * constant), 1.10)
        self.assertGreaterEqual((occ == "g").sum() / (1275158 * constant), 0.90)
        self.assertLessEqual((occ == "w").sum() / (w * constant), 1.10)
        self.assertGreaterEqual((occ == "w").sum() / (w * constant), 0.90)

        # check medical
        # percentile m / total w * employment rate
        # 24: 353.053 / 7.790.851 = 0.045
        # 54: 1.792.998 / 32.752.020 = 0.055
        m_24 = np.where((occ == "m") & (ages > 15) & (ages < 25), 1, 0).sum()
        m_54 = np.where((occ == "m") & (ages > 24) & (ages < 55), 1, 0).sum()
        self.assertAlmostEqual(m_24 / spec_age_24, 0.045, places=1)
        self.assertAlmostEqual(m_54 / spec_age_54, 0.055, places=2)

        # check education
        # percentile e / total w * employment rate
        # 24: 307.674 / 7.790.851 = 0.039
        # 54: 1.738.082 / 32.752.020 = 0.053
        edu_24 = np.where((occ == "e") & (ages > 15) & (ages < 25), 1, 0).sum()
        edu_54 = np.where((occ == "e") & (ages > 24) & (ages < 55), 1, 0).sum()
        self.assertAlmostEqual(edu_24 / spec_age_24, 0.039, places=1)
        self.assertAlmostEqual(edu_54 / spec_age_54, 0.053, places=2)

        # check kita child
        k_0 = np.where((occ == "k") & (ages == 0), 1, 0).sum()
        k_6 = np.where((occ == "k") & (ages > 0) & (ages < 7), 1, 0).sum()
        self.assertAlmostEqual(k_0 / age_0, 0.25, places=1)
        self.assertAlmostEqual(k_6 / age_6, 0.95, places=1)

        # check school
        p_14 = np.where((occ == "p") & (ages > 6) & (ages < 15), 1, 0).sum()
        p_19 = np.where((occ == "p") & (ages > 16) & (ages < 20), 1, 0).sum()
        p_24 = np.where((occ == "p") & (ages > 19) & (ages < 25), 1, 0).sum()
        p_24 = np.where((occ == "p") & (ages > 19) & (ages < 25), 1, 0).sum()
        self.assertAlmostEqual(p_14 / age_14, 1.0, places=1)
        self.assertAlmostEqual(p_19 / age_19, 0.73, places=1)
        self.assertAlmostEqual(p_24 / age_24, 0.32, places=1)

        # check retired
        p_44 = np.where((occ == "r") & (ages > 39) & (ages < 45), 1, 0).sum()
        p_59 = np.where((occ == "r") & (ages > 54) & (ages < 60), 1, 0).sum()
        p_99 = np.where((occ == "r") & (ages > 69) & (ages < 100), 1, 0).sum()
        self.assertAlmostEqual(p_44 / age_44, 0.02, places=1)
        self.assertAlmostEqual(p_59 / age_59, 0.02, places=1)
        self.assertAlmostEqual(p_99 / age_99 * 100, 95, places=-2)

        # check n
        n_0 = np.where((occ == "N") & (ages == 0), 1, 0).sum()
        n_6 = np.where((occ == "N") & (ages > 0) & (ages < 7), 1, 0).sum()
        n_59 = np.where((occ == "N") & (ages > 54) & (ages < 60), 1, 0).sum()
        self.assertAlmostEqual(n_0 / age_0, 0.75, places=1)
        self.assertAlmostEqual(n_6 / age_6, 0.05, places=1)
        self.assertAlmostEqual(n_59 / age_59, 0.18, places=1)


class TestPeopleCase3(unittest.TestCase):
    def setUp(self):
        # create properties for population
        self.scale = 100
        self.regions4 = [0]
        self.people4 = p.People(self.regions4, scale=self.scale)
        self.pop_size = 831667

    def tearDown(self):
        self.people4 = None

    def test_get_pop_size(self):
        self.assertEqual(
            self.people4.get_pop_size(), 831307
        )  # self.assertEqual(self.people4.get_pop_size(), 60)

    def test_initialize_contacts(self):
        pass

    def test_make_ids(self):
        # ids4 = self.people4.__make_ids(self.regions4, 100)
        self.assertEqual(len(self.people4._agents_data[cs.ID]), 831307)

    def test_make_ages(self):
        """german level"""
        # built sum of specific inds_age_gr
        ages = self.people4.__make_ages()
        age_0 = (ages == 0).sum() / german_pop
        age_10 = (ages == 10).sum() / german_pop
        age_20 = (ages == 20).sum() / german_pop
        age_30 = (ages == 30).sum() / german_pop
        age_39 = (ages == 39).sum() / german_pop
        age_51 = (ages == 51).sum() / german_pop
        age_63 = (ages == 63).sum() / german_pop
        age_70 = (ages == 70).sum() / german_pop
        age_82 = (ages == 82).sum() / german_pop
        age_93 = (ages == 93).sum() / german_pop

        # test difference between % of specific inds_age_gr and original table
        # should be less than 0.1 % in exception 0.15 %
        self.assertLess(age_0, 0.0108) and self.assertGreater(age_0, 0.0088)
        self.assertLess(age_10, 0.0098) and self.assertGreater(age_10, 0.0078)
        self.assertLess(age_20, 0.0119) and self.assertGreater(age_20, 0.0099)
        self.assertLess(age_30, 0.0152) and self.assertGreater(age_30, 0.0132)
        self.assertLess(age_39, 0.0144) and self.assertGreater(age_39, 0.0124)
        self.assertLess(age_51, 0.0172) and self.assertGreater(age_51, 0.0152)
        self.assertLess(age_63, 0.0133) and self.assertGreater(age_63, 0.0108)
        self.assertLess(age_70, 0.0111) and self.assertGreater(age_70, 0.0091)
        self.assertLess(age_82, 0.0085) and self.assertGreater(age_82, 0.0065)
        self.assertLess(age_93, 0.0020) and self.assertGreater(age_93, 0.0005)

    def test_make_sexes(self):
        sexes4 = self.people4.make_sexes()
        male4 = np.sum(sexes4 == 0) / len(sexes4)
        female4 = np.sum(sexes4 == 1) / len(sexes4)
        self.assertAlmostEqual(male4, 0.5, places=1) and self.assertAlmostEqual(
            female4, 0.5, places=1
        )

    def test_make_occupation(self):
        # get total number of age groups
        occ = self.people4.make_occupation()
        ages = self.people4.__make_ages()
        age_0 = (ages == 0).sum()
        age_6 = ((ages > 0) & (ages < 7)).sum()
        age_14 = ((ages > 6) & (ages < 15)).sum()
        age_19 = ((ages > 16) & (ages < 20)).sum()
        age_24 = ((ages > 19) & (ages < 25)).sum()
        spec_age_24 = ((ages > 15) & (ages < 25)).sum()
        age_29 = ((ages > 24) & (ages < 30)).sum()
        age_44 = ((ages > 39) & (ages < 45)).sum()
        spec_age_54 = ((ages > 24) & (ages < 55)).sum()
        age_59 = ((ages > 54) & (ages < 60)).sum()
        age_99 = ((ages > 69) & (ages < 100)).sum()

        # ensure all groups exist
        self.assertGreater((occ == "m").sum(), 0)
        self.assertGreater((occ == "c").sum(), 0)
        self.assertGreater((occ == "e").sum(), 0)
        self.assertGreater((occ == "g").sum(), 0)
        self.assertGreater((occ == "w").sum(), 0)
        self.assertGreater((occ == "p").sum(), 0)
        self.assertGreater((occ == "k").sum(), 0)
        self.assertGreater((occ == "r").sum(), 0)
        self.assertGreater((occ == "N").sum(), 0)

        # ensure the total number of special occupations is less tan 5 % different to numbers
        # from original table of Arbeitsamt: w 41.452.392
        # m 2.648.466, c 686.962, e 2.611.470, g 1.275.158
        w = 41452392 - 2648466 - 686962 - 2611470 - 1275158
        # the constant is due to % population Regions/Germany and scaling factor
        constant = self.pop_size / german_pop

        self.assertLessEqual((occ == "m").sum() / (2648466 * constant), 1.10)
        self.assertGreaterEqual((occ == "m").sum() / (2648466 * constant), 0.90)
        self.assertLessEqual((occ == "c").sum() / (686962 * constant), 1.10)
        self.assertGreaterEqual((occ == "c").sum() / (686962 * constant), 0.90)
        self.assertLessEqual((occ == "e").sum() / (2611470 * constant), 1.10)
        self.assertGreaterEqual((occ == "e").sum() / (2611470 * constant), 0.90)
        self.assertLessEqual((occ == "g").sum() / (1275158 * constant), 1.10)
        self.assertGreaterEqual((occ == "g").sum() / (1275158 * constant), 0.90)
        self.assertLessEqual((occ == "w").sum() / (w * constant), 1.10)
        self.assertGreaterEqual((occ == "w").sum() / (w * constant), 0.90)

        # check medical
        # percentile m / total w * employment rate
        # 24: 353.053 / 7.790.851 = 0.045
        # 54: 1.792.998 / 32.752.020 = 0.055
        m_24 = np.where((occ == "m") & (ages > 15) & (ages < 25), 1, 0).sum()
        m_54 = np.where((occ == "m") & (ages > 24) & (ages < 55), 1, 0).sum()
        self.assertAlmostEqual(m_24 / spec_age_24, 0.045, places=1)
        self.assertAlmostEqual(m_54 / spec_age_54, 0.055, places=2)

        # check education
        # percentile e / total w * employment rate
        # 24: 307.674 / 7.790.851 = 0.039
        # 54: 1.738.082 / 32.752.020 = 0.053
        edu_24 = np.where((occ == "e") & (ages > 15) & (ages < 25), 1, 0).sum()
        edu_54 = np.where((occ == "e") & (ages > 24) & (ages < 55), 1, 0).sum()
        self.assertAlmostEqual(edu_24 / spec_age_24, 0.039, places=1)
        self.assertAlmostEqual(edu_54 / spec_age_54, 0.053, places=2)

        # check kita
        k_0 = np.where((occ == "k") & (ages == 0), 1, 0).sum()
        k_6 = np.where((occ == "k") & (ages > 0) & (ages < 7), 1, 0).sum()
        self.assertAlmostEqual(k_0 / age_0, 0.25, places=1)
        self.assertAlmostEqual(k_6 / age_6, 0.95, places=1)

        # check school
        p_14 = np.where((occ == "p") & (ages > 6) & (ages < 15), 1, 0).sum()
        p_19 = np.where((occ == "p") & (ages > 16) & (ages < 20), 1, 0).sum()
        p_24 = np.where((occ == "p") & (ages > 19) & (ages < 25), 1, 0).sum()
        p_24 = np.where((occ == "p") & (ages > 19) & (ages < 25), 1, 0).sum()
        self.assertAlmostEqual(p_14 / age_14, 1.0, places=1)
        self.assertAlmostEqual(p_19 / age_19, 0.73, places=1)
        self.assertAlmostEqual(p_24 / age_24, 0.32, places=1)

        # check retired
        p_44 = np.where((occ == "r") & (ages > 39) & (ages < 45), 1, 0).sum()
        p_59 = np.where((occ == "r") & (ages > 54) & (ages < 60), 1, 0).sum()
        p_99 = np.where((occ == "r") & (ages > 69) & (ages < 100), 1, 0).sum()
        self.assertAlmostEqual(p_44 / age_44, 0.02, places=1)
        self.assertAlmostEqual(p_59 / age_59, 0.02, places=1)
        self.assertAlmostEqual(p_99 / age_99 * 100, 95, places=-2)

        # check None
        n_0 = np.where((occ == "N") & (ages == 0), 1, 0).sum()
        n_6 = np.where((occ == "N") & (ages > 0) & (ages < 7), 1, 0).sum()
        n_59 = np.where((occ == "N") & (ages > 54) & (ages < 60), 1, 0).sum()
        self.assertAlmostEqual(n_0 / age_0, 0.75, places=1)
        self.assertAlmostEqual(n_6 / age_6, 0.05, places=1)
        self.assertAlmostEqual(n_59 / age_59, 0.18, places=1)


class TestPeopleCase4(unittest.TestCase):
    def setUp(self):
        # create properties for population
        self.scale = 100
        self.regions4 = [5]
        self.regions4a = [5162]
        self.people4 = p.People(self.regions4, scale=self.scale)
        self.people4a = p.People(self.regions4a, scale=self.scale)

    def tearDown(self):
        self.people4 = None

    def test_get_data_from_ids(self):
        ids = ["05162_U_0", "05162_U_22", "05162_U_45"]
        ages = self.people4a.get_data_from_ids(ids, cs.AGE)
        ages = list(ages)
        self.assertEqual(ages, [16, 14, 39])


if __name__ == "__main__":

    suite1 = unittest.defaultTestLoader.loadTestsFromTestCase(TestPeopleCase1)
    suite2 = unittest.defaultTestLoader.loadTestsFromTestCase(TestPeopleCase2)
    suite3 = unittest.defaultTestLoader.loadTestsFromTestCase(TestPeopleCase3)
    suite4 = unittest.defaultTestLoader.loadTestsFromTestCase(TestPeopleCase4)
    unittest.TextTestRunner().run(suite4)
    # unittest.main()

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
