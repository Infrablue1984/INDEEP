import re
import time as time
import unittest

import create_household_tables
import network_manager as net
import numpy as np
import pandas as pd
import people as p
from numpy import char as ch


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.df_households = pd.read_excel(
            "../inputs/households.xls",
            "Summe Haushalte",
            dtype=str,
            na_values=["nan"],
        )
        self.df_households.set_index("BL", inplace=True)
        self.df_households["Summe HH"] = self.df_households["Summe HH"].astype(int)
        self.scale = 100
        # self.region =['00']
        # self.people = p.People(self.region, scale=self.scale)

    def test_init(self):
        pass

    def test_create_household_ids(self):
        create_household_tables.make_household_data(self.scale)

        # new test

        # old test
        path = "../inputs/household_ids_" + str(self.scale) + ".csv"
        df_household_ids = pd.read_csv(path, dtype=str)
        real_total_sum = 0
        for i in self.df_households.index:
            real_sum = self.df_households.loc[i, "Summe HH"]
            filt = df_household_ids["household_ids"].str.startswith(i)
            sum = filt.sum() * self.scale
            difference = sum / real_sum
            # ensure the difference between modelled number and real number of county hh < 1 %
            self.assertGreater(difference, 0.99)
            self.assertLess(difference, 1.01)
            real_total_sum += real_sum
        total_sum = len(df_household_ids) * self.scale
        difference = total_sum / real_total_sum
        # ensure the difference between modelled number and real number of total hh < 1 %
        self.assertAlmostEqual(difference, 1, places=2)


if __name__ == "__main__":
    unittest.main()
