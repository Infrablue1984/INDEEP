import time
import unittest

import numpy as np
from numpy import testing

# import Activities
from data_factory.generator import people as p
from data_factory.generator.activities import Activities
from synchronizer import constants as cs


class TestActivitiesCase1(unittest.TestCase):
    def setUp(self):
        # self.region = [['01001'],['06533'],['05162'],['03101'],['07111'],['08335']]
        t4 = time.time()
        self.region = [5162, 1002]
        self.people1 = p.People(self.region, scale=100)
        t5 = time.time()
        arr1 = np.array([0.5, 1, 2, 2, 2, 3, 3, 2, 2, 2])
        self.a1 = Activities(self.people1, 100, mean_contacts=arr1)
        t6 = time.time_ns()
        # print("time for people: {}, time for workplaces {}".format(t5-t4, t6-t5))

    def test_init(self):
        pass


class TestActivitiesCase2(unittest.TestCase):
    def setUp(self):
        # self.region = [['01001'],['06533'],['05162'],['03101'],['07111'],['08335']]
        self.region = [5162]
        self.people1 = p.People(self.region, scale=100)
        # self.w1 = net.Workplaces(self.people1, 100)

    def test_init(self):
        # ensure that no id is given twice
        self.assertEqual(len(self.w1.w_ids), len(set(self.w1.w_ids)))
        self.assertEqual(len(self.w1.o_ids), len(set(self.w1.o_ids)))

        # ensure all worker are distributed to work after initialization
        agents_data = self.people1.get_agents_data()
        w_reshaped = np.concatenate([self.w1.w_ids[x] for x in self.w1.w_ids])
        self.assertEqual(
            len(self.w1.members_institutions),
            len(set(self.w1.members_institutions)),
        )
        self.assertEqual(len(self.w1.members_groups), len(set(self.w1.members_groups)))

        # ensure all worker are distributed to work after initialization
        agents_data = self.people1.get_agents_data()
        w_reshaped = np.concatenate(
            [self.w1.members_institutions[x] for x in self.w1.members_institutions]
        )
        agents_data, data_mbr = self.w1.split_sampled_data_from_agents_data(
            agents_data[cs.ID], w_reshaped, agents_data
        )
        self.assertEqual(
            np.sum(np.where(data_mbr[cs.OCCUPATION] == "w", 1, 0)),
            len(w_reshaped),
        )
        self.assertEqual(np.sum(np.where(agents_data[cs.OCCUPATION] == "w", 1, 0)), 0)
        self.w1.df_workplaces.to_excel("../inputs/workplaces.xls")

        self.assertEqual(self.w1.df_workplaces["size"].sum(), len(w_reshaped))
        self.assertEqual(self.w1.df_offices["size"].sum(), len(w_reshaped))

    def test_make_workplaces_of_one_type_and_add_people(self):
        # generate data
        agents_data = self.people1.get_agents_data()
        work_type = "big"
        num_people = 310
        # determine size before
        filt = self.w1.df_workplaces["work type"] == "big"
        size_1 = self.w1.df_workplaces.loc[filt, "size"].sum()
        # ensure workplaces grow for number of people
        _ = self.w1.make_workplaces_of_one_type_and_add_people(
            agents_data, self.region, work_type, num_people
        )
        filt = self.w1.df_workplaces["work type"] == "big"
        size_2 = self.w1.df_workplaces.loc[filt, "size"].sum()
        self.assertEqual(size_2 - size_1, num_people)

    def test_split_worker_into_offices(self):
        # ensure that people from offices are equal to the people from work
        dict_keys = list(self.w1.w_ids)
        work_id = dict_keys[123]
        p_at_work_id = self.w1.w_ids[work_id]
        filt = self.w1.df_offices["work id"] == work_id
        office_ids = self.w1.df_offices.loc[filt, "office id"]
        p_at_office_ids = np.concatenate([self.w1.o_ids[id] for id in office_ids])
        dict_keys = list(self.w1.members_institutions)
        work_id = dict_keys[123]
        p_at_work_id = self.w1.members_institutions[work_id]
        filt = self.w1.df_offices["work id"] == work_id
        office_ids = self.w1.df_offices.loc[filt, "office id"]
        p_at_office_ids = np.concatenate(
            [self.w1.members_groups[id] for id in office_ids]
        )

        testing.assert_array_equal(np.sort(p_at_work_id), np.sort(p_at_office_ids))
        self.assertEqual(len(p_at_work_id), len(p_at_office_ids))


def suite1():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestActivitiesCase1)
    suite.addTests(tests)
    return suite


def suite2():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestActivitiesCase2)
    suite.addTests(tests)
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite1())

    # unittest.main()
