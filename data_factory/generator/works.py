#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Docstring."""

__author__ = "Inga Franzen"
__created__ = "2021"
__date_modified__ = "2023/05/31"
__version__ = "1.0"

import pathlib

import numpy as np
import pandas as pd
from numpy import random as rm

from data_factory.generator.networks_d import Institution
from synchronizer import constants as cs
from synchronizer.synchronizer import PathManager as PM


class Workplaces(Institution):
    def __init__(self, people, scale=None):
        """Initialize workplaces for people.

        :param people: people class object
        :param scale: 10 or 100
        """

        super(Workplaces, self).__init__(people, scale)
        self.abs_path = pathlib.Path(__file__).parent.parent.absolute()
        all_agents_data = self.extract_worker_from_people_class(people)

        # create data structures to save workplaces and people
        self.df_networks = pd.DataFrame(
            columns=["id", "type", "z_code", "size", "mean contacts"]
        )

        self.df_sub_nets = pd.DataFrame(
            columns=[
                "id",
                "sub id",
                "type",
                "size",
                "mean contacts",
                "year",
                "weekday",
            ]
        )
        self.members_institutions = {}
        self.members_groups = {}
        self.id = np.int64(2 * 10**11) / self.scale

        # create data structure to grab data from, file 48121-0001.xlsx
        work_data = pd.read_excel(PM.get_path_work_table())
        self.work_data = work_data.to_dict(orient="list")
        # iterate regions
        for region in self.regions:
            agents_data = self.filter_agents_data_by_subdata(
                cs.Z_CODE, region, all_agents_data
            )
            self.make_workplaces_for_one_region(region, agents_data)

        self.add_mean_contacts()
        # self.determine_mean_number_of_contacts()
        self.determine_mean_contacts_region()
        self.save_current_mean_contacts()

        test = 1

    def extract_worker_from_people_class(self, people):
        all_agents_data = people.get_agents_data()
        occupations = people.get_data_for(cs.OCCUPATION)
        idx = np.argwhere(np.where(occupations == "w", 1, 0))
        all_agents_data = self.extract_data_of_specified_indices(idx, all_agents_data)
        return all_agents_data

    def make_workplaces_for_one_region(self, region, agents_data):
        """Create workplaces of different sizes.

        Args:
            region(int): z_code
            agents_data(dict(ndarray)): Hold ids, ages, sexes, z_codes, landscapes for people.

        Returns:
            agents_data(dict(ndarray)): distributed people are deleted
        """
        # determine num of people for each type category
        num_mini = int(len(agents_data[cs.ID]) * self.work_data["%_people"][0])
        num_small = int(len(agents_data[cs.ID]) * self.work_data["%_people"][1])
        num_medium = int(len(agents_data[cs.ID]) * self.work_data["%_people"][2])
        num_big = len(agents_data[cs.ID]) - num_mini - num_small - num_medium
        # id count for region
        id_count = self.id + region * np.int64(10**6) / self.scale
        # create workplaces and sample people
        id_count = self.make_workplaces_of_one_type_and_add_people(
            agents_data, region, "mini", num_mini, id_count
        )

        id_count = self.make_workplaces_of_one_type_and_add_people(
            agents_data, region, "small", num_small, id_count
        )

        id_count = self.make_workplaces_of_one_type_and_add_people(
            agents_data, region, "medium", num_medium, id_count
        )

        _ = self.make_workplaces_of_one_type_and_add_people(
            agents_data, region, "big", num_big, id_count
        )

        assert len(agents_data[cs.ID]) == 0

        return

    def make_workplaces_of_one_type_and_add_people(
        self, agents_data, region, work_type, num_people, id_count
    ):
        """Create workplaces of given work_size and add people.

        Args:
            agents_data(dict(ndarry)): Hold ids, ages, sexes, z_codes, landscapes for people.
            region(int) : z_code
            work_type(int) : mini, small, medium, big
            num_people(int) : total number of people to be distributed

        Returns:
        """
        w_pl_sizes = self.determine_workplace_sizes(num_people, work_type)
        # initalize values for sampling people
        worker_count = 0
        idx = 0
        # iterate workplaces and sample people from pool
        while worker_count < num_people:
            w_pl_size = w_pl_sizes[idx]
            # round work id to the next 10th
            if worker_count + w_pl_size > num_people:
                w_pl_size = num_people - worker_count
            worker_count += w_pl_size
            idx += 1
            # do not register 1-person-work
            if (w_pl_size == 1) or (num_people - worker_count == 1):
                _ = self.sample_worker_and_remove(agents_data, w_pl_size)
                continue
            # register > 1-person-work
            w_id = id_count = self.register_work(id_count, region, work_type, w_pl_size)
            id_count = self.add_worker(agents_data, id_count, w_id, w_pl_size)
            id_count = self.create_other_activities_at_institution(
                w_id, id_count, w_pl_size
            )
        return id_count

    def add_worker(self, agents_data, id_count, w_id, w_pl_size):
        members = self.sample_worker_and_remove(agents_data, w_pl_size)
        self.add_worker_to_work(members, w_id)
        id_count = self.split_worker_into_office(members, id_count, w_id)
        return id_count

    def register_work(self, id_count, region, w_type, size=np.nan):
        # only not the case for first school
        if not int(np.ceil(id_count / 1000) * 1000) == id_count:
            id_count += 1
        w_id = int(np.ceil(id_count / 10) * 10)
        self.members_institutions[w_id] = np.arange(0)
        super(Workplaces, self).append_institution_data(w_id, w_type, size)
        return w_id

    def determine_workplace_sizes(self, num_people, work_type):
        # extract basic work data for work type
        # extract basic work data for type
        idx = self.work_data["type"].index(work_type)
        w_mean = self.work_data["mean_size"][idx]
        w_min = self.work_data["min_size"][idx]
        w_max = self.work_data["max_size"][idx]
        alpha = self.work_data["alpha"][idx]
        beta = self.work_data["beta"][idx]
        # initialize values
        range_w = w_max - w_min + 1
        w_pl_sizes = np.arange(0, dtype=int)
        diff = num_people
        # sample workplace sizes until more workplaces than people
        while diff > 0:
            num_w_pl = max(1, int(diff / w_mean))
            # TODO: why choosen beta here?
            base = rm.beta(a=alpha, b=beta, size=num_w_pl)
            temp_sizes = (base * range_w + w_min).astype(int)  # standard beta is [0,1]
            w_pl_sizes = np.hstack((w_pl_sizes, temp_sizes))
            diff = num_people - np.sum(w_pl_sizes)
        return w_pl_sizes

    def sample_worker_and_remove(self, agents_data, work_pl_size):
        # sample people from pool
        members = rm.choice(agents_data[cs.ID], work_pl_size, replace=False)
        # remove agents from pool
        agents_data, _ = super(Workplaces, self).split_sampled_data_from_agents_data(
            agents_data[cs.ID], members, agents_data
        )
        return members

    def add_worker_to_work(self, members, w_id):
        self.members_institutions[w_id] = np.hstack(
            (self.members_institutions[w_id], members)
        )

    def split_worker_into_office(self, members, id_count, w_id):
        # init statistics about office sizes
        # (https://de.statista.com/infografik/8780/bueroarbeitsplaetze-in-deutschland/)
        o_size = [1, 2, 3, 4, 5, 10, 20, 30]
        o_dist = [0.37, 0.275, 0.065, 0.07, 0.07, 0.05, 0.05, 0.05]
        count_1_o = 0
        count_all = len(members)

        # iterate worker until no one left
        while count_all > 0:
            my_size = rm.choice(o_size, 1, p=o_dist)
            if (my_size == 1) or (count_all == 1):
                count_1_o += 1
                count_all -= 1
                continue  # do not register 1-person offices
            if my_size <= count_all:
                o_members = rm.choice(members, my_size, replace=False)
                count_all -= my_size
            else:
                o_members = rm.choice(members, count_all, replace=False)
                count_all -= count_all
            indices = np.nonzero(np.isin(members, o_members))[0]
            members = np.delete(members, indices)
            id_count += 1  # create sub id
            self.add_worker_to_office(id_count, o_members)
            self.register_office(id_count, w_id, o_members)
            # delete distributed worker from choice

        assert len(members) == count_1_o
        return id_count

    def add_worker_to_office(self, id_count, members):
        self.members_groups[id_count] = members

    def register_office(self, id_count, w_id, members):
        self.append_group_data(
            w_id,
            id_count,
            type="office",
            size=len(members),
        )

    def add_mean_contacts(self):
        mean_contacts = {"office": 2, "mensa": 3, "meeting": 2, "unspecific": 1}
        super(Workplaces, self).add_mean_contacts(**mean_contacts)

    def create_other_activities_at_institution(self, w_id, id_count, w_pl_size):
        id_count = super(Workplaces, self).create_unspecific_contacts(
            w_id, id_count, w_pl_size
        )
        id_count = super(Workplaces, self).create_mensa(w_id, id_count, w_pl_size)
        id_count = super(Workplaces, self).create_other_groups(
            w_id, id_count, w_pl_size, "meeting"
        )
        return id_count
