#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Docstring."""

__author__ = "Inga Franzen"
__created__ = "2022"
__date_modified__ = "2023/05/31"
__version__ = "1.0"

import numpy as np
from numpy import random as rm

from calculation.people import People
from data_factory.generator.networks_d import Institution
from synchronizer import constants as cs
from synchronizer.synchronizer import Synchronizer as Sync


class Activities(Institution):
    def __init__(self, people: People, scale: int):
        """

        Args:
            people: People object of the simulation.
            scale: Number of people represented by a single agent.
        """

        # TODO(Felix): I would not use this many self variables. Only keep those that
        #  you need again later. For example, put the min/max_size as arguments of the
        # init method and set default values.
        super().__init__(people, scale)

        # determine group sizes
        self.min_size = 4
        self.max_size = 50
        self.mean_size = (self.max_size + self.min_size) / 2
        # determine mean contacts per person in group
        # TODO(Felix): General thought: Seeing the prefix 'mean_' I expect also
        #   something like a standard derivation to use together in a distribution. If
        #   there is no distribution happening I would just call it 'contacts_per_agent'
        #   and 'times/activities_per_week' for example.
        # TODO(Felix): Maybe specify (in documentation) that these are contacts per day.
        self.mean_c1 = 2
        self.mean_c2 = 5
        self.mean_c = (self.mean_c1 + self.mean_c2) / 2
        # TODO(Felix): What do you mean with 'times' per week?
        # determine times per week
        self.times = np.array([1, 2, 3, 4])
        self.min_times = self.times[0]
        self.max_times = self.times[-1]
        self.mean_times = np.mean(self.times)
        all_agents_data = people.get_agents_data()
        self.precision = np.zeros(10)
        self.count = np.zeros(10)
        # TODO(Felix): It is not clear what target and df_current are? Maybe rename to
        #   something like 'mean_contacts_per_age_group' and
        #   'df_mean_contacts_per_network_category'/'df_mean_contacts'.
        self.target = self.load_target_mean_contacts()
        self.df_current = self.open_current_mean_contacts()

        # iterate regions
        for region in self.regions:
            self.activity_count = 0  # count for activity id
            agents_data = self.filter_agents_data_by_subdata(
                cs.Z_CODE,
                region,
                all_agents_data,
            )
            self.make_activities_for_one_region(region, agents_data)
            # can be recalled via method get_mean_contacts_per_age_gr
        self.determine_mean_contacts_region()
        self.save_current_mean_contacts()

    def make_activities_for_one_region(self, region: int, agents_data: dict):
        """Create activities of different types.

        Args:
            region: Z_code for the selected region.
            agents_data: Hold ids, ages, sexes, z_codes, landscapes for people.
        """
        # TODO(Felix): Maybe again set local variables as default values of arguments.
        # percentage groups of mixed age
        per_mixed = 0.2
        # TODO(Felix): Why is random_factor = 1 and not random?
        random_factor = 1
        mean_current = self.get_current_mean_contacts(region)
        # TODO(Felix): Maybe rename mean_target to smth. like 'mean_contacts_activities'
        mean_target = self.target - mean_current - random_factor
        # TODO(Felix): It is a little bit confusing that 'mean_target' is returned.
        mean_target = self.make_mixed_activities(
            agents_data,
            per_mixed,
            region,
            mean_target,
        )
        self.make_same_age_activities(agents_data, region, mean_target)

    def make_mixed_activities(self, agents_data, p_mixed, region, mean_target):
        # TODO(Felix): Why shape=(0,8)?
        activity_list = np.zeros((0, 8))
        ids = agents_data[cs.ID]
        # first sample activities of mixed age groups
        mean_c = np.mean(mean_target)
        c_total = mean_c * len(ids) * p_mixed
        c_p_group = self.mean_c * self.mean_size * self.mean_times / 7
        num_groups = round(c_total / c_p_group)
        # 999 means mixed ages
        activity_list = self._create_base_data(len(ids), num_groups, 999, activity_list)
        self.sample_member_and_register(ids, region, activity_list)
        # ["id", "sub id", "type", "size", "mean contacts", "year", "weekday"]
        # TODO(Felix): What do "sub id" and "type" mean?
        # TODO(Felix): This method needs documentation.
        mean_mixed = self.determine_mean_contacts_sub(agents_data, activity_list)
        assert np.all(mean_mixed >= 0)
        mean_target -= mean_mixed
        return mean_target

    def make_same_age_activities(self, agents_data, region, mean_target):
        # then sample activities of same age groups
        for age_gr in self.mean_contacts_p_age["Age_gr"]:
            activity_list = np.zeros((0, 8))
            age_target = mean_target[age_gr]
            if age_target < 0:
                continue
            my_people = self.extract_people_from_age_group(age_gr, agents_data)
            # estimate mean_c number of events
            total_target = age_target * len(my_people)
            mean_contacts_p_group = self.mean_c * self.mean_size * self.mean_times / 7
            num_groups = round(total_target / mean_contacts_p_group)
            activity_list = self._create_base_data(
                len(my_people), num_groups, age_gr, activity_list
            )
            total_diff = self.determine_diff_current_target(
                age_gr, activity_list, total_target
            )
            # TODO(Felix): Why '> 0.1' ?
            while abs(total_diff / len(my_people)) > 0.1:
                if total_diff > 0:
                    activity_list = self._add_last_activity_for_region_and_age_gr(
                        len(my_people), age_gr, total_diff, activity_list
                    )
                else:
                    activity_list = self.remove_activity(
                        age_gr, total_diff, activity_list
                    )
                total_diff = self.determine_diff_current_target(
                    age_gr, activity_list, total_target
                )
            self.sample_member_and_register(my_people, region, activity_list)

    def _add_last_activity_for_region_and_age_gr(
        self, num_ids, age_gr, diff, activity_list
    ):
        min_size = self.min_size
        mean_c = self.mean_c1
        times = round(self.mean_times)
        if diff < self.min_size * mean_c * times / 7:
            while (times > self.min_times) & (diff < min_size * mean_c * times / 7):
                times -= 1
            while (mean_c > 1) & (diff < min_size * mean_c * times / 7):
                mean_c -= 1
            while (min_size > 2) & (diff < min_size * mean_c * times / 7):
                min_size -= 1
        elif diff > self.max_size * mean_c * times / 7:
            max_size = self.max_size
            while (mean_c < self.mean_c2) & (diff > max_size * mean_c * times / 7):
                mean_c += 1
            while (times < self.max_times) & (diff > max_size * mean_c * times / 7):
                times += 1
        if min_size < self.min_size:
            my_size = min_size
        else:
            my_size = round(diff / mean_c / (times / 7))
            my_size = min(my_size, num_ids)
        activity_list = self._create_base_data_one_group(
            my_size, mean_c, age_gr, times, activity_list
        )
        return activity_list

    def sample_member_and_register(self, my_ids, region, activity_list):
        # extract socials
        # TODO(Felix): Why are sociable and not sociable agents defined?
        other, sociable = self.extract_sociable_and_other(my_ids)
        # ["id", "sub id", "type", "size", "mean contacts", "year", "weekday"]
        type_id = Sync.SUB_NET_TRANSLATE[cs.ACTIVITY]
        for i in range(len(activity_list)):
            group_id = self.create_group_id(region)
            activity_list[i, 0] = group_id
            activity_list[i, 1] = group_id
            activity_list[i, 2] = type_id
            size = activity_list[i, 3]
            mean_contacts = activity_list[i, 4]
            activity_list[i, 4] = min(size - 1, mean_contacts)
            # TODO(Felix): Why are sociables selected twice? Once for selecting the
            #  members of groups and once for inside the groups. But in the second one
            #  the members are only ordered by sociables first and others later. Why?
            members = self.select_member_from_p_sociable(sociable, other, size)
            assert len(members) == size
            self.members_groups[group_id] = members
        self.sub_nets = np.vstack((self.sub_nets, activity_list))

    def _create_base_data(self, num_ids, num_groups, age_gr, activity_list):
        # define array of group sizes (assume uniform distribution)
        max_size = min(self.max_size, num_ids)
        my_choice = range(self.min_size, max_size + 1)
        my_sizes = rm.choice(my_choice, num_groups)
        # TODO(Felix): This gives only 2 or 5. Should it in between 2 and 5?
        my_mean = rm.choice([self.mean_c1, self.mean_c2], num_groups)
        num_wd = rm.choice(self.times, num_groups)
        # iterate activities
        for idx, size in enumerate(my_sizes):
            # register activity
            activity_list = self._create_base_data_one_group(
                size, my_mean[idx], age_gr, num_wd[idx], activity_list
            )
        return activity_list

    def _create_base_data_one_group(self, size, my_mean, age_gr, num_wd, activity_list):
        # choose weekdays
        weekday = rm.choice([0, 1, 2, 3, 4, 5, 6], num_wd, replace=False)
        # register activity
        data = np.array(
            [np.nan, np.nan, np.nan, size, my_mean, age_gr, weekday, np.nan],
            dtype=object,
        )
        return np.vstack((activity_list, data))

    def create_group_id(self, region):
        group_id = (
            np.int64(6 * 10**10) + region * np.int64(10**5) + self.activity_count
        )
        self.activity_count += 1
        return group_id

    def remove_activity(self, age_gr, diff, activity_list):
        # ["id", "sub id", "type", "size", "mean contacts", "year", "weekday"]
        cond = activity_list[:, 5] == age_gr
        my_sub = activity_list[cond]
        if len(my_sub) == 0:
            print("No activity left to remove")
            return
        my_times = np.array([len(x) for x in my_sub[:, 6]])
        activity_yields = my_sub[:, 3] * my_sub[:, 4] * my_times / 7
        while diff < 0:
            accuracy = abs(activity_yields - diff)
            ind = np.argmin(accuracy)
            value = accuracy[ind]
            my_sub = np.delete(my_sub, [ind], axis=0)
            diff += value
        return my_sub

    def determine_diff_current_target(self, a_gr, activity_list, total_target):
        # ["id", "sub id", "type", "size", "mean contacts", "year", "weekday"]
        cond = activity_list[:, 5] == a_gr
        my_sub = activity_list[cond]
        my_times = np.array([len(x) for x in my_sub[:, 6]])
        total_current = (my_sub[:, 3] * my_sub[:, 4] * my_times / 7).sum()
        return total_target - total_current

    def get_precision_and_count(self):
        return self.precision, self.count

    def get_current_mean_contacts(self, region):
        """Get."""
        name = type(self).__name__.lower()
        data = self.df_current.loc[region]
        data.drop(name, axis=1, inplace=True)
        return data.sum(axis=1).to_numpy()
