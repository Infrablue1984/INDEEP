#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Docstring."""

__author__ = "Inga Franzen"
__created__ = "2021"
__date_modified__ = "2023/05/31"
__version__ = "1.0"

import math
import os
from copy import deepcopy as deep

import numpy as np
import pandas as pd
from numpy import random as rm

from synchronizer import constants as cs
from synchronizer.synchronizer import FileLoader, FileSaver
from synchronizer.synchronizer import PathManager as PM
from synchronizer.synchronizer import Synchronizer as Sync


class Institution:
    def __init__(self, people, scale):
        self.scale = scale
        # store basic values for all methods
        self.people = people
        self.regions = people.get_regions()
        self.lost = 0
        self.mean_contacts_p_age = self.create_dict_with_mean_contacts()
        self.members_institutions = {}  # key: network id
        self.members_groups = {}  # key: subnetwork id
        self.p_social = 0.6
        # create data structures to save schools and people
        # TODO: remove z_code and mean_contacts from self.df_networks in all places
        self.networks = np.zeros((0, 3))
        self.df_networks = pd.DataFrame()

        self.sub_nets = np.zeros((0, 8))
        self.df_sub_nets = pd.DataFrame()

    @staticmethod
    def reset_data(digits, scale):
        net_types = []
        file_names = []
        for digit in digits:
            net_types.append(Sync.DIGIT_TRANSLATE[digit])
        for net_type in net_types:
            file_names.append(PM.get_path_members_file(net_type, scale))
            file_names.append(PM.get_path_parent_table(net_type, scale))
            file_names.append(PM.get_path_sub_table(net_type, scale))
        for file_name in file_names:
            if os.path.exists(file_name):
                os.remove(file_name)

    def create_dict_with_mean_contacts(self):
        age_gr_dict = Institution.get_age_gr_dict()
        # age_gr_dict["Mean contacts"] = np.zeros(10)
        return deep(age_gr_dict)

    def create_other_groups(
        self, inst_id, act_id, size, type, mean_size=20, mean_p_agent=1
    ):
        """Create activities beside office and classrooms.

        The sizes and mean values are arbitrarily chosen.

        Args:
            inst_id (int): id of parent institution
            act_id (int): id of location
            size (int): institutional size

        Returns:
        """
        # extract school members
        members = self.members_institutions[inst_id]
        other, sociable = self.extract_sociable_and_other(members)
        assert len(sociable) + len(other) == len(members)
        # add other activities
        if size > 2:
            mean_p_inst = math.ceil(size * mean_p_agent / mean_size)
            for idx in range(mean_p_inst):
                my_size = int(rm.normal(mean_size, 2.5, mean_p_inst)[0])
                my_size = min(my_size, size)
                times_p_week = rm.choice([1, 2], 1)
                weekday = rm.choice([0, 1, 2, 3, 4], times_p_week, replace=False)
                members = self.select_member_from_p_sociable(sociable, other, my_size)
                act_id = self.register_activity(
                    inst_id,
                    members,
                    my_size,
                    act_id,
                    type=type,
                    weekday=weekday,
                )
        return act_id

    @staticmethod
    def get_age_gr_dict():
        my_dict = {
            "Age_gr": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "Min": np.array([0, 5, 10, 15, 20, 30, 40, 50, 60, 70]),
            "Max": np.array([4, 9, 14, 19, 29, 39, 49, 59, 69, 99]),
            "Mean contacts": np.zeros(10),
        }
        return my_dict

    def extract_sociable_and_other(self, members):
        ids = self.people.get_data_for(cs.ID)
        sociable = self.people.get_data_for(cs.SOCIABLE)
        cond1 = np.isin(ids, members)
        cond2 = sociable == 1
        cond3 = sociable == 0
        sociable = ids[cond1 & cond2]
        other = ids[cond1 & cond3]
        return other, sociable

    def select_member_from_p_sociable(self, sociable, other, size):
        # choose socials
        num_s = max(min(round(size * self.p_social), len(sociable)), size - len(other))
        my_sociable = rm.choice(sociable, num_s, replace=False)
        # choose other
        num_n = size - num_s
        my_other = rm.choice(other, num_n, replace=False)
        # append member
        # TODO(Felix): Why are they appended?
        members = np.append(my_sociable, my_other)
        return members

    def create_mensa(self, inst_id, act_id, size, p_size=0.25):
        # extract school members
        members = self.members_institutions[inst_id]
        # add mensa
        mensa_size = int(size * p_size)
        if mensa_size > 1:
            my_ids = rm.choice(members, mensa_size, replace=False)
            act_id = self.register_activity(
                inst_id, my_ids, mensa_size, act_id, type="mensa"
            )
        return act_id

    def create_unspecific_contacts(self, inst_id, act_id, size):
        # extract school members
        members = self.members_institutions[inst_id]
        # add random
        act_id = self.register_activity(
            inst_id, members, size, act_id, type="unspecific"
        )
        return act_id

    def register_activity(
        self,
        inst_id,
        members,
        size,
        act_id,
        type,
        weekday=np.nan,
    ):
        act_id += 1
        self.members_groups[act_id] = members
        self.append_group_data(inst_id, act_id, type=type, size=size, weekday=weekday)
        return act_id

    def append_institution_data(self, sc_id, sc_type, size):
        data = np.array([sc_id, sc_type, size], dtype=object)
        self.networks = np.vstack((self.networks, data))

    def append_group_data(
        self,
        sc_id,
        cl_id,
        type,
        size,
        mean_contacts=np.nan,
        year=np.nan,
        weekday=np.nan,
        landsc=np.nan,
    ):
        """:param cl_id:

        :param sc_id:
        :param year:
        :param size:
        :param type:
        :param mean_contacts:
        """
        type_id = Sync.SUB_NET_TRANSLATE[type]
        data = np.array(
            [sc_id, cl_id, type_id, size, mean_contacts, year, weekday, landsc],
            dtype=object,
        )
        self.sub_nets = np.vstack((self.sub_nets, data))

    def add_mean_contacts(self, **kwargs):
        nets = self.sub_nets
        for name, code in Sync.SUB_NET_TRANSLATE.items():
            type_information = Sync.LOCATION_TYPE_INFORMATION[name]
            nets[nets[:, 2] == code, 4] = type_information[cs.MEAN_CONTACTS]
        # consider possible < mean contacts
        nets[:, 4] = [min(nets[x, 4], nets[x, 3] - 1) for x in range(len(nets))]

    def save_networks_to_file(self):
        # save dict to npy file
        net_type = type(self).__name__.lower()
        self.save_members_to_file(self.members_groups, net_type)
        self.df_networks = self.numpy_to_dataframe(self.networks)
        self.save_dataframe_to_file(self.df_networks, net_type)
        # transfer numpy array to dataframe
        self.df_sub_nets = self.numpy_to_dataframe(self.sub_nets, sub=True)
        self.save_dataframe_to_file(self.df_sub_nets, net_type, sub=True)

    def save_members_to_file(self, members, net_type):
        abs_file_name = PM.get_path_members_file(net_type, self.scale)
        if os.path.exists(abs_file_name):
            old_groups = FileLoader.load_dict_from_npy(abs_file_name)
            members = {**members, **old_groups}
        np.save(abs_file_name, members)

    def save_dataframe_to_file(self, df, net_type, sub=False):
        # open or create file
        if sub:
            abs_file_name = PM.get_path_sub_table(net_type, self.scale)
        else:
            abs_file_name = PM.get_path_parent_table(net_type, self.scale)
        if os.path.exists(abs_file_name):
            df_existing = pd.read_csv(abs_file_name)
            df_new = pd.concat([df_existing, df], ignore_index=True)
        else:
            df_new = df
        if "Unnamed: 0" in df_new.columns:
            df_new.drop(columns="Unnamed: 0", inplace=True)
        df_new.to_csv(abs_file_name, index=True)

    def numpy_to_dataframe(self, networks, sub=False):
        if sub:
            return pd.DataFrame(networks, columns=Sync.SUB_NET_COLUMNS.keys())
        else:
            return pd.DataFrame(networks, columns=Sync.NETWORK_COLUMNS.keys())

    def drop_rows_by_value(self, df, column, value_arr):
        filt = df[column] == -1
        for value in value_arr:
            filt = (filt) | (df[column] == value)
        drop_index = df.loc[
            filt,
        ].index
        df.drop(index=drop_index, columns=column, inplace=True)
        return df

    def sample_member_from_pool(
        self,
        age_min,
        region,
        landsc,
        sum_hh,
        agents_data,
        sex=None,
        age_max=99,
        member=None,
    ):
        """Sample members_institutions directly from pool according to min. and
        max. age.

        Before sampling check for number of agents with requested attributes in the
        pool. If the number of requested
        members_institutions is higher than the existing agents in the pool,
        reduce the requested number and remember the lost. After
        sampling use another method to specifically sample the number of lost
        members_institutions. This method is more tolerant with
        the requested attributes such as min. or max. age.


        Args:
             age_min (int): minimum age (e.g. 18)
             age_max (int): maximum age (e.g. 65)
             region (int): unique value for the region (called z_codes)
             landsc (int): value for landscape with 30 (city) and 60 (rural area)
             sum_hh (int): number of members_institutions to sample
             agents_data (dict(ndarray)): data of agents in the pool to choose from
             (ids, ages, sexes, z_codes, landsc)
             sex (int): 0 for male and 1 for female

        Returns:
            agents_data (dict(ndarray)): data of agents in the pool after deletion of
            sampled agents (ids, ages, sexes, z_codes, landsc)
            data_mbr (dict(ndarray)): data of sampled members_institutions
            (ids, ages, sexes, z_codes, landsc)
        """
        # extract data
        ids, ages, sexes, z_codes, landscapes, sociable = self.extract_agents_data(
            agents_data
        )  # unpack
        lost, sum_hh = self.ensure_request_exist_match(
            age_min, region, landsc, sum_hh, agents_data, sex, age_max
        )

        # sample household member from pool
        if sex is None:
            ids_mbr = rm.choice(
                ids[
                    (ages >= age_min)
                    & (ages <= age_max)
                    & (z_codes == region)
                    & (landscapes == landsc)
                ],
                sum_hh,
                replace=False,
            )
        else:
            ids_mbr = rm.choice(
                ids[
                    (ages >= age_min)
                    & (ages <= age_max)
                    & (sexes == sex)
                    & (z_codes == region)
                    & (landscapes == landsc)
                ],
                sum_hh,
                replace=False,
            )

        # delete sampled data from agents data and save them in a separate dict
        agents_data, data_mbr = self.split_sampled_data_from_agents_data(
            ids, ids_mbr, agents_data
        )

        # sample lost
        if lost > 0:
            (data_mbr_lost, agents_data,) = self.sample_lost_due_to_gap_request_exist(
                lost, age_min, region, landsc, agents_data, sex, age_max, member
            )
            data_mbr = self.concatenate_data_dictionary(data_mbr, data_mbr_lost)

        return data_mbr, agents_data

    def determine_age_groups_and_sample_member(
        self,
        age_grs,
        age_dist,
        region,
        landsc,
        sum,
        df_age_gr,
        agents_data,
        sex=None,
        member=None,
    ):
        """Sample members_institutions from pool according to referenced a_gr,
        range of possible mean_contacts_p_age for other members_institutions
        and age_distr.

        First sample age groups of members_institutions according to age distribution.
        Then sum the number of people within each age
        group and sample real agents from pool.


        Args:
             age_grs (ndarray(int)): range of age groups possible to choose from
             age_dist (ndarray(int)): age distribution for age groups
             region (string): key for region
             landsc (int): value for landscape with 30 (city) and 60 (rural area)
             sum (int): number of members_institutions to sample
             df_age_gr (DataFrame): Hold translation age to age group
             agents_data (dict(ndarray)): data of agents in the pool to choose from
             sex (int): 0 for male and 1 for female

        Returns:
            agents_data (dict(ndarray)): data of agents in the pool after deletion of
            sampled agents (ids, ages, sexes, z_codes, landsc)
            sampled_data_all (dict(ndarray)): data of sampled members_institutions
            (ids, ages, sexes, z_codes, landsc)
        """
        # sample age group of member according to age distribution
        age_gr_mbr = rm.choice(age_grs, sum, p=age_dist)
        # get age groups and set array to count age groups

        age_groups = np.arange(min(age_gr_mbr), max(age_gr_mbr) + 1)
        sampled_data_all = self.create_empty_data_dict(
            [cs.ID, cs.AGE, cs.SEX, cs.Z_CODE, cs.LANDSCAPE, cs.SOCIABLE]
        )
        for b in age_groups:
            # get ages for age group
            age_min = df_age_gr.loc[b, "Min bracket"]
            age_max = df_age_gr.loc[b, "Max bracket"]
            sum_age_gr = (age_gr_mbr == b).sum()
            sampled_data, agents_data = self.sample_member_from_pool(
                age_min,
                region,
                landsc,
                sum_age_gr,
                agents_data,
                sex,
                age_max,
                member,
            )
            sampled_data_all = self.concatenate_data_dictionary(
                sampled_data_all, sampled_data
            )

        return sampled_data_all, agents_data

    def sample_lost_due_to_gap_request_exist(
        self,
        lost,
        age_min,
        region,
        landsc,
        agents_data,
        sex=None,
        age_max=99,
        member=None,
    ):
        """Sample people form the pool widening the range of specified
        attributes in each loop.

        Call method, when people with specified attributes such as a_gr
        /sex are no more in the pool to sample from. Sample the lost in
        a loop until all lost are sampled. Widen the mean_contacts_p_age
        by +-5 in each loop. If loop is called 10 times, disregard sex
        and sample again.

        :param lost: number lost
        :param age_min: original age to sample from
        :param region: code
        :param landsc: 30 (city), 60 (landscape)
        :param agents_data:
        :param sex: 1 female, 0 male
        :param age_max: specified or 99
        :param member: in some cases it can be necessary to specify
            parent, children etc.
        :return: agents_data (dict(ndarray)): data of agents in the pool
            after deletion of sampled agents (ids, ages, sexes, z_codes,
            landsc)
        :return: data_mbr_lost (dict(ndarray)): data of sampled lost
            members_institutions (ids, ages, sexes, z_codes, landsc)
        """
        ids, ages, sexes, z_codes, landscapes, _ = self.extract_agents_data(agents_data)
        if sum((z_codes == region) & (landscapes == landsc)) < lost:
            raise AssertionError(
                "Not enough people for requested number of household"
                " members_institutions"
            )
        if member == "parent":
            low_rg = 20
            up_rg = 69
        elif member == "child":
            low_rg = 0
            up_rg = 29
        else:
            low_rg = 0
            up_rg = 99

        count = 0
        while lost > 0:
            age_max, age_min = self.adopt_range_of_age_group(
                age_max, age_min, low_rg, member, up_rg
            )
            if (sex is not None) and (count < 10):
                my_ids = ids[
                    (ages >= age_min)
                    & (ages <= age_max)
                    & (sexes == sex)
                    & (z_codes == region)
                    & (landscapes == landsc)
                ]
            elif (sex is None) and (count < 10):
                my_ids = ids[
                    (ages >= age_min)
                    & (ages <= age_max)
                    & (z_codes == region)
                    & (landscapes == landsc)
                ]
            else:
                my_ids = ids[(z_codes == region) & (landscapes == landsc)]
            if len(my_ids) >= lost:
                ids_mbr_lost = rm.choice(my_ids, lost, replace=False)
                lost = 0
            else:
                count += 1

        agents_data, data_mbr_lost = self.split_sampled_data_from_agents_data(
            ids, ids_mbr_lost, agents_data
        )

        return data_mbr_lost, agents_data

    def adopt_range_of_age_group(self, age_max, age_min, low_rg, member, up_rg):
        if (member == "parent") or (member == "mate") or (member == "couple"):
            age_max = min(age_max + 5, up_rg)
        elif member == "child":
            age_min = max(age_min - 5, low_rg)
        else:
            age_min = max(age_min - 5, low_rg)
            age_max = min(age_max + 5, up_rg)
        return age_max, age_min

    def ensure_request_exist_match(
        self, age_min, region, landsc, sum_hh, agents_data, sex=None, age_max=99
    ):
        # ensure requested number <= existing number
        ids, ages, sexes, z_codes, landscapes, _ = self.extract_agents_data(agents_data)
        if sex is None:
            possible_count = np.where(
                (ages >= age_min)
                & (ages <= age_max)
                & (z_codes == region)
                & (landscapes == landsc),
                1,
                0,
            ).sum()
        else:
            possible_count = np.where(
                (ages >= age_min)
                & (ages <= age_max)
                & (sexes == sex)
                & (z_codes == region)
                & (landscapes == landsc),
                1,
                0,
            ).sum()
        lost = 0
        while possible_count < sum_hh:
            sum_hh -= 1
            lost += 1
        return lost, sum_hh

    def sort_and_count_ages_by_age_group(self, ages_mbr, df_age_gr):
        """Count number of people according to ages and filter indices of people in the array.
        Args:
            ages_mbr (ndarray(int)): Hold ages of hh members_institutions.
            df_age_gr (DataFrame): Hold translation age to age group

        Returns:
            age_groups_count (ndarray(int)): Counts for each age group.
            age_groups_indices(ndarray(ndarray)): Indices where the members_institutions of age group were found.

        """
        # get age groups and count for age groups
        age_groups = np.arange(len(df_age_gr))
        age_groups_count = np.zeros(len(df_age_gr), dtype=int)
        age_groups_indices = np.zeros(len(df_age_gr), dtype=list)
        # iterate age groups
        for a in age_groups:
            # get range of ages for age group
            age_min = df_age_gr.loc[a, "Min bracket"]
            age_max = df_age_gr.loc[a, "Max bracket"]
            # filter ages, get sum and indices
            filt = np.where((ages_mbr >= age_min) & (ages_mbr <= age_max), 1, 0)
            age_groups_count[a] = filt.sum()
            age_groups_indices[a] = np.argwhere(filt).reshape((filt == 1).sum())

        return age_groups_count, age_groups_indices

    def set_possible_age_groups_for_other_members(
        self, a, age_gr_min, age_gr_max, low_rg, up_rg, age_dist, df_age_gr
    ):
        """Cut array of age groups and age distribution, if mean_contacts_p_age
        is determined by min or max.

        For instance: The first hh member is belongs to age group 2. The range of age for another member is
        within +-4 age groups (for instance +-20 Years). But the age groups to choose the second member can not be -2.
        So the lowest possible age group to choose from must be age group 0. Same with the highest possible age group.

        Args:
             age_gr_min (int): minimum age group to choose from
             age_gr_max (int): maximum age group to choose from
             a (int): age of the other member
             low_rg (int): lower boundary of age groups related to a_gr (is neg. for age group below a_gr)
             up_rg (int): upper boundary of age groups related to a_gr (is neg. for age group below a_gr and pos. upper a_gr)
             age_dist list(float): general age distribution for that range

        Returns:
            age_dist (list(float)): final age distribution
            age_gr_other_mbr (ndarray(int)): possible inds_age_gr for other member
        """
        if (low_rg < 0) or (up_rg < 0):
            raise AssertionError(
                "Values for lower and upper range must be greater zero"
            )
        if (
            (age_gr_min < 0)
            or (age_gr_min > len(df_age_gr))
            or (age_gr_max < 0)
            or (age_gr_max > len(df_age_gr))
        ):
            raise AssertionError(
                "Age group min and max must be within range of age groups"
            )
        if age_gr_min > age_gr_max:
            raise AssertionError("Age group min must be lower age group max")
        if low_rg + up_rg + 1 != len(age_dist):
            raise AssertionError(
                "Range of age groups around age must be of same length as age"
                " distribution"
            )

        # case 1: the given range of age groups is on lower border
        elif (
            (a - low_rg < age_gr_min)
            and (a + up_rg <= age_gr_max)
            and (a + up_rg > age_gr_min)
        ):
            age_gr_other_mbr = np.arange(age_gr_min, a + up_rg + 1)
            index = abs(age_gr_min - (a - low_rg))
            # cut the reduction from the head
            age_dist = age_dist[index:]

        # case 2: the given range of age groups is on upper border
        elif (
            (a + up_rg > age_gr_max)
            and (a - low_rg >= age_gr_min)
            and (a - low_rg < age_gr_max)
        ):
            age_gr_other_mbr = np.arange(a - low_rg, age_gr_max + 1)
            index = len(age_dist) - abs(age_gr_max - (a + up_rg))
            # cut the reduction from the tail
            age_dist = age_dist[:index]

        # case 3: the given range of age groups is out of lower border
        elif a + up_rg <= age_gr_min:
            age_gr_other_mbr = np.arange(age_gr_min, age_gr_min + 1)
            age_dist = age_dist[-1:]

        # case 4: the given range of age groups is out of upper border
        elif a - low_rg >= age_gr_max:
            age_gr_other_mbr = np.arange(age_gr_max, age_gr_max + 1)
            age_dist = age_dist[:1]

        # case 5: the given range of age groups is on lower and upper border
        elif (a - low_rg < age_gr_min) and (a + up_rg > age_gr_max):
            age_gr_other_mbr = np.arange(a - low_rg, a + up_rg + 1)
            low_index = abs(age_gr_min - (a - low_rg))
            up_index = len(age_dist) - abs(age_gr_max - (a + up_rg))
            age_dist = age_dist[low_index:up_index]
        else:
            age_gr_other_mbr = np.arange(a - low_rg, a + up_rg + 1)

        age_dist = [x / sum(age_dist) for x in age_dist]

        return age_dist, age_gr_other_mbr

    def split_sampled_data_from_agents_data(self, data, sub_data, agents_data):
        """Delete sampled data from agents_data and save them to a separate
        dict.

        :param data (ndarray): e.g. ids
        :param data_groups(ndarray): sampled ids to delete from
            agents_data
        :param agents_data (dict(ndarray)): data to delete agents from
        :return: agents_data (dict(ndarray)): data of agents in the pool
            after deletion of sampled agents
        :return: data_mbr (dict(ndarray)): agents data of sampled people
        """
        indices = np.argwhere(np.where(np.isin(data, sub_data), 1, 0))
        indices = indices.reshape(len(indices))
        data_mbr = self.extract_data_of_specified_indices(indices, agents_data)
        agents_data = self.delete_indices(indices, agents_data)
        return agents_data, data_mbr

    def filter_agents_data_by_subdata(self, general_key, key, agents_data):
        indices = np.argwhere(np.where(agents_data[general_key] == key, 1, 0))
        indices = indices.reshape(len(indices))
        filtered_data = self.extract_data_of_specified_indices(indices, agents_data)
        return filtered_data

    def extract_agents_data(self, agents_data):
        ids = agents_data[cs.ID]
        ages = agents_data[cs.AGE]
        sexes = agents_data[cs.SEX]
        z_codes = agents_data[cs.Z_CODE]
        landscapes = agents_data[cs.LANDSCAPE]
        sociables = agents_data[cs.SOCIABLE]
        if cs.OCCUPATION in agents_data:
            occupations = agents_data[cs.OCCUPATION]
            return ids, ages, sexes, z_codes, landscapes, occupations, sociables
        else:
            return ids, ages, sexes, z_codes, landscapes, sociables

    def extract_data_of_specified_indices(self, indices, agents_data):
        """Extract sampled data from agents data and save them in a separate
        dict.

        :param indices: indices of sampled members_institutions
        :param agents_data: original agents data
        :return (dict(ndarray): sampled people
        """
        # indices = np.sort(indices)
        sampled_data = {}
        if cs.OCCUPATION in agents_data:
            (
                ids,
                ages,
                sexes,
                z_codes,
                landscapes,
                occupations,
                sociables,
            ) = self.extract_agents_data(agents_data)
            sampled_data[cs.OCCUPATION] = occupations[indices].reshape(len(indices))
        else:
            ids, ages, sexes, z_codes, landscapes, sociables = self.extract_agents_data(
                agents_data
            )

        sampled_data[cs.ID] = ids[indices].reshape(len(indices))
        sampled_data[cs.AGE] = ages[indices].reshape(len(indices))
        sampled_data[cs.SEX] = sexes[indices].reshape(len(indices))
        sampled_data[cs.Z_CODE] = z_codes[indices].reshape(len(indices))
        sampled_data[cs.LANDSCAPE] = landscapes[indices].reshape(len(indices))
        sampled_data[cs.SOCIABLE] = sociables[indices].reshape(len(indices))

        return sampled_data

    def extract_regions_from_people_class(self, people):
        z_codes = people.get_data_for(cs.Z_CODE).astype(int)
        regions = np.unique(z_codes)
        return regions

    def delete_indices(self, indices, agents_data):
        if cs.OCCUPATION in agents_data:
            (
                ids,
                ages,
                sexes,
                z_codes,
                landscapes,
                occupations,
                sociable,
            ) = self.extract_agents_data(agents_data)
        else:
            ids, ages, sexes, z_codes, landscapes, sociable = self.extract_agents_data(
                agents_data
            )
        agents_data[cs.ID] = np.delete(ids, indices)
        agents_data[cs.AGE] = np.delete(ages, indices)
        agents_data[cs.SEX] = np.delete(sexes, indices)
        agents_data[cs.Z_CODE] = np.delete(z_codes, indices)
        agents_data[cs.LANDSCAPE] = np.delete(landscapes, indices)
        agents_data[cs.SOCIABLE] = np.delete(sociable, indices)
        if cs.OCCUPATION in agents_data:
            agents_data[cs.OCCUPATION] = np.delete(occupations, indices)
        return agents_data

    def create_empty_data_dict(self, keys):
        data = {}
        for key in keys:
            data[key] = np.zeros(0, dtype=int)
        return data

    def concatenate_data_dictionary(self, data_dict_1, data_dict_2):
        assert (
            data_dict_1.keys() == data_dict_2.keys()
        ), "The keys of the dictionaries must be the same."
        data_dict_3 = {}
        for key in data_dict_1.keys():
            data_dict_3[key] = np.concatenate((data_dict_1[key], data_dict_2[key]))

        return data_dict_3

    def get_age_min_max_from_age_gr(self, age_gr, df):
        age_min = df.loc[age_gr, "Min bracket"]
        age_max = df.loc[age_gr, "Max bracket"]

        return age_min, age_max

    def get_data_from_ids_agents_data(self, ids, data_key, agents_data):
        # get copy of agents data from people class
        ids_pool = agents_data[cs.ID]
        key_data_pool = agents_data[data_key]

        ids = np.sort(ids)
        indices = np.argwhere(np.where(np.isin(ids_pool, ids), 1, 0))
        indices = indices.reshape(len(indices))
        data = key_data_pool[indices]

        return data

    def get_age_gr_from_age(self, age, df):
        filt = (age >= df["Min bracket"]) & (age <= df["Max bracket"])
        age_gr = df.loc[
            filt,
        ].index
        return age_gr[0]

    # def determine_mean_number_of_contacts(self):
    # ["parent id", "id", "type", "size", "mean contacts", "year", "weekday"]
    # agents_data = self.people.get_agents_data()
    # age_mean = self.determine_mean_contacts_sub(agents_data, self.sub_nets)
    # self.mean_contacts_p_age["Mean contacts"] = age_mean

    def determine_mean_contacts_region(self):
        # ["parent id", "id", "type", "size", "mean contacts", "year", "weekday"]
        all_agents_data = self.people.get_agents_data()
        for region in self.regions:
            agents_data = self.filter_agents_data_by_subdata(
                cs.Z_CODE, region, all_agents_data
            )
            if region == 11000:
                filt = [int(str(x)[1:3]) == region / 1000 for x in self.sub_nets[:, 0]]
            else:
                filt = [int(str(x)[1:6]) == region for x in self.sub_nets[:, 0]]
            sub_nets = self.sub_nets[
                filt,
            ]
            age_mean = self.determine_mean_contacts_sub(agents_data, sub_nets)
            self.mean_contacts_p_age[region] = age_mean

    def determine_mean_contacts_sub(self, agents_data, sub_nets):
        # ["parent id", "id", "type", "size", "mean contacts", "year", "weekday"]
        ids = agents_data[cs.ID]
        num_contacts = np.zeros(ids.shape, dtype=float)
        for i in range(len(sub_nets)):
            act_id = sub_nets[i, 1]
            my_num = sub_nets[i, 4]
            if type(sub_nets[i, 6]) is np.ndarray:
                p_day = len(sub_nets[i, 6]) / 7
            elif type(self).__name__.lower() == "households":
                p_day = 7 / 7
            else:
                p_day = 5 / 7
            my_members = self.members_groups[act_id]
            agents_idx = np.nonzero(np.isin(ids, my_members))[0]
            num_contacts[agents_idx] += my_num * p_day
        mean_p_age = np.zeros(len(self.mean_contacts_p_age["Age_gr"]))
        for idx, age_gr in enumerate(self.mean_contacts_p_age["Age_gr"]):
            age_ids = self.extract_people_from_age_group(age_gr, agents_data)
            age_idx = np.nonzero(np.isin(agents_data[cs.ID], age_ids))[0]
            if len(age_idx) > 0:
                age_mean = np.mean(num_contacts[age_idx])
            else:
                age_mean = np.nan
            mean_p_age[idx] = age_mean
        return mean_p_age

    def get_mean_contacts_per_age_gr(self):
        return self.mean_contacts_p_age["Mean contacts"]

    def extract_people_from_age_group(self, age_gr, agents_data):
        ids = agents_data[cs.ID]
        a_min = self.mean_contacts_p_age["Min"][age_gr]
        a_max = self.mean_contacts_p_age["Max"][age_gr]
        cond1 = agents_data[cs.AGE] >= a_min
        cond2 = agents_data[cs.AGE] <= a_max
        my_people = ids[cond1 & cond2]
        return my_people

    def open_current_mean_contacts(self):
        abs_file_name = PM.get_path_current_mean_contacts(self.scale)
        if not os.path.exists(abs_file_name):
            Institution.create_empty_table_mean_contacts(self.scale)
        df = pd.read_csv(abs_file_name, index_col=[0, 1])
        return df

    def save_current_mean_contacts(self):
        df = self.open_current_mean_contacts()
        for region in self.regions:
            data = self.mean_contacts_p_age[region]
            name = type(self).__name__.lower()
            df.loc[region, name] = data
        abs_file_name = PM.get_path_current_mean_contacts(self.scale)
        FileSaver.save_csv_table(abs_file_name, df)

    @staticmethod
    def create_empty_table_mean_contacts(scale):
        path_counties = PM.get_path_counties()
        df_counties = pd.read_excel(path_counties)
        counties = df_counties["Schlüsselnummer"]
        iterables = [counties, range(10)]
        multi_index = pd.MultiIndex.from_product(iterables, names=["region", cs.AGE_GR])
        columns = [cs.HOUSEHOLDS, cs.SCHOOLS, cs.WORKPLACES, cs.ACTIVITIES]
        df = pd.DataFrame(index=multi_index, columns=columns)
        abs_file_name = PM.get_path_current_mean_contacts(scale)
        df.to_csv(abs_file_name, index=True)

    # def load_current_mean_contacts(self, region, how="self"):
    #     name = type(self).__name__.lower()
    #     data = Sync.load_current_mean_contacts(region, name, how)
    #     return data

    def load_target_mean_contacts(self):
        abs_file_name = PM.get_path_target_mean_contacts()
        df_target = pd.read_excel(abs_file_name)
        target = df_target["mean contacts"].to_numpy()
        # TODO(Felix): Caution! The age groups in the DataFrame do not coincide with the
        #   age groups in the simulation.
        return target

    def get_contact_diff_current_target(self, *networks):
        # TODO: remove this method later
        # This method is only for steps in between. Not really used.
        age_gr_dict = Institution.get_age_gr_dict()
        current = np.zeros(len(age_gr_dict["Age_gr"]))
        mean_random = 1.0
        # TODO: create array with correction_factor for high or low mean_numbers
        for network in networks:
            current += network.get_mean_contacts_per_age_gr()
        current += mean_random
        # save results for copy and paste
        target_file = PM.get_path_target_mean_contacts
        df_target = pd.read_excel(target_file)
        target = df_target["Mean contacts"].to_numpy()
        diff = target - current
        return diff
