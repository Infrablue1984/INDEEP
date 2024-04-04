#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Docstring."""

__author__ = "Inga Franzen"
__created__ = "2020"
__date_modified__ = "2023/05/31"
__version__ = "1.0"


import os
import numpy as np
import pandas as pd
from numpy import random as rm

from synchronizer import constants as cs
from synchronizer.synchronizer import FileLoader
from synchronizer.synchronizer import PathManager as PM


class People:
    """Class to create and manage peoples data and their relations."""

    AGE_CUTOFFS = {
        cs.AGE_GR: np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        cs.MIN: np.array([0, 5, 10, 15, 20, 30, 40, 50, 60, 70]),
        cs.MAX: np.array([4, 9, 14, 19, 29, 39, 49, 59, 69, 99]),
    }

    def __init__(self, regions, scale=100):
        """Extract data and initialize people from different region and their
        relations.

        Args:
                regions (list(list)): Hold key of region.

        Returns: None
        """
        self.scale = scale
        self.counties = regions
        self._agents_data = {}
        self._pop_size = 0

    @staticmethod
    def reset_data(scale):
        new_data = np.arange(0)
        for key in [
            cs.ID,
            cs.Z_CODE,
            cs.LANDSCAPE,
            cs.AGE,
            cs.SEX,
            cs.OCCUPATION,
            cs.SOCIABLE,
        ]:
            abs_file_name = PM.get_path_agents_data(key, scale)
            np.save(abs_file_name, new_data)

    def initialize_data(self):
        """initialize agents data (dict(key : ndarray)): Hold array with information for each agent and key.
        Regional dependant properties are specified, non-regional dependant remain default and will be specified later.
        Args:
                regions (list(list)): Hold keys for region.

        Returns:
        """
        self._agents_data[cs.ID] = self.__make_ids()

        # extract regional code as attribute
        # self._agents_data['z_codes'] = np.full(len(self._agents_data['ids']), 5162)
        self._agents_data[cs.Z_CODE] = self.__make_z_code()

        # extract region urban 30 or rural 60
        # self._agents_data['landscapes'] = np.full(len(self._agents_data['ids']), 30)
        self._agents_data[cs.LANDSCAPE] = self.__make_landscape()

        # create ages for
        self._agents_data[cs.AGE] = self.__make_ages()

        # create sexes
        self._agents_data[cs.SEX] = self.make_sexes()

        # create occupation
        self._agents_data[cs.OCCUPATION] = self.make_occupation()

        # create sociable
        self._agents_data[cs.SOCIABLE] = self.make_sociable()

        self._pop_size = self.get_pop_size()

        return

    def reload_data(self):
        """initialize agents data (dict(key : ndarray)): Hold array with information for each agent and key.
        Regional dependant properties are specified, non-regional dependant remain default and will be specified later.
        Args:
                regions (list(list)): Hold keys for region.

        Returns:
        """
        self._agents_data[cs.Z_CODE] = FileLoader.load_agents_data(
            cs.Z_CODE, self.scale
        )
        idx = np.nonzero(np.isin(self._agents_data[cs.Z_CODE], self.counties))[0]
        for key in [
            cs.ID,
            cs.LANDSCAPE,
            cs.AGE,
            cs.SEX,
            cs.OCCUPATION,
            cs.SOCIABLE,
            cs.Z_CODE,
        ]:
            self._agents_data[key] = FileLoader.load_agents_data(key, self.scale)[idx]

    def get_pop_size(self):
        """Get total pop_size.

        Returns:
                pop_size (int): total size of population
        """
        pop_size = len(self._agents_data[cs.ID])

        return pop_size

    def __make_ids(self):
        """Filter ids for Germany, the choosen counties or federal states out
        of an existing table and create ndarray with personal ids.

        Args: region(list(list)): Hold keys for region.

        Returns:
                ids ndarray: Hold ids for population
        """

        # read files
        df_counties = pd.read_excel(PM.get_path_counties())
        df_counties.set_index("Schlüsselnummer", inplace=True)
        ids = np.arange(0)
        # iterate counties
        for county in self.counties:
            # get size of county population
            county_size = np.int64(df_counties.loc[county, "Bevölkerung"] / self.scale)
            county_ids = np.arange(county_size) + county * 10**5
            ids = np.append(ids, county_ids)
        return ids

    def __make_z_code(self):
        """Filter ids for Germany, the choosen counties or federal states out
        of an existing table and create ndarray with z_codes.

        Args: region(list(list)): Hold keys for region.

        Returns:
                z_codes ndarray: Hold z_codes for each person
        """
        z_code = (self._agents_data[cs.ID] / 10**5).astype(int)

        return z_code

    def __make_landscape(self):
        """Filter ids for Germany, the choosen counties or federal states out
        of an existing table and create ndarray with z_codes.

        Args: region(list(list)): Hold keys for region.

        Returns:
                landscapes ndarray: Hold code for landscapes for each person 30:urban, 60:rural
        """

        # read files
        df_counties = pd.read_excel(PM.get_path_counties())
        df_counties.set_index("Schlüsselnummer", inplace=True)
        z_codes = self._agents_data[cs.Z_CODE]
        landscape = np.zeros(len(z_codes), dtype=int)
        for county in self.counties:
            # get size of county population
            inds = np.nonzero(z_codes == county)[0]
            county_rural_pop = int(df_counties.loc[county, "ländlich"] / self.scale)
            county_urban_pop = len(inds) - county_rural_pop
            assert abs(len(inds) - county_rural_pop - county_urban_pop) <= 1
            landscape[inds[:county_urban_pop]] = 30
            landscape[inds[county_urban_pop:]] = 60

        return landscape

    def __make_ages(self):
        """Set ages for agents dependant on regional data.

        Args:
                region(list(list)): Hold keys for region.

        Returns:
                ages (ndarray)): Hold array with age for each agent.
        """
        # read data
        df_ages = pd.read_excel(PM.get_path_age_dist())
        df_ages.set_index("Alter", inplace=True)

        # define age groups
        age_groups = np.arange(len(df_ages))

        # get z_codes from agents_data
        z_codes = self._agents_data[cs.Z_CODE]

        # ToDo: create age table considering landscape
        # create array for ages
        ages = np.arange(len(z_codes), dtype=int)
        state_arr = (z_codes / 1000).astype(int)
        states = np.unique(state_arr)
        # sample ages by age_distribution from states
        for state in states:
            inds = np.nonzero(state_arr == state)[0]
            age_dist = np.array(df_ages[state])
            ages[inds] = rm.choice(age_groups, len(inds), p=age_dist)
        return ages

    def make_sexes(self):
        """Set sexes for agents dependant on ages.

        Args:
                region (list(list)): Hold keys for region.
        Returns:
                sexes (ndarray)): Hold array with sex for each agent.
        """
        # read data
        df_sexes = pd.read_excel(PM.get_path_gender_dist(), na_values=["NA"])
        df_sexes.set_index("Alter", inplace=True)
        ages = self._agents_data[cs.AGE]
        sexes = np.empty(len(ages), dtype=int)
        age_groups = np.array(df_sexes.index)

        # define sexes
        for age in df_sexes.index:
            sex_prob = [df_sexes["männlich"][age], df_sexes["weiblich"][age]]
            nmb_of_age_group = np.sum(np.isin(ages, age))
            sexes[(np.isin(ages, age))] = rm.choice(
                [0, 1], nmb_of_age_group, p=sex_prob
            )

        return sexes

    def make_occupation(self):
        """Set occupation for agents dependant on regional data.

        e.g. "worker", "kita_child", "pupil", "retired", "None", "medical", "education", "elderly care", "gastronomy"

        Args:
                region (list(list)): Hold keys for region.

        Returns:
                occ (ndarray)): Hold array with occupation for each agent.
        """
        abs_file_name = PM.get_path_occupation_dist()
        # read data
        df_fem_occ = pd.read_excel(
            abs_file_name,
            "female occupations",
            na_values=["NA"],
        )
        df_male_occ = pd.read_excel(
            abs_file_name,
            "male occupations",
            na_values=["NA"],
        )
        # df_fem_occ(columns={'employed': 'w', 'pupil': 'p', 'kita': 'k', 'retired': 'r', 'None': 'n'}, inplace=True)
        # df_male_occ(columns={'employed': 'w', 'pupil': 'p', 'kita': 'k', 'retired': 'r', 'None': 'n'}, inplace=True)
        ages = self._agents_data[cs.AGE]
        sexes = self._agents_data[cs.SEX]
        occ = np.empty(len(ages), dtype="U12")

        # define general occupation

        for i in range(len(df_fem_occ)):
            age_group = np.arange(
                df_fem_occ["Min bracket"][i], df_fem_occ["Max bracket"][i] + 1
            )
            nmb_of_age_group_fem = np.sum((np.isin(ages, age_group)) & (sexes == 1))
            nmb_of_age_group_male = np.sum((np.isin(ages, age_group)) & (sexes == 0))
            occ_prob_fem = df_fem_occ.iloc[i, 2:]
            occ_prob_male = df_male_occ.iloc[i, 2:]
            occ[(np.isin(ages, age_group)) & (sexes == 1)] = rm.choice(
                df_fem_occ.columns[2:], nmb_of_age_group_fem, p=occ_prob_fem
            )
            occ[(np.isin(ages, age_group)) & (sexes == 0)] = rm.choice(
                df_male_occ.columns[2:], nmb_of_age_group_male, p=occ_prob_male
            )

        # define special groups
        df_fem_spec = pd.read_excel(
            abs_file_name,
            "special female",
            na_values=["NA"],
        )
        df_male_spec = pd.read_excel(
            abs_file_name,
            "special male",
            na_values=["NA"],
        )
        # df.fem_spec(columns={'medical': 'm', 'elderly care': 'c', 'education': 'e', 'gastronomy': 'g', 'employed': 'e'}
        # , inplace=True)

        # drop last row ('total)
        df_fem_spec.drop(columns="total", inplace=True)
        df_male_spec.drop(columns="total", inplace=True)
        df_fem_spec.drop(df_fem_spec.tail(1).index, inplace=True)
        df_male_spec.drop(df_male_spec.tail(1).index, inplace=True)
        # df_spec_male_occ.set_index(['Occupation'], inplace = True)
        # spec_prob = df_spec_occ['%']
        for i in range(len(df_fem_spec)):
            age_group = np.arange(
                df_fem_spec["Min bracket"][i], df_fem_spec["Max bracket"][i] + 1
            )
            nmb_of_fem = np.sum(
                (np.isin(ages, age_group)) & (np.isin(occ, "employed")) & (sexes == 1)
            )
            nmb_of_male = np.sum(
                (np.isin(ages, age_group)) & (np.isin(occ, "employed")) & (sexes == 0)
            )
            fem_prob = df_fem_spec.iloc[i, 2:]
            male_prob = df_male_spec.iloc[i, 2:]
            occ[
                (np.isin(ages, age_group)) & (np.isin(occ, "employed")) & (sexes == 1)
            ] = rm.choice(df_fem_spec.columns[2:], nmb_of_fem, p=fem_prob)
            occ[
                (np.isin(ages, age_group)) & (np.isin(occ, "employed")) & (sexes == 0)
            ] = rm.choice(df_male_spec.columns[2:], nmb_of_male, p=male_prob)

        # change occupatiion to shortcut --> less bytes and memory
        occ[occ == "medical"] = "m"
        occ[occ == "elderly care"] = "c"
        occ[occ == "education"] = "e"
        occ[occ == "gastronomy"] = "g"
        occ[occ == "employed"] = "w"  # for work
        occ[occ == "pupil"] = "p"
        occ[occ == "kita"] = "k"
        occ[occ == "retired"] = "r"
        occ[occ == "None"] = "N"
        occ = occ.astype("U1", copy=False)
        for i in range(len(df_fem_occ)):
            age_group = np.arange(
                df_fem_occ["Min bracket"][i], df_fem_occ["Max bracket"][i] + 1
            )
            idx = np.argwhere(np.where(np.isin(ages, age_group), 1, 0))
            test = occ[(idx)]
            test2 = ages[(idx)]

        return occ

    def make_sociable(self):
        # extract data
        ages = self._agents_data[cs.AGE]
        ids = self._agents_data[cs.ID]
        p = 0.30  # set percentage sociable
        sociable = np.zeros(len(ages))  # initialize
        # iterate age gr to ensure attribute is well distributed
        for age_gr in People.AGE_CUTOFFS[cs.AGE_GR]:
            a_min = People.AGE_CUTOFFS[cs.MIN][age_gr]
            a_max = People.AGE_CUTOFFS[cs.MAX][age_gr]
            cond1 = ages >= a_min
            cond2 = ages <= a_max
            my_people = ids[cond1 & cond2]
            num = round(len(my_people) * p)
            people_s = rm.choice(my_people, num, replace=False)
            sociable[np.isin(ids, people_s)] = 1
        return sociable

    def initialize_use_of_mobile_app(self, properties):
        """Determine agents with use of mobile app dependant on %use and age
        distribution.

        set field "mobile_app" in self._agents_data
        """
        # read data
        df_mobile = pd.read_excel("../data/mobile_app.xls", na_values=["NA"])
        mobile = properties["mobile_app"]
        ages = properties[cs.AGE]
        mobile_dist = df_mobile["Proportion"]
        per_of_use = 0.2  # normally gathered by intervention 'mobile app'

        # define use of mobile by age groups
        total_nmb_of_agents_with_app = int(per_of_use * self._pop_size)
        age_groups = np.arange(len(df_mobile))
        mobile_by_age_groups = rm.choice(
            age_groups, total_nmb_of_agents_with_app, p=mobile_dist
        )

        # define who has mobile app
        for i in range(len(df_mobile)):
            ages_in_group = np.arange(
                df_mobile["Min bracket"][i], df_mobile["Max bracket"][i] + 1
            )
            nmb_of_age_group = np.sum(np.isin(ages, ages_in_group))
            nmb_with_app_by_age_group = np.sum(np.isin(mobile_by_age_groups, i))
            if nmb_of_age_group == 0:
                prob_with_app_by_age_group = 0
            else:
                prob_with_app_by_age_group = (
                    nmb_with_app_by_age_group / nmb_of_age_group
                )
            mobile_prob = [
                prob_with_app_by_age_group,
                1 - prob_with_app_by_age_group,
            ]
            mobile[(np.isin(ages, ages_in_group))] = rm.choice(
                [True, False], nmb_of_age_group, p=mobile_prob
            )

        return mobile

    def get_data_for(self, key):
        """return agents data for specific key
        Args:
                key (str): Specific agents property, e.g. 'age' or 'recovered'.

        Returns:
                _agents_data[key] (ndarray): Hold values for agents at that key
        """
        return self._agents_data[key].copy()

    def get_agents_data(self):
        """return agents data
        Args:
                None

        Returns:
                _agents_data(dict): Hold all keys and values (ndarrays) for agents.
        """
        return self._agents_data.copy()

    def get_regions(self):
        return self.counties

    def get_data_from_ids(self, ids, data_key):
        # get copy of agents data from people class
        ids_pool = self.get_data_for(cs.ID)
        key_data_pool = self.get_data_for(data_key)
        indices = np.argwhere(np.where(np.isin(ids_pool, ids), 1, 0))
        indices = indices.reshape(len(indices))
        data = key_data_pool[indices]

        return data

    def get_ID_from_array(self, bool_array):
        """Transform a_gr boolean ndarray into a_gr list of IDs for True
        values.

        Args:
                bool_array (ndarray(bool)) : with True for ID to be gathered

        Returns:
                set_of_IDs (set) : set with agents IDs
        """
        if not isinstance(bool_array, np.ndarray):
            raise TypeError("The Input value must be an ndarray")
        if len(bool_array) != self._pop_size:
            raise IndexError("Length of array must go along with total _pop_size")
        if bool_array.dtype != bool:
            raise TypeError("The dtype of ndarray must be Boolean")

        # try:
        # take IDs of relevant agents
        selected_ID = np.zeros(self._pop_size).astype(str)
        selected_ID[(bool_array is True)] = self._agents_data[cs.ID][
            (bool_array is True)
        ]

        # create a_gr set with no double values (e.g. '0.0')
        set_of_ID = set(selected_ID)

        # remove None Value
        if "0.0" in set_of_ID:
            set_of_ID.remove("0.0")

        return set_of_ID

    def get_index_from_array(self, bool_array):
        """Transform a_gr boolean ndarray into a_gr list of indices for True
        values.

        Args:
                bool_array (ndarray(bool)) : with True for ID to be gathered

        Returns:
                list_of_indices (list) : set with agents indices
        """
        if not isinstance(bool_array, np.ndarray):
            raise TypeError("The Input value must be an ndarray")
        if len(bool_array) != self._pop_size:
            raise IndexError("Length of array must go along with total _pop_size")
        if bool_array.dtype != bool:
            raise TypeError("The dtype of ndarray must be Boolean")

        indices = np.where(bool_array == 1)
        list_of_indices = indices[0].tolist()
        return list_of_indices

    def get_array_from_ID(self, set_of_IDs):
        """Transform a_gr set of IDs into a_gr boolean ndarray with True, if
        agent in set and False if not.

        Args:
                set_of_IDs (set) : set with agents IDs

        Returns:
                bool_array (ndarray(bool)) : With True, if agent was in set and False if not
        """

        # Raise Exceptions
        if not isinstance(set_of_IDs, set):
            raise TypeError("The Input value must be a_gr set")
        for x in set_of_IDs:
            if type(x) != str:
                raise TypeError("The type of the set elements must be String")
            if np.isin(x, self._agents_data["id"], invert=True):
                raise AssertionError("Some IDs are not valid in the population")

        # remove None Value
        if "0.0" in set_of_IDs:
            set_of_IDs.remove("0,0")

        # get array of IDs
        bool_array = np.zeros(self._pop_size, dtype=bool)
        for ID in set_of_IDs:
            bool_array = np.where(self._agents_data[cs.ID] == ID, True, bool_array)

        return bool_array

    def get_index_from_ID(self, set_of_IDs):
        """Transform a_gr set of IDs into a_gr list of indices for agents in
        set.

        Args:
                set_of_IDs (set) : set with agents IDs

        Returns:
                list_of_indices (list) : set with agents indices
        """
        # Raise Exceptions
        if not isinstance(set_of_IDs, set):
            raise TypeError("The Input value must be a_gr set")
        for x in set_of_IDs:
            if type(x) != str:
                raise TypeError("The type of the set elements must be String")
            if np.isin(x, self._agents_data["id"], invert=True):
                raise AssertionError("Some IDs are not valid in the population")

        bool_array = self.get_array_from_ID(set_of_IDs)
        list_of_indices = self.get_index_from_array(bool_array)
        return list_of_indices

    """"""

    def get_array_from_index(self, list_of_indices):
        """Transform a_gr list with agents indices into an array with True at
        indices that are in the list and False for not.

        Args:
                list_of_indices (list) : list with relevant agents indices

        Returns:
                bool_array (ndarray(bool)) : With True, if agents index was in list
        """
        # Raise Exceptions
        if not (
            (isinstance(list_of_indices, set)) or (isinstance(list_of_indices, list))
        ):
            raise TypeError("The Input value must be a_gr set or list")

        bool_array = np.zeros(self._pop_size, dtype=bool)
        for index in list_of_indices:
            if type(index) == str:
                index = int(index)
            if type(index) != int:
                raise TypeError(
                    "The type of the set elements must be Integer or String"
                )
            if np.isin(index, np.arange(len(self._agents_data["id"])), invert=True):
                raise IndexError("Some indices are out of bounds")
            bool_array[index] = True
        return bool_array

    def get_ID_from_index(self, list_of_indices):
        """Transform a_gr list with agents indices into a_gr set of IDs
        Args:
                list_of_indices (list) : list with relevant agents indices

        Returns:
                set_of_IDs (set) : set with agents IDs, where index was in list
        """
        # Raise Exceptions
        if not (
            (isinstance(list_of_indices, set)) or (isinstance(list_of_indices, list))
        ):
            raise TypeError("The Input value must be a_gr set or list")
        bool_array = self.get_array_from_index(list_of_indices)
        for index in list_of_indices:
            if type(index) == str:
                index = int(index)
            if type(index) != int:
                raise TypeError(
                    "The type of the set elements must be Integer or String"
                )
            if np.isin(index, np.arange(len(self._agents_data["id"])), invert=True):
                raise IndexError("Some indices are out of bounds")
        set_of_ID = self.get_ID_from_array(bool_array)
        return set_of_ID

    def save_agents_data_to_file(self):
        """Export agents data."""
        abs_file_name = PM.get_path_agents_data(cs.Z_CODE, self.scale)
        if os.path.exists(abs_file_name):
            old_codes = FileLoader.load_agents_data(cs.Z_CODE, self.scale)
            assert len(np.nonzero(np.isin(old_codes, self.counties))[0]) == 0
            for key in self._agents_data:
                # assume that if file for z_code exists, the other files do exist too
                old_data = FileLoader.load_agents_data(key, self.scale)
                new_data = np.append(old_data, self._agents_data[key])
                abs_file_name = PM.get_path_agents_data(key, self.scale)
                np.save(abs_file_name, new_data)
        else:
            for key in self._agents_data:
                abs_file_name = PM.get_path_agents_data(key, self.scale)
                np.save(abs_file_name, self._agents_data[key])
