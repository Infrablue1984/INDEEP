#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Manage path names and global conventions, load and save data.

Translate int Code to str name and vice versa. Provide name of global
file path. Load and save files providing right format.
"""

__author__ = "Inga Franzen"
__created__ = "2022/09/12"
__date_modified__ = "2022/09/30"
__version__ = "1.0"

import math
import os
import pathlib
import warnings
from math import isnan
from os.path import exists

import numpy as np
import pandas as pd
from pandas import ExcelWriter

from synchronizer import constants as cs
from synchronizer.config import Config


class Synchronizer:
    """Synchronize path and file names.

    Offer save and open file functions and supply data transfer.
    """

    # TODO: Should this also be moved outside the class
    AGE_CUTOFFS = {
        cs.AGE_GR: np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        cs.MIN: np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]),
        cs.MAX: np.array([9, 19, 29, 39, 49, 59, 69, 79, 89, 99]),
    }

    PUBLIC_DIGITS = [2, 3, 4, 5, 6]
    BUILDING_DIGITS = [1, 2, 3, 4, 5, 6]

    DIGIT_TRANSLATE = {
        1: cs.HOUSEHOLDS,
        2: cs.WORKPLACES,
        3: cs.KITAS,
        4: cs.SCHOOLS,
        5: cs.UNIVERSITIES,
        6: cs.ACTIVITIES,
        7: cs.GEOGRAPHIC,
    }

    DIGIT_TRANSLATE_2 = dict((zip(DIGIT_TRANSLATE.values(), DIGIT_TRANSLATE.keys())))

    SUB_NET_TRANSLATE = {
        cs.OFFICE: 51,
        cs.MEETING: 52,
        cs.CLASS: 53,
        cs.LECTURE: 54,
        cs.FREE: 55,
        cs.MENSA: 56,
        cs.UNSPECIFIC: 57,
        cs.ACTIVITY: 58,
    }

    SUB_NET_TRANSLATE_2 = dict(
        zip(SUB_NET_TRANSLATE.values(), SUB_NET_TRANSLATE.keys())
    )

    SUB_NET_COLUMNS = {
        cs.PARENT_ID: 0,
        cs.NET_ID: 1,
        cs.TYPE: 2,
        cs.SIZE: 3,
        cs.MEAN_CONTACTS: 4,
        cs.YEAR: 5,
        cs.WEEKDAY: 6,
        cs.LANDSCAPE: 7,
    }

    NETWORK_COLUMNS = {cs.ID: 0, cs.TYPE: 1, cs.SIZE: 2}

    LOCATION_TYPE_INFORMATION = {
        cs.OFFICE: {
            cs.ROOM_VOLUME: 300,
            cs.AIR_EXCHANGE_RATE: 0.5,
            cs.INHALATION_RATE: 0.54,
            cs.CONTACT_HOURS: 8,
            cs.MEAN_CONTACTS: 2,
        },
        cs.MEETING: {
            cs.ROOM_VOLUME: 300,
            cs.AIR_EXCHANGE_RATE: 0.5,
            cs.INHALATION_RATE: 0.54,
            cs.CONTACT_HOURS: 2,
            cs.MEAN_CONTACTS: 2,
        },
        cs.CLASS: {
            cs.ROOM_VOLUME: 300,
            cs.AIR_EXCHANGE_RATE: 0.5,
            cs.INHALATION_RATE: 0.54,
            cs.CONTACT_HOURS: 6,
            cs.MEAN_CONTACTS: 3,
        },
        cs.LECTURE: {
            cs.ROOM_VOLUME: 800,
            cs.AIR_EXCHANGE_RATE: 0.5,
            cs.INHALATION_RATE: 0.54,
            cs.CONTACT_HOURS: 1.5,
            cs.MEAN_CONTACTS: 2,
        },
        cs.FREE: {
            cs.ROOM_VOLUME: 400,
            cs.AIR_EXCHANGE_RATE: 0.5,
            cs.INHALATION_RATE: 0.54,
            cs.CONTACT_HOURS: 1,
            cs.MEAN_CONTACTS: 3,
        },
        cs.MENSA: {
            cs.ROOM_VOLUME: 5000,
            cs.AIR_EXCHANGE_RATE: 0.5,
            cs.INHALATION_RATE: 0.54,
            cs.CONTACT_HOURS: 0.75,
            cs.MEAN_CONTACTS: 3,
        },
        cs.UNSPECIFIC: {
            cs.ROOM_VOLUME: -999,
            cs.AIR_EXCHANGE_RATE: 0.5,
            cs.INHALATION_RATE: 0.54,
            cs.CONTACT_HOURS: 0.25,
            cs.MEAN_CONTACTS: 1,
        },
        cs.GEOGRAPHIC: {
            cs.ROOM_VOLUME: -999,
            cs.AIR_EXCHANGE_RATE: 0.5,
            cs.INHALATION_RATE: 0.54,
            cs.CONTACT_HOURS: 0.25,
            cs.MEAN_CONTACTS: 3,
        },
        cs.ACTIVITY: {
            cs.ROOM_VOLUME: 400,
            cs.AIR_EXCHANGE_RATE: 0.5,
            cs.INHALATION_RATE: 1.38,
            cs.CONTACT_HOURS: 2,
            cs.MEAN_CONTACTS: -999,
        },
        cs.LIVING_ROOM: {
            cs.ROOM_VOLUME: 175,
            cs.AIR_EXCHANGE_RATE: 0.5,
            cs.INHALATION_RATE: 0.54,
            cs.CONTACT_HOURS: 4,
            cs.MEAN_CONTACTS: -999,
        },
    }

    INTERVENTION_USER_INPUT = {
        cs.TEST_INFO: {
            cs.LABORATORY_TEST: {cs.SENSITIVITY: 99, cs.SPECIFICITY: 99},
            cs.RAPID_TEST: {cs.SENSITIVITY: 80, cs.SPECIFICITY: 97},
            cs.PERCENTAGE_RAPID_TEST: 0.5,
        },
        cs.QUARANTINED: {cs.PERIOD: 10},
        cs.MANUAL_CONTACT_TRACING: {cs.TIME_DELAY: 3, cs.PERIOD: 7},
        cs.DIAGNOSED: {cs.PERIOD: 10, cs.PERCENTAGE_MILD_FEEL_BAD: 0.50},
    }

    @staticmethod
    def help_filter_network_ids_by_digit(my_nets, digit):
        """Filter net_ids starting with digit.

        Args:
            my_nets (list[int]): list with net_ids
            digit (int): digit standing for network type.

        Returns:
            new_list (list[int]): list with filtered net_ids.
        """
        new_list = []
        for net_id in my_nets:
            my_digit = int(str(net_id)[0])
            if digit == my_digit:
                new_list.append(net_id)
        return new_list

    @staticmethod
    def get_region_from_agent_id(agent_id):
        return int(agent_id / 10**5)

    @staticmethod
    def get_region_from_net_id(net_id):
        region = int(str(net_id)[1:6])
        if int(region / 1000) == 2:
            return 2000
        elif int(region / 1000) == 11:
            return 11000
        else:
            return region

    @staticmethod
    def get_first_digit(net_id):
        return int(str(net_id)[0])

    @staticmethod
    def get_network_type_from_digit(digit):
        return Synchronizer.DIGIT_TRANSLATE[digit]

    @staticmethod
    def type_to_code(local_type):
        return Synchronizer.SUB_NET_TRANSLATE[local_type][cs.CODE]

    @staticmethod
    def code_to_type(code):
        return Synchronizer.SUB_NET_TRANSLATE_2[code]

    @staticmethod
    def help_unpack_numbers_from_string(string):
        my_str_list = string.strip("[]").split(" ")
        my_int_list = [int(x) for x in my_str_list]
        my_int_list.sort()
        return my_int_list

    @staticmethod
    def extract_weekdays(data):
        if type(data) == str:
            weekdays = Synchronizer.help_unpack_numbers_from_string(data)
        else:
            weekdays = [0, 1, 2, 3, 4]
        return weekdays


class PathManager:
    @staticmethod
    def get_path_members_file(net_type, scale):
        file_name = f"{net_type}_members_{scale}_subnet.npy"
        abs_file_name = os.path.join(Config.NETWORK_FILE_DIR, file_name)
        return abs_file_name

    @staticmethod
    def get_path_agents_networks(scale):
        # TODO: integrate scale into name
        file_name = f"agents_networks_{scale}.npy"
        return os.path.join(Config.NETWORK_FILE_DIR, file_name)

    @staticmethod
    def get_path_info_file(scale):
        return os.path.join(Config.INFO_FILE_DIR, f"info_file_{scale}.npy")

    @staticmethod
    def get_path_scenario_file():
        return os.path.join(Config.SCENARIO_FILE_DIR, "COVID_default.csv")

    @staticmethod
    def get_path_agents_data(key, scale):
        file_name = f"agents_{key}_{scale}.npy"
        return os.path.join(Config.AGENT_FILE_DIR, file_name)

    @staticmethod
    def get_path_parent_table(net_type, scale):
        file_name = f"{net_type}_{scale}.csv"
        abs_file_name = os.path.join(Config.NETWORK_FILE_DIR, file_name)
        return abs_file_name

    @staticmethod
    def get_path_sub_table(net_type, scale):
        file_name = f"{net_type}_{scale}_subnet.csv"
        abs_file_name = os.path.join(Config.NETWORK_FILE_DIR, file_name)
        return abs_file_name

    @staticmethod
    def get_path_bokeh_server():
        abs_path = os.path.join(
            Config.BUNDLE_DIR, Config.AGENT_MODEL_DIR, "bokeh_server.py"
        )
        abs_path = abs_path.replace(" ", r"\ ")  # remove empty spaces
        return abs_path

    @staticmethod
    def get_path_results():
        return os.path.join(Config.OUTPUT_FILE_DIR, "data.csv")

    @staticmethod
    def get_path_current_mean_contacts(scale):
        file_name = f"mean_contacts_current_{scale}.csv"
        return os.path.join(Config.NETWORK_FILE_DIR, file_name)

    @staticmethod
    def get_path_target_mean_contacts():
        return os.path.join(Config.REGPOP_FILE_DIR, "mean_contacts.xlsx")

    @staticmethod
    def get_path_interventions():
        test = os.path.join(Config.REGPOP_FILE_DIR, "interventions.xls")
        return test

    @staticmethod
    def get_path_counties():
        return os.path.join(Config.REGPOP_FILE_DIR, "pop_counties.xlsx")

    @staticmethod
    def get_path_household_table():
        return os.path.join(Config.REGPOP_FILE_DIR, "households.xlsx")

    @staticmethod
    def get_path_household_age_table():
        return os.path.join(Config.REGPOP_FILE_DIR, "households_ages.xlsx")

    @staticmethod
    def get_path_school_table():
        return os.path.join(Config.REGPOP_FILE_DIR, "meta_school.xlsx")

    @staticmethod
    def get_path_work_table():
        return os.path.join(Config.REGPOP_FILE_DIR, "meta_work.xlsx")

    @staticmethod
    def get_path_states():
        return os.path.join(Config.REGPOP_FILE_DIR, "pop_states.xlsx")

    @staticmethod
    def get_path_age_dist():
        return os.path.join(Config.REGPOP_FILE_DIR, "age_distribution.xlsx")

    @staticmethod
    def get_path_gender_dist():
        return os.path.join(Config.REGPOP_FILE_DIR, "gender_distribution.xlsx")

    @staticmethod
    def get_path_occupation_dist():
        return os.path.join(Config.REGPOP_FILE_DIR, "occupations.xlsx")


class FileLoader:
    @staticmethod
    def load_dict_from_npy(abs_file_name):
        my_load = np.load(abs_file_name, allow_pickle=True)
        my_dict = dict(enumerate(my_load.flatten(), 1))  # numpy array to dict
        my_dict = my_dict[1]  # numpy array to dict
        return my_dict

    @staticmethod
    def read_excel_table(abs_file_name, sheet_name=0, dtype=None):
        # read original file
        # TODO: path is abs_file_name already in load files
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("always", category=UserWarning, module="openpyxl")
            # warnings.simplefilter("ignore")
            df = pd.read_excel(
                abs_file_name, dtype=dtype, sheet_name=sheet_name, na_values=["NA"]
            )
        return df

    @staticmethod
    def load_network_table(net_type, scale=100, sub=False):
        assert type(net_type) == str
        if sub:
            abs_file_name = PathManager.get_path_sub_table(net_type, scale)
        else:
            abs_file_name = PathManager.get_path_parent_table(net_type, scale)
        df_data = pd.read_csv(abs_file_name, index_col=0)
        return df_data

    @staticmethod
    def load_network_data(digit, scale):
        net_type = Synchronizer.get_network_type_from_digit(digit)
        df_data = FileLoader.load_network_table(net_type, scale)
        df_sub = FileLoader.load_network_table(net_type, scale, sub=True)
        df_all = df_sub.join(df_data.set_index("id"), on="parent id", rsuffix=" parent")
        df_all["region"] = [
            Synchronizer.get_region_from_net_id(x) for x in df_all["id"]
        ]
        return df_all

    @staticmethod
    def load_members(digit, scale):
        # TODO: add scale
        net_type = Synchronizer.get_network_type_from_digit(digit)
        abs_file_name = PathManager.get_path_members_file(net_type, scale)
        members_groups = FileLoader.load_dict_from_npy(abs_file_name)
        return members_groups

    @staticmethod
    def load_agents_networks(regions, scale):
        abs_file_name = PathManager.get_path_agents_networks(scale)
        agents_nets = FileLoader.load_dict_from_npy(abs_file_name)
        agents_nets = dict(
            filter(
                lambda x: Synchronizer.get_region_from_agent_id(x[0]) in regions,
                agents_nets.items(),
            )
        )
        return agents_nets

    @staticmethod
    def load_agents_data(attribute, scale):
        abs_file_name = PathManager.get_path_agents_data(attribute, scale)
        return np.load(abs_file_name, allow_pickle=True)


class FileSaver:
    @staticmethod
    def save_csv_table(abs_file_name, df):
        df.to_csv(abs_file_name, index=True)

    @staticmethod
    def save_excel_table(path, file, df, sheet_name=0):
        abs_file_name = os.path.join(path, file)
        if exists(abs_file_name):
            with ExcelWriter(
                abs_file_name,
                mode="a",
                engine="openpyxl",
                if_sheet_exists="replace",
            ) as writer:
                df.to_excel(writer, sheet_name)
        else:
            writer = ExcelWriter(abs_file_name, engine="openpyxl")
            df.to_excel(writer, sheet_name)
            writer.close()

    @staticmethod
    def save_dict_to_npy(abs_file_name, my_dict):
        np.save(abs_file_name, my_dict)
