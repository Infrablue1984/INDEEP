#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Docstring."""

__author__ = "Inga Franzen"
__created__ = "2021"
__date_modified__ = "2023/05/31"
__version__ = "1.0"

import numpy as np


def remove_people_from_list(people_ids, my_list):
    """Remove agents from given list.

    :param people_ids: ids to be removed
    :param my_list: list to check and remove agents from
    :return: list with removed agents
    """
    if len(my_list) > 0:
        indices = np.nonzero(np.isin(my_list, people_ids))
        my_list = np.delete(my_list, indices)
    return my_list


def stack_dict_values_to_list_and_eliminate_dublicates(my_values):
    my_values = list(my_values.values())
    my_networks = np.vstack(my_values)
    my_networks = np.unique(my_networks, axis=0)  # make set to avoid dublicates
    return my_networks


def extract_cases(people):
    public_data = people.get_public_data()
    ids = people.get_data_for("id")
    cond1 = ids[public_data["quarantined"] is True]
    cond2 = ids[public_data["isolated"] is True]
    cond3 = ids[public_data["dead"] is True]
    quarantined = np.concatenate((cond1, cond2, cond3), axis=0)
    return quarantined


# TODO(Felix): Maybe rename ids to indices for clarification
def filter_ids_by_ages(array, dict, age_gr):
    min_age = dict["Min"][age_gr]
    max_age = dict["Max"][age_gr]
    idx = np.nonzero((array >= min_age) & (array <= max_age))[0]
    return idx
