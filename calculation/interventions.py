#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Create and handle different types of interventions.

Each intervention is some kind of calender entry holding all the
information about itself. Such as start and end date, a value of
compliance or the information, if it has changed in the last time step.
"""

__author__ = "Inga Franzen"
__created__ = "2020"
__date_modified__ = "2023/05/31"
__version__ = "1.0"

import datetime
import datetime as dt
import pathlib

import numpy as np
import pandas as pd

from synchronizer import constants as cs
from synchronizer.synchronizer import FileLoader, Synchronizer

# TODO: check meaning of test symptomatic, primary contacts \
#  --> can/should I test them again every day?


class Intervention:
    def __init__(
        self,
        name: str = None,
        start_date: datetime.date = None,
        end_date: datetime.date = None,
    ):
        """Initialize super class intervention.

        Args:
            name: Name of the intervention.
            start_date: Start date of the intervention.
            end_date: End date of the intervention.
        """
        if (start_date is None) or (end_date is None):
            raise AssertionError("start- and end date must be defined")
        if name is None:
            raise AssertionError("intervention must be defined")
        if not isinstance(name, str):
            # TODO(Felix): Maybe: "intervention name must be of type string"?
            raise AssertionError("intervention input parameter must be a_gr string")
        self._activated = False
        self._status_changed = False
        self._start_date = start_date
        self._end_date = end_date
        self._name = name
        self._one_time = None
        self._organizer = None
        self._set_attributes()

    def is_activated(self):
        """Return activation status of intervention."""
        return self._activated

    def status_activated_changed(self):
        """Return whether activation status of intervention was changed."""
        return self._status_changed

    def update_status(self, date_of_time_step: datetime.date):
        """Update activation status of intervention.

        Compare the input date of the time step with the start and end date of the
        intervention and set the activation status of the intervention accordingly.

        Args:
            date_of_time_step: Date of the time step of the simulation.
        """
        if self._start_date == self._end_date:
            return
        if pd.Timestamp(date_of_time_step) == pd.Timestamp(self._start_date):
            self._activated = True
            self._status_changed = True
        elif pd.Timestamp(date_of_time_step) == pd.Timestamp(self._end_date):
            self._activated = False
            self._status_changed = True
        else:
            self._status_changed = False

    def get_type(self):
        """Return name of subclass.

        TODO(Felix): Intervention Type?
        """
        return type(self).__name__

    def get_name(self):
        """Return name of intervention."""
        return self._name

    def get_start_date(self):
        """Return start date of intervention."""
        return self._start_date

    def get_end_date(self):
        """Return end date of intervention."""
        return self._end_date

    def get_organizer(self):
        """Return class responsible for the implementation of the
        intervention."""
        return self.organizer

    def get_one_time(self):
        """Return whether the intervention is of type 'one time'."""
        return self._one_time

    def _set_attributes(self):
        """Set attributes according to intervention name."""
        if (
            (cs.LIMIT_EVENT_SIZE in self._name)
            or (cs.SPLIT_GROUPS in self._name)
            or (cs.HOME_DOING in self._name)
            or (cs.SHUT_INSTITUTIONS in self._name)
        ):
            self.organizer = cs.NETWORK_MANAGER
            self._one_time = True
        elif (
            (self._name == cs.TESTING)
            or (self._name == cs.MANUAL_CONTACT_TRACING)
            or (self._name == cs.CLUSTER_ISOLATION)
        ):
            self.organizer = cs.PUBLIC_REGULATOR
            self._one_time = False
        elif self._name == cs.SELF_ISOLATION:
            self.organizer = cs.PUBLIC_REGULATOR
            self._one_time = True

        elif self._name in (cs.SOCIAL_DISTANCING, cs.MASK):
            self.organizer = cs.EPIDEMIC_SPREADER
            self._one_time = True
        else:
            raise Exception(f"The name of the intervention {self._name} does not exist")

    def reset_data(self):
        """Reset activation status data of intervention."""
        self._activated = False
        self._status_changed = False


class InterventionType1(Intervention):
    def __init__(self, name=None, percentage=None, start_date=None, end_date=None):
        """Initialize intervention from type one.

        Type1 classes specify percentages to what extend an intervention is implemented.

        Args:
            percentage (int) : percentage of implementation (dependant on governmental decision and public commitment)
            start_date(date) : start date of intervention
            end_date(date) : end date of intervention
        """
        assert percentage is not None, "percentage must be defined"
        assert isinstance(
            percentage, int
        ), "percentage must be a_gr number between 0 and 100"
        assert percentage <= 100, "percentage cant be higher than hundred"

        super(InterventionType1, self).__init__(name, start_date, end_date)
        self._value = percentage

    def get_value(self):
        """Getter for private field.

        Returns: percentage(int) : percentage of use
        """
        return self._value


class InterventionType2(InterventionType1):
    """Class to supply details for test strategy."""

    def __init__(
        self,
        name=None,
        value=None,  # TODO: is this default value necessary --> gives error in line 260
        target_group=None,
        intervall=None,  # TODO: was "=1" before, is that important? Was leading to error message in line 103
        start_date=None,
        end_date=None,
    ):
        """Initialize intervention."""
        # assert (target_group != None), "target group must be defined"
        # assert (isinstance(target_group, list)), "there must be list with target group"
        super(InterventionType2, self).__init__(name, int(value), start_date, end_date)
        self._target_group = target_group
        self._intervall = intervall

    def get_target_group(self):
        return self._target_group

    def get_intervall(self):
        return self._intervall


class InterventionMaker:

    INTERVENTION_LIST = [
        cs.LIMIT_EVENT_SIZE,
        cs.LIMIT_EVENT_SIZE_WORKPLACES,
        cs.LIMIT_EVENT_SIZE_KITAS,
        cs.LIMIT_EVENT_SIZE_SCHOOLS,
        cs.LIMIT_EVENT_SIZE_UNIVERSITIES,
        cs.LIMIT_EVENT_SIZE_ACTIVITIES,
        cs.SPLIT_GROUPS,
        cs.SPLIT_GROUPS_WORKPLACES,
        cs.SPLIT_GROUPS_KITAS,
        cs.SPLIT_GROUPS_SCHOOLS,
        cs.SPLIT_GROUPS_UNIVERSITIES,
        cs.SPLIT_GROUPS_ACTIVITIES,
        cs.HOME_DOING,
        cs.HOME_DOING_WORKPLACES,
        cs.HOME_DOING_KITAS,
        cs.HOME_DOING_SCHOOLS,
        cs.HOME_DOING_UNIVERSITIES,
        cs.HOME_DOING_ACTIVITIES,
        cs.SHUT_INSTITUTIONS,
        cs.SHUT_WORKPLACES,
        cs.SHUT_SCHOOLS,
        cs.SHUT_KITAS,
        cs.SHUT_UNIVERSITIES,
        cs.SHUT_ACTIVITIES,
        cs.SOCIAL_DISTANCING,
        cs.MASK,
        cs.CLUSTER_ISOLATION,
        cs.MANUAL_CONTACT_TRACING,
        cs.SELF_ISOLATION,
        cs.TESTING,
    ]

    @staticmethod
    def initialize_interventions(abs_file_name):
        """Transform the name of intervention from data sheet into python
        objects.

        Args:
            intervention_data(xls) : excel data sheet with names of simple intervention_data
                                and values (e.g. % of implementation).

        Returns:
            intervention_data(list) : List with references to intervention objects.
        """
        interventions_type_1 = InterventionMaker.create_interventions_type_1(
            abs_file_name
        )
        interventions_type_2 = InterventionMaker.create_interventions_type_2(
            abs_file_name
        )
        interventions = interventions_type_1 + interventions_type_2

        return interventions

    @staticmethod
    def create_interventions_type_1(abs_file_name):
        """Create intervention_data of type 1.

        Transform the name of intervention from data sheet into python objects.

        Args:
            intervention_data(xls) : excel data sheet with names of simple intervention_data
                                and values (e.g. % of implementation).

        Returns:
            intervention_data(list) : List with references to intervention objects.
        """
        # create list for intervention objects
        interventions_type_1 = []
        # read data for intervention_data type 1 (with values)
        df_type_1 = FileLoader.read_excel_table(abs_file_name, "Type1")
        InterventionMaker.clear_data(df_type_1)

        # create all other intervention_data of type 1
        for line in range(len(df_type_1)):
            df_type_1.index = list(range(len(df_type_1)))
            name = df_type_1["intervention"][line]
            if name not in InterventionMaker.INTERVENTION_LIST:
                print("Valid names for interventions are:")
                [print(i) for i in InterventionMaker.INTERVENTION_LIST]
                raise AssertionError(f"The name '{name}' is not a valid.")
            start_date = df_type_1["start date"][line]
            end_date = df_type_1["end date"][line]
            value = int(df_type_1["value"][line])
            intervention = InterventionType1(name, value, start_date, end_date)
            interventions_type_1.append(intervention)

        return interventions_type_1

    @staticmethod
    def create_interventions_type_2(abs_file_name):
        """Create intervention_data of type 2.

        Transform the name of intervention from data sheet into python objects.

        Args:
            intervention_data(xls) : excel data sheet with names of simple intervention_data
                                and values (e.g. % of implementation).

        Returns:
            intervention_data(list) : List with references to intervention objects.
        """
        interventions_type_2 = []
        # read data for intervention_data type 2
        df_type_2 = FileLoader.read_excel_table(abs_file_name, "Type2")
        InterventionMaker.clear_data(df_type_2)
        # create all intervention_data of type 2
        for line in df_type_2.index:
            name = df_type_2["intervention"][line]
            if name not in InterventionMaker.INTERVENTION_LIST:
                raise AssertionError(
                    f"The name '{name}' is not a valid. Valid names for interventions"
                    f" are: {InterventionMaker.INTERVENTION_LIST}"
                )
            value = df_type_2["value"][line]
            intervall = df_type_2["intervall"][line]
            start_date = df_type_2["start date"][line]
            end_date = df_type_2["end date"][line]
            target_group = df_type_2["target group 1"][line]
            # entfernt im Bedarfsfall ungewünschte Tabulaturen
            target_group = target_group.strip("\t\r\n")
            intervention = InterventionType2(
                name, value, target_group, intervall, start_date, end_date
            )
            interventions_type_2.append(intervention)

        return interventions_type_2

    @staticmethod
    def clear_data(inter_sheet):
        """Clear non data and transform name of intervention into camel case.

        Args:
            inter_sheet (DataFrame): dataframe holding information on intervention_data
        """
        inter_sheet.replace([0, False], np.nan, inplace=True)  # replace False with zero
        inter_sheet.dropna(
            axis=0, how="any", subset=["On"], inplace=True
        )  # drop interventions that are off
        inter_sheet.dropna(
            axis=0, how="any", subset=["value"], inplace=True
        )  # drop zero value

    @staticmethod
    def pre_check_intervention_user_input(intervention_list):
        InterventionMaker.check_dates_for_same_name_interventions(intervention_list)
        InterventionMaker.check_dates_for_grouped_interventions(intervention_list)

    @staticmethod
    def check_dates_for_same_name_interventions(intervention_list):
        sorted_intervention_list = InterventionMaker.sort_intervention_list_by_name(
            intervention_list
        )
        for idx, intervention in enumerate(sorted_intervention_list):
            if intervention.get_start_date() > intervention.get_end_date():
                raise BaseException(
                    f"{cs.LANGUAGE_DICT[cs.CHECK_INTERVENTION_EXCEPTION][cs.ENGLISH]} "
                    f"{cs.LANGUAGE_DICT[cs.START_END_DATE_EXCEPTION][cs.ENGLISH]}"
                )
            if idx == len(sorted_intervention_list) - 1:
                break
            next_idx = idx + 1
            next_intervention = sorted_intervention_list[next_idx]
            # check dates for same interventions
            while has_same_name(
                intervention, next_intervention
            ) and is_same_specific_type(intervention, next_intervention):
                if has_overlaps_in_date(intervention, next_intervention):
                    raise BaseException(
                        f"{cs.LANGUAGE_DICT[cs.CHECK_INTERVENTION_EXCEPTION][cs.ENGLISH]}\n "
                        f"{cs.LANGUAGE_DICT[cs.INTERVENTION_DATE_OVERLAP_EXCEPTION][cs.ENGLISH]}"
                    )
                next_idx += 1
                next_intervention = sorted_intervention_list[next_idx]

    @staticmethod
    def check_dates_for_grouped_interventions(intervention_list):
        # check dates for same super and specific interventions
        grouped_intervention_names = [
            cs.HOME_DOING,
            cs.SHUT_INSTITUTIONS,
            cs.SPLIT_GROUPS,
            cs.LIMIT_EVENT_SIZE,
        ]
        for group_name in grouped_intervention_names:
            intervention_group = [
                x for x in intervention_list if group_name in x.get_name()
            ]
            for intervention in intervention_group:
                if not intervention.get_name().split(group_name)[1].strip():
                    for to_compare_intervention in intervention_group:
                        if not has_same_name(intervention, to_compare_intervention):
                            if has_overlaps_in_date(
                                intervention, to_compare_intervention
                            ):
                                raise BaseException(
                                    f"{cs.LANGUAGE_DICT[cs.CHECK_INTERVENTION_EXCEPTION][cs.ENGLISH]}\n "
                                    f"{cs.LANGUAGE_DICT[cs.INTERVENTION_SUPER_SPECIFIC_OVERLAP_EXCEPTION][cs.ENGLISH]}"
                                )

    @staticmethod
    def sort_intervention_list_by_name(intervention_list):
        sorted_intervention_list = sorted(
            intervention_list, key=lambda intervention: intervention.get_name()
        )
        return sorted_intervention_list

    @staticmethod
    def sort_intervention_list_by_date(intervention_list):
        sorted_intervention_list = sorted(
            intervention_list, key=lambda intervention: intervention.get_start_date()
        )
        return sorted_intervention_list

    @staticmethod
    def intervention_needs_tracing(intervention):
        """Check if intervention needs trace of network status.

        Tracing is time-consuming and will only be done if necessary.

        Args:
            intervention (ClassIntervention): Pointer to intervention object.
        """
        my_name = intervention.get_name()
        tracing_needed = (
            (my_name == cs.MANUAL_CONTACT_TRACING)
            or (my_name == cs.CLUSTER_ISOLATION)
            or (
                (my_name == cs.TESTING)
                and (intervention.get_target_group() == "primary contact")
            )
        )
        return tracing_needed


def has_overlaps_in_date(intervention, next_intervention):
    has_overlaps = (
        (  # Case 1: Both interventions overlap each other
            (intervention.get_start_date() < next_intervention.get_end_date())
            and (intervention.get_end_date() > next_intervention.get_start_date())
        )
        or (  # Case 2: The first intervention overlaps the second.
            (intervention.get_start_date() < next_intervention.get_start_date())
            and (intervention.get_end_date() > next_intervention.get_end_date())
        )
        or (  # Case 3: The second intervention overlaps the first.
            (intervention.get_start_date() > next_intervention.get_start_date())
            and (intervention.get_end_date() < next_intervention.get_end_date())
        )  # Case 4: Both interventions have the same start date.
        or (intervention.get_start_date() == next_intervention.get_start_date())
    )
    return has_overlaps


def is_same_specific_type(intervention, intervention_next):
    if type(intervention).__name__ == "InterventionType2":
        cond_same_specific_type = (
            intervention.get_target_group() == intervention_next.get_target_group()
        )
    else:
        cond_same_specific_type = True
    return cond_same_specific_type


def has_same_name(intervention, intervention_next):
    cond_same_intervention = intervention.get_name() == intervention_next.get_name()
    return cond_same_intervention
