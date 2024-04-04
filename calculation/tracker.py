#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""This module is responsible for the Tracker class.

# TODO(Felix): Finish this.
"""

import datetime as dt
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from calculation import functions
from synchronizer.synchronizer import PathManager as PM
from synchronizer.synchronizer import Synchronizer

# TODO: develop a method to track infection paths, e.g.
#   (date, network_id, source_id, target_id) --> in pd.DataFrame
#   this method could be called by epidemic spreader in any spreading loop

__author__ = "Inga Franzen, Felix Rieth"
__created__ = "2023"
__date_modified__ = "2023/05/31"
__version__ = "1.0"


class Tracker(ABC):
    """Tracks simulation data to feed into plotter."""

    @abstractmethod
    def track(self, *args, **kwargs):
        """Track simulation data.

        Args:
            args: Arguments of the track method.
            kwargs: Keyword arguments of the track method.
        """

    @staticmethod
    @abstractmethod
    def export_results(*args, **kwargs):
        """Export the results of the simulation.

        Args:
            args: Arguments of the export_results method.
            kwargs: Keyword arguments of the export_results method.
        """


class TrackResults(Tracker):
    """Tracks simulation data to feed into plotter."""

    def __init__(
        self,
        start_date: dt.date,
        number_time_steps: int,
        keys: list,
        scale: int,
    ):
        """Set the instance variables of the TrackResults object.

        Args:
            start_date: Starting date of the simulation.
            number_time_steps: Number of time steps of the simulation.
            keys: Keys to be tracked throughout the simulation.
            scale: Number of people represented by a single agent.
        """
        self._scale = scale
        self._keys = keys
        self._start_date = start_date
        self._number_time_steps = number_time_steps
        self._df_tracking = pd.DataFrame()
        self.age_cutoffs = Synchronizer.AGE_CUTOFFS

    def setup_run(self):
        """Set up the tracker for one run of the simulation.

        Create one DataFrame for the tracking of the active cases and
        another one for the tracking of the new cases in every time
        step. The DataFrames have the same structure: The columns are
        given by the keys to be tracked and the index is a multi index
        containing the dates and the age groups of the simulation. The
        two DataFrames are then concatenated and stored as an instance
        variable of the Tracker object.
        """
        steps = range(self._number_time_steps)
        dates = [self._start_date + dt.timedelta(days=step) for step in steps]
        age_groups = self.age_cutoffs["Age_gr"]
        multi_index = pd.MultiIndex.from_product(
            [dates, age_groups],
            names=["date", "age_group"],
        )
        df_active = pd.DataFrame(
            columns=self._keys,
            index=multi_index,
        )
        df_new = pd.DataFrame(
            columns=self._keys,
            index=multi_index,
        )
        self._df_tracking = pd.concat(
            [df_active, df_new],
            axis=1,
            keys=["active", "new"],
        )

    def track(self, track_data: dict, time_step: int, date_time_step: dt.datetime):
        """Track simulation data for one time step.

        Args:
            track_data: Data of the simulation.
            time_step: Time step of the simulation in days.
            date_time_step: Date of the current time step.
        """
        assert "age" in track_data
        assert (key in track_data for key in self._keys)
        for key in self._keys:
            age_groups = self.age_cutoffs["Age_gr"]
            for age_group in age_groups:
                indices_by_age_group = functions.filter_ids_by_ages(
                    track_data["age"],
                    self.age_cutoffs,
                    age_group,
                )
                self._df_tracking.at[(date_time_step, age_group), ("active", key)] = (
                    self._scale * track_data[key][indices_by_age_group].sum()
                )
                indices_by_date = np.nonzero(
                    track_data["date_{key}".format(key=key)] == time_step,
                )[0]
                indices_new = np.intersect1d(indices_by_age_group, indices_by_date)
                self._df_tracking.at[(date_time_step, age_group), ("new", key)] = (
                    self._scale * track_data[key][indices_new].sum()
                )

    def track_run(self) -> pd.DataFrame:
        """Track data for one complete run of the simulation.

        Assert that the simulation run was complete and return the
        corresponding tracked data.

        Returns:
            A DataFrame containing the tracked data for one complete
            run of the simulation.
        """
        assert self._df_tracking.shape == self._df_tracking.dropna().shape
        return self._df_tracking

    # TODO(Felix): In Args: Why is it called dict, but says 'stored in a pd.DataFrame'?
    @staticmethod
    def export_results(result_dict: dict):
        """Process the track data for all runs from result_dict.

        Calculate mean, minimal and maximal values for all runs of the
        simulation and create a DataFrame for each of those. Finally,
        concatenate all three DataFrames and store them as the result
        of the whole simulation. Export the results of the simulation
        to a csv file.

        Args:
            result_dict: Contain results from each run stored in a pd.Dataframe.
        """

        df_track_data_all_runs = pd.concat(
            result_dict.values(),
            axis=1,
            keys=result_dict.keys(),
        )
        df_mean = df_track_data_all_runs.groupby(axis=1, level=[1, 2]).mean()
        df_min = df_track_data_all_runs.groupby(axis=1, level=[1, 2]).min()
        df_max = df_track_data_all_runs.groupby(axis=1, level=[1, 2]).max()
        df_results = pd.concat(
            [df_min, df_mean, df_max],
            axis=1,
            keys=["min", "mean", "max"],
        )

        result_path = PM.get_path_results()
        df_results.to_csv(result_path)
