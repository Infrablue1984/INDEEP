#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""This module is responsible for the people class.

TODO(Felix): Finish this.
"""

__author__ = "Inga Franzen"
__created__ = "2020"
__date_modified__ = "2023/05/31"
__version__ = "1.0"

from copy import deepcopy as deep
from typing import Union

import numpy as np

from synchronizer import constants as cs
from synchronizer.synchronizer import PathManager as PM


class People:
    """Class to create and manage peoples data and their relations."""

    def __init__(self, regions: list = None, scale: int = 100):
        """Initialize agents data (dict(key : ndarray)): Hold array with
        information for each agent and key. Regional dependant properties are
        specified, non-regional dependant remain default and will be specified
        later.

        Args:
            regions: Selected regions of the simulation.
            scale: Number of people represented by a single agent.
        """

        assert regions is not None, "Region must be defined"

        # dict to store agents data
        self._agents_data = {}
        self._public_data = {}
        self._epidemic_data = {}

        # get idx for chosen region
        idx = self.filter_data_by_z_code(regions, scale)

        # TODO(Felix): The below could all happen in a dedicated method (automated).
        self._agents_data["id"] = self.filter_data_by_idx(
            idx,
            np.load(PM.get_path_agents_data(cs.ID, scale)),
        )

        # extract regional code as attribute
        self._agents_data[cs.Z_CODE] = self.filter_data_by_idx(
            idx,
            np.load(PM.get_path_agents_data(cs.Z_CODE, scale)),
        )

        # extract region urban 30 or rural 60
        self._agents_data[cs.LANDSCAPE] = self.filter_data_by_idx(
            idx,
            np.load(PM.get_path_agents_data(cs.LANDSCAPE, scale)),
        )

        # create ages for
        self._agents_data[cs.AGE] = self.filter_data_by_idx(
            idx,
            np.load(PM.get_path_agents_data(cs.AGE, scale)),
        )

        # create sexes
        self._agents_data["sex"] = self.filter_data_by_idx(
            idx,
            np.load(PM.get_path_agents_data(cs.SEX, scale)),
        )

        # create occupation
        self._agents_data[cs.OCCUPATION] = self.filter_data_by_idx(
            idx,
            np.load(PM.get_path_agents_data(cs.OCCUPATION, scale)),
        )

        self._pop_size = len(self._agents_data["id"])  # int

        # TODO: Make sure that the arrays of self._agents_data all have the same length

        # initialize fields for public data
        initially_false_public_keys = [
            cs.QUARANTINED,
            cs.ISOLATED,
            cs.DIAGNOSED,
            cs.TESTED,
            cs.HOSPITALIZED,
            cs.VACCINATED,
            cs.MASK_WEARING,
            cs.DO_SOCIAL_DISTANCING,
            cs.FEEL_BAD,
        ]
        for key in initially_false_public_keys:
            self._public_data[key] = np.zeros(self._pop_size, dtype=bool)

        initially_nan_public_keys = [
            f"date_{cs.QUARANTINED}",
            f"date_{cs.ISOLATED}",
            f"date_{cs.DIAGNOSED}",
            f"date_{cs.TESTED}",
            f"date_{cs.HOSPITALIZED}",
            f"date_{cs.VACCINATED}",
            f"date_{cs.FEEL_BAD}",
            f"date_{cs.TEST_RESULTS}",
            cs.LABORATORY_TEST,
            cs.SENSITIVITY,
            cs.SPECIFICITY,
            cs.POS_TESTED,
            cs.NEG_TESTED,
        ]
        for key in initially_nan_public_keys:
            self._public_data[key] = np.full(self._pop_size, np.nan)

        # Store agents' ids together with their indices in a dictionary
        self.agents_idx = dict(zip(self._agents_data["id"], range(self._pop_size)))

    def get_pop_size(self) -> int:
        """Get the population size of the simulation.

        Returns:
            The total size of the simulation's population.
        """
        return self._pop_size

    def filter_data_by_z_code(
        self,
        regions: list,
        scale: int,
    ) -> np.ndarray:
        """Get all agents' indices of the selected regions.

        Load the z-codes of all agents in the simulation. If the
        selected region is the whole of Germany (0) then return the
        indices of all agents. Otherwise, return the indices of all
        agents that are part of the selected regions.

        Args:
            regions: Selected regions to filter agent indices for.
            scale: Number of people represented by a single agent.

        Returns:
            The indices of agents that are part of the selected regions.
        """
        # Load file containing numpy array with z-codes.
        all_data = np.load(PM.get_path_agents_data(cs.Z_CODE, scale))

        # TODO(Felix): Think about using a list for concatenation and transforming to
        #   numpy array afterwards.
        # The region '0' corresponds to the whole of Germany.
        if regions[0] == 0:
            idx = np.arange(len(all_data), dtype=int)
        else:
            idx = np.arange(0, dtype=int)
            for region in regions:
                regional_idx = np.nonzero(all_data == region)[0]
                idx = np.concatenate((idx, regional_idx), axis=0)
        return idx

    def filter_data_by_idx(self, idx: list, all_data: np.ndarray) -> np.ndarray:
        """Get data for the selected indices.

        Args:
            idx: Indices the data should be filtered for.
            all_data: Array with data to be filtered.

        Returns:
            A modified array containing only the entries corresponding
            to the selected indices.
        """
        return all_data[idx]

    def update_epidemic_data(self, epidemic_data: dict):
        """Update epidemic data of all agents.

        Args:
            epidemic_data: Epidemic data of all agents.
        """
        for key in epidemic_data:
            if len(epidemic_data[key]) != len(self._agents_data["id"]):
                raise AssertionError(
                    "All arrays must have the same length as number of people"
                )
        self._epidemic_data = epidemic_data

    def update_public_data(self, client, public_data: dict):
        """Update public data of all agents.

        Args:
            client: Object calling the method.
            public_data: Public data of all agents.
        """
        if type(client).__name__ != "PublicHealthRegulator":
            raise AssertionError(
                "Only the Public Health Regulator is allowed to enter that field."
            )
        for key in public_data:
            if len(public_data[key]) != (len(self._agents_data["id"])):
                raise AssertionError(
                    "All arrays must have the same length as number of people"
                )
        self._public_data = public_data

    def get_data_for(
        self,
        key: str,
        indices: Union[list, np.ndarray] = None,
    ) -> np.ndarray:
        """Get data for a specific key.

        Args:
            key: Specific property of agents, e.g. 'age' or 'recovered'.
            indices: Indices of agents.

        Returns:
            The requested key data in form of a np.ndarray. If no
            indices are specified the data for all agents is
            returned. Otherwise, only the key data for the
            specified agents is returned.
        """

        if indices is not None:
            if key in self._agents_data.keys():
                return self._agents_data[key][indices].copy()
            elif key in self._epidemic_data.keys():
                return self._epidemic_data[key][indices].copy()
            elif key in self._public_data.keys():
                return self._public_data[key][indices].copy()
            else:
                raise (AssertionError("key does not exist"))

        if key in self._agents_data.keys():
            return self._agents_data[key].copy()
        elif key in self._epidemic_data.keys():
            return self._epidemic_data[key].copy()
        elif key in self._public_data.keys():
            return self._public_data[key].copy()
        else:
            raise (AssertionError("key does not exist"))

    def get_agents_data(self) -> dict:
        """Get agents data for all agents.

        Returns:
            A deep copy of all agents' private data.
        """
        return deep(self._agents_data)

    def get_public_data(self) -> dict:
        """Get public data for all agents.

        Returns:
            A deep copy of all agents' public data.
        """
        return deep(self._public_data)

    def get_epidemic_data(self) -> dict:
        """Get epidemic data for all agents.

        Returns:
            A deep copy of all agents' epidemic data.
        """
        if not bool(self._epidemic_data):
            raise AssertionError("Epidemic data have not been set")
        return deep(self._epidemic_data)

    def get_ind_from_agent_id(self, agent_id: int):
        """Get agent's index from their id.

        Args:
            agent_id: Id of the associated agent.

        Returns:
            The index corresponding to the requested agent id.
        """
        assert type(agent_id) == np.int64
        return self.agents_idx[agent_id]

    def get_ind_from_multiple_agent_ids(self, agent_ids: list):
        """Get agents' indices from their ids.

        Args:
            agent_ids: Ids of the associated agents.

        Returns:
            An np.ndarray containing the corresponding indices of the
            agent ids.
        """
        assert type(agent_ids) == list
        if not agent_ids:
            return np.arange(0)
        return np.array([self.agents_idx[x] for x in agent_ids])

    def determine_number_from_percentage(self, percentage: float) -> int:
        """Determine number of agents by percentage.

        Args:
            percentage: Percentage of agents to population size.

        Returns:
            The number of agents corresponding to the given percentage.
            If the percentage is not exactly 0.0, the number of agents
            returned is at least 1.
        """
        if percentage == 0.0:
            return 0
        number_agents = round(percentage / 100 * self._pop_size)
        if number_agents == 0:
            number_agents = 1
        return number_agents
