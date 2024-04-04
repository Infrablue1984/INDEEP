#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""This module is responsible for the EpidemicSpreader class.

TODO: Finish this

The transitions in this method are based on the Covasim model, cf.
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009149#pcbi-1009149-g003
Overall description.
"""

__author__ = "Felix Rieth"
__created__ = "2023"
__date_modified__ = "2023/05/31"
__version__ = "1.0"

import math
import statistics
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
# from line_profiler_pycharm import profile
from scipy import integrate
# from tqdm import tqdm

from calculation.network_manager import NetworkManager
from calculation.people import People
from synchronizer import constants as cs
from synchronizer.synchronizer import Synchronizer as Sync

# TODO:(Jan) think about: new class epidemic_data(dict)
#   => e.g. self.enter_newstate(epedemic_data) ->self.epdemic_data.enter_new_state()
#   => better readable, that dict will be changed in place?


class EpidemicSpreader(ABC):
    """Spreads the disease in the population.

    The EpidemicSpreader communicates with the People and NetworkManager
    objects of the simulation. In the beginning of the simulation an
    EpidemicSpreader object is created and initialized. During the
    simulation, the EpidemicSpreader is responsible for infecting new
    agents and modelling the disease course of infected agents.
    """

    # TODO: Think about using more "raise Error"
    # TODO: Think about converting methods to functions (check "method" comments)

    def __init__(
        self,
        people: People,
        network_manager: NetworkManager,
        epidemic_user_input: pd.DataFrame,
        seed: np.random.Generator,
    ):
        """Set the instance variables of the EpidemicSpreader.

        To communicate with the People and NetworkManager objects of the
        simulation, set these as instance variables of the
        EpidemicSpreader. Furthermore, set the epidemic user input as
        instance variable, as most instance methods need to have access
        to it.

        Args:
            people: People object of the simulation.
            network_manager: NetworkManager object of the simulation.
            epidemic_user_input: User input to specify the disease scenario.
            seed: Random number generator with specified seed.
        """
        self.seed = seed
        self.people = people
        self.network_manager = network_manager
        self.DURATION_PARAMETERS = self._generate_duration_parameters(
            epidemic_user_input,
        )
        self.EPIDEMIC_PARAMETERS = self._generate_epidemic_parameters(
            epidemic_user_input,
        )
        self._initialize_epidemic_data(epidemic_user_input)

    @abstractmethod
    def infect_agents(self, time_step: int):
        """Change the epidemic status of agents within a time step.

        This method has to be implemented by the subclasses of the
        EpidemicSpreader class.

        Args:
            time_step: Time step of the simulation in days.
        """
        pass

    def _generate_duration_parameters(self, epidemic_user_input: pd.DataFrame) -> dict:
        """Get durations for all possible status transitions.

        Read the epidemic user input and generate a dict containing the
        duration parameters for every possible status transition of the
        disease course. The parameters for a transition are kept in a
        dictionary and consist of the mean and the standard deviation
        of the log-normal duration distribution.

        Args:
            epidemic_user_input: User input to specify the disease scenario.

        Returns:
            A dictionary containing the duration parameters for all
            possible disease course status transitions.
        """
        ################################################################################
        # The possible disease course transitions of this model are **NOT** meant to be
        # changed and therefore hard coded. The order of the transitions in the duration
        # parameters is also important to allow for agents to do more then one status
        # transition per time step.
        ################################################################################
        duration_parameters = {}
        transition_pairs = [
            [cs.SUSCEPTIBLE, cs.EXPOSED],
            [cs.EXPOSED, cs.ASYMPTOMATIC],
            [cs.EXPOSED, cs.PRESYMPTOMATIC],
            [cs.ASYMPTOMATIC, cs.RECOVERED],
            [cs.PRESYMPTOMATIC, cs.MILD],
            [cs.MILD, cs.SEVERE],
            [cs.SEVERE, cs.CRITICAL],
            [cs.CRITICAL, cs.DEAD],
            [cs.MILD, cs.RECOVERED],
            [cs.SEVERE, cs.RECOVERED],
            [cs.CRITICAL, cs.RECOVERED],
            # TODO: Possible to add the transition pair [cs.RECOVERED, cs.SUSCEPTIBLE]
            #       1.) if duration parameters for this transition are added in the
            #           epidemic user input and
            #       2.) if cs.SUSCEPTIBLE is added as the next status after cs.RECOVERED
            #           in the method _get_possible_next_statuses()
        ]
        for pair in transition_pairs:
            transition = f"{pair[0]}_to_{pair[1]}"
            if pair[0] == cs.SUSCEPTIBLE:
                duration_parameters[transition] = {}
                continue
            duration_parameters[transition] = {
                "mean": epidemic_user_input.loc[
                    f"{transition}_mean",
                    cs.VALUE,
                ],
                "std": epidemic_user_input.loc[
                    f"{transition}_std",
                    cs.VALUE,
                ],
            }
        return duration_parameters

    def _generate_epidemic_parameters(self, epidemic_user_input: pd.DataFrame) -> dict:
        """Filter the epidemic user input.

        Args:
            epidemic_user_input: User input to specify the disease scenario.

        Returns:
            A dictionary containing the parameters of the epidemic
            user input, that are necessary during the simulation.
        """
        epidemic_user_input = epidemic_user_input.loc[
            epidemic_user_input[cs.TYPE] == "epidemics",
        ]
        return epidemic_user_input.drop(cs.TYPE, axis=1).to_dict()[cs.VALUE]

    def _initialize_epidemic_data(self, epidemic_user_input: pd.DataFrame):
        """Initialize the epidemic status of all agents.

        Create a dictionary holding all the epidemically relevant
        information of the agents and set initial values for each agent
        in form of an array as specified by the epidemic user input.

        Args:
            epidemic_user_input: User input to specify the disease scenario.
        """
        population_size = self.people.get_pop_size()
        epidemic_data = {}  # Dict will be filled inside the subfunctions.
        self._set_status_fields(epidemic_data, population_size)
        self._set_probability_fields(
            epidemic_data,
            epidemic_user_input,
            population_size,
        )
        self._set_date_fields(epidemic_data, population_size)
        self._set_initially_infected(epidemic_data, epidemic_user_input)
        self._set_initially_recovered(epidemic_data, epidemic_user_input)
        initially_infected_indices = np.nonzero(epidemic_data[cs.INFECTED])[0]
        self._determine_status_of_initially_infected(
            epidemic_data,
            initially_infected_indices,
        )
        self._set_infection_transmission_fields(epidemic_data, epidemic_user_input)
        self._initialize_infection_evolution(
            epidemic_data,
            initially_infected_indices,
        )
        self.people.update_epidemic_data(epidemic_data)

    ######################################################
    # TODO:(Jan) I think, it is better, to create a own dict in each function and update epidemics inside _initzialize.. . Somethink like this:
    # def _initiialize():
    #       epidemics_data.update(f())
    # def f():
    #       return {k:np.zeros() for k in [1,2,3..]}
    #       => possible to use (better readable) dict-comprehension instead of forloop
    # OR to use additional argument 'inplace=True' as in pandas.
    ######################################################
    def _set_status_fields(self, epidemic_data: dict, population_size: int):
        """Set fields for the epidemic statuses of the disease.

        For each epidemic status of the disease course a field is
        created in form of a boolean array. An entry of the array being
        True represents the corresponding agent's current status.
        Initially every agent is set to be cs.SUSCEPTIBLE.

        Args:
            epidemic_data: Epidemic data of all agents.
            population_size: Number of all agents.
        """
        epidemic_data[cs.SUSCEPTIBLE] = np.ones(population_size, dtype=bool)
        initially_false_statuses = [
            cs.EXPOSED,
            cs.ASYMPTOMATIC,
            cs.PRESYMPTOMATIC,
            cs.MILD,
            cs.SEVERE,
            cs.CRITICAL,
            cs.RECOVERED,
            cs.DEAD,
            cs.NEXT_STATUS_SET,
            cs.INFECTED,
            cs.INFECTIOUS,
            cs.SYMPTOMATIC,
        ]
        for status in initially_false_statuses:
            epidemic_data[status] = np.zeros(population_size, dtype=bool)

    def _set_probability_fields(
        self,
        epidemic_data: dict,
        epidemic_user_input: pd.DataFrame,
        population_size: int,
    ):
        """Set fields for the progression probabilities of the disease.

        For each status at which a probability to determine the further
        disease course of the agents is needed a field is created
        containing each agent's progression probability in form of
        an array. This probability depends on the agent's age.

        Args:
            epidemic_data: Epidemic data of all agents.
            epidemic_user_input: User input to specify the disease scenario.
            population_size: Number of all agents.
        """
        age_groups_of_agents = self._generate_age_groups_of_agents()
        progression_probabilities = self._get_progression_probabilities_by_age_group(
            epidemic_user_input,
        )
        probability_statuses = [
            cs.SYMPTOMATIC,
            cs.SEVERE,
            cs.CRITICAL,
            cs.DEAD,
        ]
        for status in probability_statuses:
            epidemic_data[f"{status}_probability"] = progression_probabilities[
                f"{status}_probabilities_by_age_group"
            ][age_groups_of_agents]

    def _generate_age_groups_of_agents(self) -> np.ndarray:
        """Generate the age groups of all agents.

        Returns:
            The corresponding age group of each agent stored in an
            array.
        """
        agents_data = self.people.get_agents_data()
        ages = agents_data[cs.AGE]
        age_cutoffs = Sync.AGE_CUTOFFS[cs.MIN]
        iterable = ((np.nonzero(age_cutoffs <= age)[0][-1]) for age in ages)
        return np.fromiter(iterable, dtype=int)

    def _get_progression_probabilities_by_age_group(
        self,
        epidemic_user_input: pd.DataFrame,
    ) -> dict:
        """Get the progression probabilities for each age group.

        Read the epidemic user input and generate a dict containing the
        progression probabilities of each status at which a probability
        to determine the further disease course of the agents is needed.
        For each status the corresponding dict entry is given by an
        array containing the progression probability for each age group.

        Args:
            epidemic_user_input: User input specifying the disease.

        Returns:
            A dict containing the progression probabilities for each age
            group.
        """
        progression_probabilities = {}
        probability_statuses = [
            cs.SYMPTOMATIC,
            cs.SEVERE,
            cs.CRITICAL,
            cs.DEAD,
        ]
        for status in probability_statuses:
            status_probabilities_by_age_group = []
            for age in Sync().AGE_CUTOFFS["Min"]:
                status_probabilities_by_age_group.append(
                    epidemic_user_input.loc[
                        f"{status}_probability_from_{age}",
                        cs.VALUE,
                    ],
                )
            progression_probabilities[
                f"{status}_probabilities_by_age_group"
            ] = np.array(status_probabilities_by_age_group)
        return progression_probabilities

    def _set_date_fields(self, epidemic_data: dict, population_size: int):
        """Set fields for the duration and date of status transitions.

        For all possible status transitions create a field of the
        duration between the old and the new status. Furthermore, for
        each status of the disease course create a field to contain the
        date at which to enter this new status. All fields are
        initially kept unset (set to np.nan).

        Args:
            epidemic_data: Epidemic data of all agents.
            population_size: Number of all agents.
        """
        date_statuses = [
            cs.SUSCEPTIBLE,
            cs.EXPOSED,
            cs.ASYMPTOMATIC,
            cs.PRESYMPTOMATIC,
            cs.MILD,
            cs.SEVERE,
            cs.CRITICAL,
            cs.RECOVERED,
            cs.DEAD,
            cs.INFECTED,
            cs.INFECTIOUS,
            cs.SYMPTOMATIC,
        ]
        for status in date_statuses:
            epidemic_data[f"date_{status}"] = np.full(
                population_size,
                np.nan,
            )

    def _set_initially_infected(
        self,
        epidemic_data: dict,
        epidemic_user_input: pd.DataFrame,
    ):
        """Set the epidemic status of initially infected agents.

        Read the percentage of the initially infected population from
        the epidemic user input and convert it into a number of
        infected agents by multiplying it with the unreported factor of
        infections. Select the initially infected agents at random and
        set their disease status as cs.INFECTED.

        Args:
            epidemic_data: Epidemic data of all agents.
            epidemic_user_input: User input to specify the disease scenario.
        """
        reported_infected_percentage = epidemic_user_input.loc[
            "reported_infected_percentage",
            cs.VALUE,
        ]
        reported_infected_number = self.people.determine_number_from_percentage(
            reported_infected_percentage,
        )
        unreported_factor = epidemic_user_input.loc["unreported_factor", cs.VALUE]
        initially_infected_number = int(reported_infected_number * unreported_factor)
        initially_infected_ids = self.seed.choice(
            self.people.get_data_for(cs.ID),
            initially_infected_number,
            replace=False,
        ).tolist()
        initially_infected_indices = self.people.get_ind_from_multiple_agent_ids(
            initially_infected_ids,
        )
        epidemic_data[cs.SUSCEPTIBLE][initially_infected_indices] = False
        epidemic_data[cs.INFECTED][initially_infected_indices] = True

    def _set_initially_recovered(
        self,
        epidemic_data: dict,
        epidemic_user_input: pd.DataFrame,
    ):
        """Set the epidemic status of initially recovered agents.

        Read the percentage of the initially recovered population from
        the epidemic user input and convert it into a number of
        agents. Select the initially recovered agents at random and
        set their disease status as cs.RECOVERED.

        Args:
            epidemic_data: Epidemic data of all agents.
            epidemic_user_input: User input to specify the disease scenario.
        """
        initially_recovered_percentage = epidemic_user_input.loc[
            "initially_recovered_percentage",
            cs.VALUE,
        ]
        initially_recovered_number = self.people.determine_number_from_percentage(
            initially_recovered_percentage,
        )
        initially_recovered_ids = self.seed.choice(
            self.people.get_data_for(cs.ID)[epidemic_data[cs.SUSCEPTIBLE]],
            initially_recovered_number,
            replace=False,
        ).tolist()
        initially_recovered_indices = self.people.get_ind_from_multiple_agent_ids(
            initially_recovered_ids,
        )
        epidemic_data[cs.SUSCEPTIBLE][initially_recovered_indices] = False
        epidemic_data[cs.RECOVERED][initially_recovered_indices] = True
        # TODO: Think about putting also random numbers here
        epidemic_data[f"date_{cs.RECOVERED}"][initially_recovered_indices] = -1

    def _determine_probabilities_of_initial_statuses(
        self,
        epidemic_data: dict,
        initially_infected_indices: np.ndarray,
    ) -> dict:
        """Determine the probabilities of initial disease statuses.

        For each disease status within the category cs.INFECTED determine
        the probabilities of an initially infected agent being in this
        status. The probability of each agent is determined as a
        combination of their personal probability to move to the given
        disease status and the mean duration of the given disease
        status.

        Args:
            epidemic_data: Epidemic data of all agents.
            initially_infected_indices: Indices of the initially infected agents.

        Returns:
            A dict containing for each initially infected agent the
            probability to be in a specific cs.INFECTED disease status.
        """
        number_of_infected = len(initially_infected_indices)
        pre_probabilities = {}
        initial_statuses = [
            cs.EXPOSED,
            cs.ASYMPTOMATIC,
            cs.PRESYMPTOMATIC,
            cs.MILD,
            cs.SEVERE,
            cs.CRITICAL,
        ]
        for status in initial_statuses:
            transition_durations = self._get_mean_transition_durations(
                status,
            )
            if status == cs.EXPOSED:
                status_probabilities = np.ones(number_of_infected)
            elif status == cs.ASYMPTOMATIC:
                probability_status = cs.SYMPTOMATIC
                status_probabilities = (
                    np.ones(number_of_infected)
                    - epidemic_data[f"{probability_status}_probability"][
                        initially_infected_indices
                    ]
                )
            elif status in {cs.PRESYMPTOMATIC, cs.MILD}:
                probability_status = cs.SYMPTOMATIC
                status_probabilities = epidemic_data[
                    f"{probability_status}_probability"
                ][initially_infected_indices]
            else:
                status_probabilities = epidemic_data[f"{status}_probability"][
                    initially_infected_indices
                ]

            pre_probabilities[status] = (
                statistics.mean(transition_durations) * status_probabilities
            )
        return normalize_dictionary_of_arrays(pre_probabilities)

    def _determine_status_of_initially_infected(
        self,
        epidemic_data: dict,
        initially_infected_indices: np.ndarray,
    ):
        """Determine the disease status of initially infected agents.

        For each disease status within the category cs.INFECTED determine
        the probabilities of an agent initially being in this status.
        Set the disease status of each infected agent at random and
        collect the indices of agents in the same status in the entry of
        a dict. Finally, set the information in the epidemic_data dict
        accordingly.

        Args:
            epidemic_data: Epidemic data of all agents.
            initially_infected_indices: Indices of the initially infected agents.
        """
        status_probabilities = self._determine_probabilities_of_initial_statuses(
            epidemic_data,
            initially_infected_indices,
        )
        number_of_infected = len(initially_infected_indices)
        initial_statuses = np.full([number_of_infected], "status_description")
        for index in range(number_of_infected):
            initial_statuses[index] = self.seed.choice(
                list(status_probabilities.keys()),
                1,
                p=[
                    status_probabilities[cs.EXPOSED][index],
                    status_probabilities[cs.ASYMPTOMATIC][index],
                    status_probabilities[cs.PRESYMPTOMATIC][index],
                    status_probabilities[cs.MILD][index],
                    status_probabilities[cs.SEVERE][index],
                    status_probabilities[cs.CRITICAL][index],
                ],
            )[0]
        initial_statuses_indices = {}
        for status in status_probabilities:
            initial_statuses_indices[status] = initially_infected_indices[
                np.nonzero(initial_statuses == status)[0]
            ]
        self._set_epidemic_data_for_initially_infected(
            epidemic_data,
            initial_statuses_indices,
        )

    def _get_mean_transition_durations(self, status: str) -> list:
        """Get the mean durations of possible status transitions.

        Collect the mean transition durations for all possible disease
        status transitions starting at the given status in a list.

        Args:
            status: Initial status of the disease status transition.

        Returns:
            A list containing the mean transition duration for all
            possible disease status transitions of the given status.
        """
        transition_durations = []
        transitions = self.DURATION_PARAMETERS.keys()
        for transition in transitions:
            if status == transition.split("_to_")[0]:
                transition_durations.append(
                    self.DURATION_PARAMETERS[transition]["mean"],
                )
        return transition_durations

    # TODO: Change cs.STATUS in docstrings to regular strings and change date of initial
    #   status in docstring.
    def _set_epidemic_data_for_initially_infected(
        self,
        epidemic_data: dict,
        initial_statuses_indices: dict,
    ):
        """Set the epidemic data for initially infected agents.

        For every cs.INFECTED disease status set the epidemic data of the
        corresponding initially infected agents. This includes setting
        the disease status, the cs.INFECTIOUS and cs.SYMPTOMATIC fields as
        well as the dates of all possible next disease course statuses.
        The date of the initial disease course status is for all agents
        set to a random integer depending on the status.

        Args:
            epidemic_data: Epidemic data of all agents.
            initial_statuses_indices: Indices of agents in the possible initial disease statuses.
        """
        for status in initial_statuses_indices.keys():
            status_indices = initial_statuses_indices[status]
            if status_indices.size == 0:
                continue
            mean_transition_duration = statistics.mean(
                self._get_mean_transition_durations(status),
            )
            initial_time_steps = self.seed.integers(
                -mean_transition_duration,
                0,
                status_indices.size,
                endpoint=True,
            )
            epidemic_data[status][status_indices] = True
            epidemic_data[f"date_{status}"][status_indices] = initial_time_steps
            epidemic_data[f"date_{cs.INFECTED}"][status_indices] = initial_time_steps
            if status != cs.EXPOSED:
                epidemic_data[cs.INFECTIOUS][status_indices] = True
                epidemic_data[f"date_{cs.INFECTIOUS}"][
                    status_indices
                ] = initial_time_steps
            if status in {cs.MILD, cs.SEVERE, cs.CRITICAL}:
                epidemic_data[cs.SYMPTOMATIC][status_indices] = True
                epidemic_data[f"date_{cs.SYMPTOMATIC}"][
                    status_indices
                ] = initial_time_steps
            next_statuses = self._get_possible_next_statuses(status)
            for next_status in next_statuses:
                self._set_date_of_next_status(
                    epidemic_data,
                    status,
                    next_status,
                    status_indices,
                    initial_time_steps,
                )

    @abstractmethod
    def _set_infection_transmission_fields(
        self,
        epidemic_data: dict,
        epidemic_user_input: pd.DataFrame,
    ):
        """Set fields necessary for the infection transmission.

        This method has to be implemented by the subclasses of the
        EpidemicSpreader class.

        Args:
            epidemic_data: Epidemic data of all agents.
            epidemic_user_input: User input to specify the disease scenario.
        """
        pass

    @abstractmethod
    def _initialize_infection_evolution(
        self,
        epidemic_data: dict,
        infected_indices: np.ndarray,
    ):
        """Initialize the evolution of the infection.

        This method has to be implemented by the subclasses of the
        EpidemicSpreader class.

        Args:
            epidemic_data: Epidemic data of all agents.
            infected_indices: Indices of the newly infected agents.
        """
        pass

    def _move_epidemic_status(self, epidemic_data: dict, time_step: int):
        """Move the epidemic status of all infected agents.

        For every possible disease course status transition move the
        status of agents transitioning at this time step of the
        simulation from the old to the new disease course status and
        change their epidemic data accordingly. Finally, the transition
        date of the next disease course status is determined. For some
        new statuses there are two possible next statuses. In that
        case the transition date is determined for both of them.

        Later in the simulation, when the first of the possibly two
        dates is reached, it is decided to which of the possible next
        statuses the agent actually transitions.

        Args:
            epidemic_data: Epidemic data of all agents.
            time_step: Time step of the simulation in days.
        """
        for transition in self.DURATION_PARAMETERS.keys():
            old_status, new_status = transition.split("_to_")
            self._determine_new_status(
                epidemic_data,
                old_status,
                new_status,
                time_step,
            )
            transition_indices = self._filter_transitioning_agents(
                epidemic_data,
                old_status,
                new_status,
                time_step,
            )
            if transition_indices.size == 0:
                continue
            self._exit_old_status(
                epidemic_data,
                old_status,
                transition_indices,
            )
            self._enter_new_status(
                epidemic_data,
                new_status,
                transition_indices,
                time_step,
            )
            next_statuses = self._get_possible_next_statuses(new_status)
            for next_status in next_statuses:
                self._set_date_of_next_status(
                    epidemic_data,
                    new_status,
                    next_status,
                    transition_indices,
                    time_step,
                )
            epidemic_data[cs.NEXT_STATUS_SET][transition_indices] = False

    def _determine_new_status(
        self,
        epidemic_data: dict,
        old_status: str,
        new_status: str,
        time_step: int,
    ):
        """Determine the new disease course status of agents.

        For a given disease course status transition collect the agents
        that could transition to the new status in this time step,
        but for which the new status of their transition is not yet
        determined. This is necessary, since for some statuses there are
        two possible new statuses to transition to. Determine and set
        the new status of the transitioning agents.

        Args:
            epidemic_data: Epidemic data of all agents.
            old_status: Departure status of the transition.
            new_status: Destination status of the transition.
            time_step: Time step of the simulation in days.
        """
        condition1 = epidemic_data[old_status]
        condition2 = np.invert(epidemic_data[cs.NEXT_STATUS_SET])
        condition3 = epidemic_data[f"date_{new_status}"] == time_step
        indices = np.nonzero(condition1 & condition2 & condition3)[0]
        self._set_new_status(
            epidemic_data,
            old_status,
            new_status,
            indices,
        )

    def _set_new_status(
        self,
        epidemic_data: dict,
        old_status: str,
        new_status: str,
        indices: np.ndarray,
    ):
        """Set the new disease course status of agents.

        From the selected agents, determine those that actually
        transition to the given new disease course status. If there
        exists another possible new status, collect the agents
        transitioning to this other new status. Set the epidemic data of
        all agents accordingly: Reset (to np.nan) the date of the status
        an agent is not transitioning to. Finally, capture that the next
        status of all selected agents is set.

        Args:
            epidemic_data: Epidemic data of all agents.
            old_status: Departure status of the transition.
            new_status: Destination status of the transition.
            indices: Indices of selected agents.
        """
        epidemic_data[cs.NEXT_STATUS_SET][indices] = True
        indices_this_new_status = self._determine_indices_for_new_status(
            epidemic_data,
            old_status,
            new_status,
            indices,
        )
        possible_new_statuses = self._get_possible_next_statuses(old_status)
        possible_new_statuses.remove(new_status)
        if not possible_new_statuses:
            return
        # For every disease course status of this model there are at most two possible
        # next statuses. If the possible status transitions of this model were to be
        # changed, this method would also have to be updated.
        other_new_status = possible_new_statuses[0]
        epidemic_data[f"date_{other_new_status}"][indices_this_new_status] = np.nan
        indices_other_new_status = np.setdiff1d(indices, indices_this_new_status)
        if indices_other_new_status.size > 0:
            epidemic_data[f"date_{new_status}"][indices_other_new_status] = np.nan

    def _determine_indices_for_new_status(
        self,
        epidemic_data: dict,
        old_status: str,
        new_status: str,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Determine the agents transitioning to the given new status.

        If the selected agents transition from a disease course status,
        where there is only one possible next status, return all
        indices.
        Otherwise, get the disease progression probabilities of the
        given new status and evaluate these for the selected agents.

        Args:
            epidemic_data: Epidemic data of all agents.
            old_status: Departure status of the transition.
            new_status: Destination status of the transition.
            indices: Indices of selected agents.

        Returns:
            An array containing the indices of the selected agents that
            transition to the given new disease course status.
        """
        if old_status in {cs.SUSCEPTIBLE, cs.PRESYMPTOMATIC, cs.ASYMPTOMATIC}:
            return indices
        probabilities = self._get_progression_probabilities(
            epidemic_data,
            old_status,
            new_status,
            indices,
        )
        this_new_status = evaluate_probabilities(probabilities, self.seed)
        return indices[this_new_status]

    def _get_progression_probabilities(
        self,
        epidemic_data: dict,
        old_status: str,
        new_status: str,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Get the progression probabilities of a status transition.

        Get the selected agents' disease progression probabilities for
        the given transition. The probabilities are either read or
        calculated from the epidemic_data entries and are furthermore
        modified, by taking epidemic influence factors into account.

        Args:
            epidemic_data: Epidemic data of all agents.
            old_status: Departure status of the transition.
            new_status: Destination status of the transition.
            indices: Indices of selected agents.

        Returns:
            An array containing the probabilities of selected agents to
            move to the given new status.
        """
        if new_status == cs.PRESYMPTOMATIC:
            probabilities = epidemic_data["symptomatic_probability"][indices]
            probabilities = self._modify_progression_probabilities(
                cs.SYMPTOMATIC,
                probabilities,
                indices,
            )

        elif new_status == cs.ASYMPTOMATIC:
            symptomatic_probabilities = epidemic_data["symptomatic_probability"][
                indices
            ]
            symptomatic_probabilities = self._modify_progression_probabilities(
                cs.SYMPTOMATIC,
                symptomatic_probabilities,
                indices,
            )
            probabilities = (
                np.ones(indices.size, dtype=float) - symptomatic_probabilities
            )

        elif new_status == cs.RECOVERED:
            possible_new_statuses = self._get_possible_next_statuses(old_status)
            possible_new_statuses.remove(cs.RECOVERED)
            other_new_status = possible_new_statuses[0]
            other_new_status_probabilities = epidemic_data[
                f"{other_new_status}_probability"
            ][indices]
            other_new_status_probabilities = self._modify_progression_probabilities(
                other_new_status,
                other_new_status_probabilities,
                indices,
            )
            probabilities = (
                np.ones(indices.size, dtype=float) - other_new_status_probabilities
            )

        else:
            probabilities = epidemic_data[f"{new_status}_probability"][indices]
            probabilities = self._modify_progression_probabilities(
                new_status,
                probabilities,
                indices,
            )

        return probabilities

    def _modify_progression_probabilities(
        self,
        new_status: str,
        probabilities: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Modify the progression probabilities of agents.

        Depending on the destination status of the disease course
        transition, modify the probability to move to this new status by
        taking epidemic influence factors like hospitalization or
        vaccination into account for all selected agents.

        Args:
            new_status: Destination status of the transition.
            probabilities: Probabilities to move to the given new status.
            indices: Indices of selected agents.

        Returns:
            An array containing the modified probabilities of the
            selected agents to move to the given new status.
        """
        if indices.size == 0:
            return probabilities
        hospital_factors = self._determine_hospital_factors(
            new_status,
            indices,
        )
        vaccination_factors = self._determine_vaccination_factors(
            indices,
        )
        return probabilities * hospital_factors * vaccination_factors

    def _determine_hospital_factors(
        self,
        new_status: str,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Determine the influence of hospitalization on the progression
        probabilities.

        For transitions to a cs.CRITICAL or cs.DEAD status, check which of
        the selected agents are hospitalized. For non-hospitalized
        agents the progression probability is multiplied by a factor,
        that is read from the epidemic user input.

        Args:
            new_status: Destination status of the transition.
            indices: Indices of the selected agents.

        Returns:
            An array containing factors to multiply with the disease
            progression probabilities of the selected agents.
        """
        hospital_factors = np.ones(indices.size)
        if new_status in {cs.CRITICAL, cs.DEAD}:
            not_hospitalized_indices = np.invert(
                self.people.get_data_for(cs.HOSPITALIZED, indices),
            )
            not_hospitalized_factor = self.EPIDEMIC_PARAMETERS[
                "not_hospitalized_factor"
            ]
            hospital_factors[not_hospitalized_indices] = (
                not_hospitalized_factor * hospital_factors[not_hospitalized_indices]
            )
        return hospital_factors

    def _determine_vaccination_factors(self, indices) -> np.ndarray:
        """Determine the influence of vaccination on the progression
        probabilities.

        Check which of the selected agents are vaccinated. For
        vaccinated agents the progression probability is multiplied by a
        factor, that is read from the epidemic user input.

        Args:
            indices: Indices of the selected agents.

        Returns:
            An array containing factors to multiply with the disease
            progression probabilities of the selected agents.
        """
        vaccinated_indices = self.people.get_data_for(cs.VACCINATED, indices)
        vaccination_factor = self.EPIDEMIC_PARAMETERS["vaccination_factor_progression"]
        vaccination_factors = np.ones(indices.size)
        vaccination_factors[vaccinated_indices] = (
            vaccination_factor * vaccination_factors[vaccinated_indices]
        )
        return vaccination_factors

    def _filter_transitioning_agents(
        self,
        epidemic_data: dict,
        old_status: str,
        new_status: str,
        time_step: int,
    ) -> np.ndarray:
        """Filter for transitioning agents.

        Collect the agents, that are currently in the given old status
        and that are set to transition to the given new status in this
        time step.

        Args:
            epidemic_data: Epidemic data of all agents.
            old_status: Departure status of the transition.
            new_status: Destination status of the transition.
            time_step: Time step of the simulation in days.

        Returns:
            An array containing the indices of agents that transition
            from the given old status to the given new status in this
            time step.
        """
        condition1 = epidemic_data[old_status]
        condition2 = epidemic_data[cs.NEXT_STATUS_SET]
        condition3 = epidemic_data[f"date_{new_status}"] == time_step
        return np.nonzero(
            condition1 & condition2 & condition3,
        )[0]

    def _exit_old_status(
        self,
        epidemic_data: dict,
        old_status: str,
        transition_indices: np.ndarray,
    ):
        """Exit the old status of transitioning agents.

        Set the given disease course status as False for the
        transitioning agents.

        Args:
            epidemic_data: Epidemic data of all agents.
            old_status: Departure status of the transition.
            transition_indices: Indices of transitioning agents.
        """
        epidemic_data[old_status][transition_indices] = False

    def _enter_new_status(
        self,
        epidemic_data: dict,
        new_status: str,
        transition_indices: np.ndarray,
        time_step: int,
    ):
        """Enter the new status of transitioning agents.

        Set the given disease course status as True for the
        transitioning agents and set fields like cs.INFECTIOUS and
        cs.SYMPTOMATIC accordingly.

        Args:
            epidemic_data: Epidemic data of all agents.
            new_status: Destination status of the transition.
            transition_indices: Indices of transitioning agents.
            time_step: Time step of the simulation in days.
        """
        epidemic_data[new_status][transition_indices] = True
        if new_status == cs.EXPOSED:
            epidemic_data[cs.INFECTED][transition_indices] = True
            epidemic_data[f"date_{cs.INFECTED}"][transition_indices] = time_step
        elif new_status in {cs.ASYMPTOMATIC, cs.PRESYMPTOMATIC}:
            epidemic_data[cs.INFECTIOUS][transition_indices] = True
            epidemic_data[f"date_{cs.INFECTIOUS}"][transition_indices] = time_step
        elif new_status == cs.MILD:
            epidemic_data[cs.SYMPTOMATIC][transition_indices] = True
            epidemic_data[f"date_{cs.SYMPTOMATIC}"][transition_indices] = time_step
        elif new_status in {cs.RECOVERED, cs.DEAD}:
            epidemic_data[cs.INFECTED][transition_indices] = False
            epidemic_data[cs.INFECTIOUS][transition_indices] = False
            epidemic_data[cs.SYMPTOMATIC][transition_indices] = False

    def _get_possible_next_statuses(self, status: str) -> list:
        """Get a list of possible next disease course statuses.

        Args:
            status: Disease course status.

        Returns:
            A list containing possible next disease course statuses
            of the given status.

        Raises:
            ValueError: An invalid status was inserted.
        """
        if status == cs.SUSCEPTIBLE:
            next_statuses = [cs.EXPOSED]
        elif status == cs.EXPOSED:
            next_statuses = [cs.PRESYMPTOMATIC, cs.ASYMPTOMATIC]
        elif status == cs.PRESYMPTOMATIC:
            next_statuses = [cs.MILD]
        elif status == cs.ASYMPTOMATIC:
            next_statuses = [cs.RECOVERED]
        elif status == cs.MILD:
            next_statuses = [cs.SEVERE, cs.RECOVERED]
        elif status == cs.SEVERE:
            next_statuses = [cs.CRITICAL, cs.RECOVERED]
        elif status == cs.CRITICAL:
            next_statuses = [cs.DEAD, cs.RECOVERED]
        elif status in {cs.RECOVERED, cs.DEAD}:
            # TODO: Possible to add cs.SUSCEPTIBLE as the next status after cs.RECOVERED
            #       1.) if duration parameters for this transition are added in the
            #           epidemic user input and
            #       2.) if fields for the transition are added in
            #           self.DURATION_PARAMETERS (in the correct position!).
            next_statuses = []

        else:
            raise ValueError(
                f"{status} is not a valid status",
            )
        return next_statuses

    def _set_date_of_next_status(
        self,
        epidemic_data: dict,
        new_status: str,
        next_status: str,
        transition_indices: np.ndarray,
        time_step: Union[int, np.ndarray],
    ):
        """Set the date of the next disease course status.

        For a given next disease course status of the transitioning
        agents determine the transition duration to this next status and
        set the date of a possible transition. If this method is called
        during the initialization of the simulation, an array of time
        steps is used as input parameter (instead of a single integer).

        Args:
            epidemic_data: Epidemic data of all agents.
            new_status: Destination status of the transition.
            next_status: Next disease course status.
            transition_indices: Indices of transitioning agents.
            time_step: Time step of the simulation in days.
        """
        transition = f"{new_status}_to_{next_status}"
        transition_durations = self._determine_transition_duration(
            transition,
            transition_indices,
        )
        initialization = isinstance(time_step, np.ndarray)
        if initialization:
            # The first time step in which the agents move through the disease course
            # is the first, i.e. time_step==1 -> the dates of the next possible statuses
            # in the initialization must not be lower then 1
            next_status_dates = np.where(
                time_step + transition_durations < 1,
                1,
                time_step + transition_durations,
            )
        else:
            next_status_dates = time_step + transition_durations
        epidemic_data[f"date_{next_status}"][transition_indices] = next_status_dates

    def _determine_transition_duration(
        self,
        transition: str,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Determine durations for a given status transition.

        For a given disease course status transition determine its
        duration for the selected agents. The durations are randomly
        distributed following a converted log-normal distribution, for
        which the desired mean and standard deviation are used as
        parameters.

        Args:
            transition: Transition between two disease course statuses.
            indices: Indices of selected agents.

        Returns:
            An array containing the transition durations of the selected
            agents in days.
        """
        distribution_parameters = self.DURATION_PARAMETERS.get(transition)
        duration_distribution = converted_lognormal(
            mean=distribution_parameters["mean"],
            std=distribution_parameters["std"],
            size=len(indices),
            seed=self.seed,
        )
        return duration_distribution.round()


class AirborneVirusSpreader(EpidemicSpreader):
    """EpidemicSpreader subclass dedicated to infection transmission.

    The AirborneVirusSpreader is a subclass of the EpidemicSpreader that
    models the infection transmission of diseases spread by viruses over
    air.
    """

    # TODO: Create classes ..Aresole and ..Proxmily and conect than via inheriance to one class
    #   -> e.g. _calculate_received_quanta_doses_aerosols reduced to _calculate_received_quanta_doses

    def infect_agents(self, time_step: int):
        """Change the epidemic status of agents within a time step.

        Collect all infectious agents and transfer the infection from
        these to other agents. If new agents are infected in this time
        step set their exposure date accordingly. After spreading the
        disease in the population, move the epidemic status of infected
        agents by transitioning from the current to a next disease
        course status. Finally, initialize the viral load evolution of
        newly infected agents. All changes of this time step are written
        into a copy of the epidemic_data dict of the People object,
        which at the end of the method is returned back to the People
        object.

        Args:
            time_step: Time step of the simulation in days.
        """
        epidemic_data = self.people.get_epidemic_data()
        agents_data = self.people.get_agents_data()

        infectious_ids = self._get_infectious_ids(epidemic_data, agents_data)
        infected_indices = self._transfer_infection(
            epidemic_data,
            infectious_ids,
            time_step,
        )
        if infected_indices:
            infected_indices = np.unique(infected_indices).astype(np.int64)
            epidemic_data[f"date_{cs.EXPOSED}"][infected_indices] = time_step
        self._move_epidemic_status(epidemic_data, time_step)
        self._initialize_viral_load_evolution(epidemic_data, infected_indices)
        self.people.update_epidemic_data(epidemic_data)

    def _get_infectious_ids(
        self,
        epidemic_data: dict,
        agents_data: dict,
    ) -> np.ndarray:
        """Get the ids of all infectious agents.

        Args:
            epidemic_data: Epidemic data of all agents.
            agents_data: Personal data of all agents.

        Returns:
            An array containing the ids of all infectious agents.
        """
        infectious_indices = np.nonzero(epidemic_data[cs.INFECTIOUS])[0]
        return agents_data[cs.ID][infectious_indices]

    def _set_infection_transmission_fields(
        self,
        epidemic_data: dict,
        epidemic_user_input: pd.DataFrame,
    ):
        """Set fields necessary for the infection transmission.

        In the epidemic_data dict set fields, necessary for the
        transmission of the infection, e.g. the viral load or the
        relative sensitivity of each agent. The viral load and droplet
        volume concentration of each agent are set following a
        log-normal and normal distribution, respectively. The
        corresponding parameters are read from the user input. Since the
        viral load parameters are given to with respect to the basis
        log_10, they have to be multiplied by the factor ln(10) to yield
        the correct distribution.

        Args:
            epidemic_data: Epidemic data of all agents.
            epidemic_user_input: User input to specify the disease scenario.
        """
        population_size = self.people.get_pop_size()
        epidemic_data["viral_load"] = self.seed.lognormal(
            mean=math.log(10) * epidemic_user_input.loc["viral_load_mu", cs.VALUE],
            sigma=math.log(10) * epidemic_user_input.loc["viral_load_sigma", cs.VALUE],
            size=population_size,
        )
        epidemic_data["droplet_volume_concentration"] = self.seed.normal(
            loc=epidemic_user_input.loc["droplet_volume_concentration_mean", cs.VALUE],
            scale=epidemic_user_input.loc["droplet_volume_concentration_std", cs.VALUE],
            size=population_size,
        )
        epidemic_data["relative_sensitivity"] = np.ones(population_size)
        epidemic_data["viral_load_evolution_start"] = np.full(population_size, np.nan)

    def _initialize_infection_evolution(
        self,
        epidemic_data: dict,
        infected_indices: np.ndarray,
    ):
        """Initialize the evolution of infection of infected agents.

        Args:
            epidemic_data: Epidemic data of all agents.
            infected_indices: Indices of the newly infected agents.
        """
        self._initialize_viral_load_evolution(epidemic_data, infected_indices)

    # TODO: Check if this method has to be updated, since the initialization was changed.
    def _initialize_viral_load_evolution(
        self,
        epidemic_data: dict,
        infected_indices: np.ndarray,
    ):
        """Initialize the viral load evolution of infected agents.

        For all newly infected agents collect the dates of transition to
        a presymptomatic and a asymptomatic disease course status and
        set the starting date of the viral load evolution as their rounded
        mean.  If for some newly infected agents these dates are not
        set, as for example in the initialization of the simulation, the
        starting date of their viral load evolution is set to 0.

        in [19]: infected_inidices = [0, 1, 3]

        in [20]:    asym.   presym.
                0   4       6
                1   2       0
                2
                3   5       nan

        out [21]:   asym.   presym.    viral_load_evo.
                0   4       6           5
                1   2       0           1
                2
                3   5       nan         3

        Args:
            epidemic_data: Epidemic data of all agents.
            infected_indices: Indices of the infected agents.
        """
        date_infectious = np.round(
            np.mean(
                [
                    # change nan -> 0 (e.g. in the initialization of the simulation)
                    np.nan_to_num(
                        epidemic_data["date_presymptomatic"][infected_indices],
                    ),
                    np.nan_to_num(
                        epidemic_data["date_asymptomatic"][infected_indices],
                    ),
                ],
                axis=0,
            ),
        )
        epidemic_data["viral_load_evolution_start"][infected_indices] = date_infectious

    def _evolve_viral_load(
        self,
        epidemic_data: dict,
        agent_index: int,
        time_step: int,
        days: int = 4,
    ) -> float:
        f"""Evolve the viral load of an infectious agent.

        Read the regular viral load and the start date of the viral load
        evolution of the infectious agent. Check if the agent is
        vaccinated and modify the viral load by multiplying with a
        factor read from the user input. Furthermore, take the time
        evolution of the viral load into account, which is modeled by a
        Heaviside step function. Here the viral load is by default
        doubled for the first {days} days after the start of the viral
        load evolution.

        Args:
            epidemic_data: Epidemic data of all agents.
            agent_index: Index of the infectious agent.
            time_step: Time step of the simulation in days.
            days: Number of days for which the viral load is doubled.

        Returns:
            The evolved viral load of the infectious agent.
        """
        viral_load = epidemic_data["viral_load"][agent_index]
        evolution_time = (
            time_step - epidemic_data["viral_load_evolution_start"][agent_index]
        )
        if self.people.get_data_for(cs.VACCINATED, agent_index):
            vaccination_factor = self.EPIDEMIC_PARAMETERS[
                "vaccination_factor_viral_load"
            ]
        else:
            vaccination_factor = 1.0
        days_of_increased_viral_load = days
        return (
            vaccination_factor
            * viral_load
            * (1.0 + (evolution_time <= days_of_increased_viral_load))
        )

    def _evolve_relative_sensitivity(
        self,
        epidemic_data: dict,
        susceptible_indices: np.ndarray,
        time_step: int,
    ) -> np.ndarray:
        """Evolve the relative sensitivity of susceptible agents.

        Read the relative sensitivity of the susceptible agents and
        check which agents are vaccinated. The relative sensitivity of
        the vaccinated agents is then multiplied by a factor read from
        the user input.

        Args:
            epidemic_data: Epidemic data of all agents.
            susceptible_indices: Indices of the susceptible agents.
            time_step: Time step of the simulation in days.

        Returns:
            An array containing the evolved relative sensitivities of
            the susceptible agents.
        """
        vaccination_factor = self.EPIDEMIC_PARAMETERS[
            "vaccination_factor_relative_sensitivity"
        ]
        evolved_relative_sensitivity = epidemic_data["relative_sensitivity"][
            susceptible_indices
        ]
        is_vaccinated = self.people.get_data_for(cs.VACCINATED, susceptible_indices)
        evolved_relative_sensitivity[is_vaccinated] = (
            vaccination_factor * evolved_relative_sensitivity[is_vaccinated]
        )
        return evolved_relative_sensitivity

    def _transfer_infection(
        self,
        epidemic_data: dict,
        infectious_ids: np.ndarray,
        time_step: int,
    ) -> list:
        """Transfer the infection from infectious to susceptible agents.

        For each infectious agent consider the infection transfer by
        close proximity contacts as well as by aerosols.
        # TODO:(Jan) think about to split in two classes and than connected via inheriance into one class

        Args:
            epidemic_data: Epidemic data of all agents.
            infectious_ids: Ids of the infectious agents.
            time_step: Time step of the simulation in days.

        Returns:
            A list containing the indices of newly infected agents.
        """
        infected_indices = []
        for agent_id in infectious_ids:
            quanta_emission_load = self._calculate_quanta_emission_load(
                epidemic_data,
                agent_id,
                time_step,
            )
            infected_indices += self._transfer_infection_proximity(
                epidemic_data,
                agent_id,
                quanta_emission_load,
                time_step,
            )
            infected_indices += self._transfer_infection_aerosols(
                epidemic_data,
                agent_id,
                quanta_emission_load,
                time_step,
            )
        return infected_indices

    def _calculate_quanta_emission_load(
        self,
        epidemic_data: dict,
        agent_id: int,
        time_step: int,
    ) -> float:
        """Calculate the quanta emission load of an infectious agent.

        The quanta emission load of an infectious agent is determined by
        the multiplication of the agent's viral load, the inverse of the
        quanta conversion factor specific to the disease and the agent's
        droplet volume concentration. Influence factors like the wearing
        of a mask are taken into account by extra multiplication factors
        as specified in the user input. The quanta emission rate is
        later calculated as the multiplication of the quanta emission
        load and the network specific inhalation rate of the infectious
        agent.

        Args:
            epidemic_data: Epidemic data of all agents.
            agent_id: Id of the infectious agent.
            time_step: Time step of the simulation in days.

        Returns:
            The quanta emission load of the infectious agent.
        """
        agent_index = self.people.get_ind_from_agent_id(agent_id)
        viral_load = self._evolve_viral_load(
            epidemic_data,
            agent_index,
            time_step,
        )
        quanta_conversion_factor = self.EPIDEMIC_PARAMETERS["quanta_conversion_factor"]
        droplet_volume_concentration = epidemic_data["droplet_volume_concentration"][
            agent_index
        ]
        if self.people.get_data_for(cs.MASK_WEARING, agent_index):
            mask_factor = self.EPIDEMIC_PARAMETERS["mask_factor_emitter"]
        else:
            mask_factor = 1.0
        return (
            viral_load
            * droplet_volume_concentration
            * mask_factor
            / quanta_conversion_factor
        )

    def _transfer_infection_proximity(
        self,
        epidemic_data: dict,
        agent_id: int,
        quanta_emission_load: float,
        time_step: int,
    ) -> list:
        """Transfer the infection by close proximity contacts.

        Get all primary contacts of the infectious agent and determine
        which of those get infected.

        Args:
            epidemic_data: Epidemic data of all agents.
            agent_id: Id of the infectious agent.
            quanta_emission_load: The agent's quanta emission load.
            time_step: Time step of the simulation in days.

        Returns:
            A list containing the indices of newly infected agents.
        """
        infected_contact_indices = []
        networks_of_agent = self.network_manager.get_all_network_ids_from_agent(
            agent_id,
        )
        for network_id in networks_of_agent:
            primary_contacts = (
                self.network_manager.get_primary_contacts_from_agent_in_network(
                    network_id,
                    agent_id,
                )
            )
            if not primary_contacts:
                continue
            infected_contact_indices += self._determine_infected_contacts(
                epidemic_data,
                network_id,
                agent_id,
                primary_contacts,
                quanta_emission_load,
                time_step,
            )
        return infected_contact_indices

    def _determine_infected_contacts(
        self,
        epidemic_data: dict,
        network_id: int,
        agent_id: int,
        primary_contacts: list,
        quanta_emission_load: float,
        time_step: int,
    ) -> list:
        """Determine infected primary contacts of an infectious agent.

        For a given network of an infectious agent determine the newly
        infected primary contacts, by calculating the received quanta
        doses for all susceptible primary contacts and evaluating them
        in combination with the contacts' relative sensitivities.

        Args:
            epidemic_data: Epidemic data of all agents.
            network_id: Id of the agent's network.
            agent_id: Id of the infectious agent.
            primary_contacts: List of agent's primary network contacts.
            quanta_emission_load: The agent's quanta emission load.
            time_step: Time step of the simulation in days.

        Returns:
            A list containing the indices of newly infected agents.
        """
        agent_index = self.people.get_ind_from_agent_id(agent_id)
        primary_contact_indices = self.people.get_ind_from_multiple_agent_ids(
            primary_contacts,
        )
        susceptible_primary_contact_indices = primary_contact_indices[
            epidemic_data[cs.SUSCEPTIBLE][primary_contact_indices]
        ]
        if susceptible_primary_contact_indices.size == 0:
            return []
        received_quanta_doses = self._calculate_received_quanta_doses_proximity(
            network_id,
            agent_index,
            susceptible_primary_contact_indices,
            quanta_emission_load,
        )
        relative_sensitivity = self._evolve_relative_sensitivity(
            epidemic_data,
            susceptible_primary_contact_indices,
            time_step,
        )
        return self._evaluate_received_quanta_doses(
            susceptible_primary_contact_indices,
            relative_sensitivity,
            received_quanta_doses,
        )

    def _calculate_received_quanta_doses_proximity(
        self,
        network_id: int,
        agent_index: int,
        susceptible_primary_contact_indices: np.ndarray,
        quanta_emission_load: float,
    ) -> np.ndarray:
        """Calculate received quanta doses for transmission by close proximity.

        The calculation of received quanta doses for susceptible primary
        contacts of the infectious agent follows the approach described
        in the following paper:
        https://www.sciencedirect.com/science/article/pii/S0160412020320675?via%3Dihub
        Since the paper considers the airborne transmission of
        infection in rooms and not of direct contacts, here abstract
        parameters 'gamma' and 'beta' are introduced and calibrated to
        model the close proximity infection transmission of COVID-19.

        Args:
            network_id: Id of the agent's network.
            agent_index: Index of the infectious agent.
            susceptible_primary_contact_indices: Indices of susceptible primary contacts.
            quanta_emission_load: The agent's quanta emission load.

        Returns:
            An array containing the received quanta doses of susceptible
            primary contacts of the infectious agent.
        """
        gamma = self.EPIDEMIC_PARAMETERS["gamma"]
        beta = self.EPIDEMIC_PARAMETERS["beta"]
        network_type = self.network_manager.get_type_of_location(
            network_id,
        )
        wearing_mask = self.people.get_data_for(
            cs.MASK_WEARING,
            susceptible_primary_contact_indices,
        )
        mask_factors = np.ones(susceptible_primary_contact_indices.size)
        mask_factors[wearing_mask] = self.EPIDEMIC_PARAMETERS["mask_factor_receiver"]
        social_distancing_factors = self._determine_social_distancing_factors(
            agent_index,
            susceptible_primary_contact_indices,
        )
        emitter_inhalation_rate = Sync.LOCATION_TYPE_INFORMATION[network_type][
            cs.INHALATION_RATE
        ]
        receivers_inhalation_rates = np.full(
            shape=susceptible_primary_contact_indices.size,
            fill_value=emitter_inhalation_rate,
        )
        contact_hours = Sync.LOCATION_TYPE_INFORMATION[network_type][cs.CONTACT_HOURS]
        return (
            receivers_inhalation_rates
            * social_distancing_factors
            * mask_factors
            * gamma
            * emitter_inhalation_rate
            * quanta_emission_load
            * integrate.quad(
                lambda time: (1 - math.exp(-beta * time)),
                0,
                contact_hours,
            )[0]
        )

    def _determine_social_distancing_factors(
        self,
        agent_index: int,
        contact_indices: np.ndarray,
    ) -> np.ndarray:
        """Determine social distancing factors of close proximity contacts.

        Check if the infectious agent and their close proximity contacts
        adhere to social distancing rules. Then determine the social
        distancing influence factor for calculating the received quanta
        doses of the contacts depending on the agent's and the contacts'
        social distancing behavior: If both agents of a contact event
        adhere to the rules, set the influence factor as the
        'social_distancing_factor' read from the user input. If both are
        not social distancing set the influence factor to 1 and if only
        one of the agents is adhering to the social distancing rules set
        the influence factor to the mean of the two previous factors.

        Args:
            agent_index: Index of the infectious agent.
            contact_indices: Indices of agent's primary contacts.

        Returns:
            An array containing social distancing factors for the
            calculation of close proximity received quanta doses.
        """
        agent_is_distancing = self.people.get_data_for(
            cs.DO_SOCIAL_DISTANCING, agent_index
        )
        contacts_are_distancing = self.people.get_data_for(
            cs.DO_SOCIAL_DISTANCING,
            contact_indices,
        )
        social_distancing_factor = self.EPIDEMIC_PARAMETERS["social_distancing_factor"]
        no_social_distancing_factor = 1.0
        # TODO: Reconsider this mixed factor
        mixed_social_distancing_factor = statistics.mean(
            [social_distancing_factor, no_social_distancing_factor],
        )
        social_distancing_factors = np.ones(contacts_are_distancing.size)
        if agent_is_distancing:
            social_distancing_factors[
                contacts_are_distancing
            ] = social_distancing_factor

            social_distancing_factors[
                ~contacts_are_distancing
            ] = mixed_social_distancing_factor
        else:
            social_distancing_factors[
                contacts_are_distancing
            ] = mixed_social_distancing_factor

            social_distancing_factors[
                ~contacts_are_distancing
            ] = no_social_distancing_factor
        return social_distancing_factors

    # @profile
    def _transfer_infection_aerosols(
        self,
        epidemic_data: dict,
        agent_id: int,
        quanta_emission_load: float,
        time_step: int,
    ) -> list:
        """Transfer the infection by aerosols.

        For every localized network of the agent, get all associated
        network members and determine those that get infected.

        Args:
            epidemic_data: Epidemic data of all agents.
            agent_id: Id of the infectious agent.
            quanta_emission_load: The agent's quanta emission load.
            time_step: Time step of the simulation in days.

        Returns:
            A list containing the indices of newly infected agents.
        """
        infected_member_indices = []
        localized_networks_of_agent = (
            self.network_manager.get_localized_network_ids_from_agent(agent_id)
        )
        for localized_network_id in localized_networks_of_agent:
            members = self.network_manager.get_members_from_localized_network(
                localized_network_id,
                agent_id,
            )
            if not members:
                continue
            infected_member_indices += self._determine_infected_network_members(
                epidemic_data,
                localized_network_id,
                members,
                quanta_emission_load,
                time_step,
            )
        return infected_member_indices

    def _determine_infected_network_members(
        self,
        epidemic_data: dict,
        network_id: int,
        members: list,
        quanta_emission_load: float,
        time_step: int,
    ) -> list:
        """Determine infected network members of an infectious agent.

        For a given network of an infectious agent determine the newly
        infected members, by calculating the received quanta doses for
        all susceptible members and evaluating them in combination with
        the members' relative sensitivities.

        Args:
            epidemic_data: Epidemic data of all agents.
            network_id: Id of the agent's network.
            members: List of agent's network members.
            quanta_emission_load: The agent's quanta emission load.
            time_step: Time step of the simulation in days.

        Returns:
            A list containing the indices of newly infected agents.
        """
        member_indices = self.people.get_ind_from_multiple_agent_ids(members)
        susceptible_member_indices = member_indices[
            epidemic_data[cs.SUSCEPTIBLE][member_indices]
        ]
        if susceptible_member_indices.size == 0:
            return []
        received_quanta_doses = self._calculate_received_quanta_doses_aerosols(
            network_id,
            susceptible_member_indices,
            quanta_emission_load,
        )
        relative_sensitivity = self._evolve_relative_sensitivity(
            epidemic_data,
            susceptible_member_indices,
            time_step,
        )
        return self._evaluate_received_quanta_doses(
            susceptible_member_indices,
            relative_sensitivity,
            received_quanta_doses,
        )

    def _calculate_received_quanta_doses_aerosols(
        self,
        network_id: int,
        susceptible_member_indices: np.ndarray,
        quanta_emission_load: float,
    ) -> np.ndarray:
        """Calculate received quanta doses for transmission by aerosols.

        The calculation of received quanta doses for susceptible network
        members of the infectious agent follows the approach described
        in the following paper:
        https://www.sciencedirect.com/science/article/pii/S0160412020320675?via%3Dihub

        Args:
            network_id: Id of the agent's network.
            susceptible_member_indices: Indices of susceptible network members.
            quanta_emission_load: The infectious agent's quanta emission load.

        Returns:
            An array containing the received quanta doses of susceptible
            network members of the infectious agent.
        """
        network_type = self.network_manager.get_type_of_location(
            network_id,
        )
        emitter_inhalation_rate = Sync.LOCATION_TYPE_INFORMATION[network_type][
            cs.INHALATION_RATE
        ]
        receivers_inhalation_rates = np.full(
            shape=susceptible_member_indices.size,
            fill_value=emitter_inhalation_rate,
        )
        air_exchange_rate = Sync.LOCATION_TYPE_INFORMATION[network_type][
            cs.AIR_EXCHANGE_RATE
        ]
        room_volume = Sync.LOCATION_TYPE_INFORMATION[network_type][cs.ROOM_VOLUME]
        contact_hours = Sync.LOCATION_TYPE_INFORMATION[network_type][cs.CONTACT_HOURS]
        particle_deposition = self.EPIDEMIC_PARAMETERS["particle_deposition"]
        viral_inactivation = self.EPIDEMIC_PARAMETERS["viral_inactivation"]
        infectious_virus_removal_rate = (
            air_exchange_rate + particle_deposition + viral_inactivation
        )
        wearing_mask = self.people.get_data_for(
            cs.MASK_WEARING,
            susceptible_member_indices,
        )
        mask_factors = np.ones(susceptible_member_indices.size)
        mask_factors[wearing_mask] = self.EPIDEMIC_PARAMETERS["mask_factor_receiver"]
        return (
            receivers_inhalation_rates
            * mask_factors
            * quanta_emission_load
            * emitter_inhalation_rate
            / (infectious_virus_removal_rate * room_volume)
            * integrate.quad(
                lambda time: (1 - math.exp(-infectious_virus_removal_rate * time)),
                0,
                contact_hours,
            )[0]
        )

    def _evaluate_received_quanta_doses(
        self,
        contact_indices: np.ndarray,
        relative_sensitivity: np.ndarray,
        received_quanta_doses: np.ndarray,
    ) -> list:
        """Evaluate received quanta doses of susceptible contacts.

        Determine for each susceptible contact the probability to be
        infected using an exponential dose response model and evaluate
        the probabilities.

        Args:
            contact_indices: Indices of the susceptible contacts.
            relative_sensitivity: Relative sensitivities of the contacts.
            received_quanta_doses: The received quanta doses of each contact.

        Returns:
            A list containing the indices of contacts that get infected.
        """
        probabilities_of_infection = 1 - np.exp(
            -relative_sensitivity * received_quanta_doses,
        )
        infected_distribution = evaluate_probabilities(
            probabilities_of_infection, self.seed
        )
        infected_indices = contact_indices[infected_distribution]
        return infected_indices.tolist()


def converted_lognormal(
    mean: float,
    std: float,
    size: int,
    seed: np.random.Generator,
) -> np.ndarray:
    r"""Draw samples from a converted log-normal distribution.

    Draw samples from a converted log-normal distribution using a
    specified mean, standard deviation and array shape. For the
    converted log-normal distribution the input parameters represent the
    values of the distribution itself.

    .. math::
        converted_mean = \log(mean²/\sqrt{std² + mean²})
        sigma = \sqrt{\log(std²/mean²+1)}
        return lognormal(mean = converted_mean,sigma = sigma)

    Args:
        mean: Mean value of the log-normal distribution.
        std: Standard deviation of the log-normal distribution.
        size: Number of drawn samples.
        seed: Random number generator with specified seed.

    Returns:
        An array containing drawn samples from the parameterized
        log-normal distribution.
    """
    converted_mean = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(std**2 / mean**2 + 1))
    return seed.lognormal(mean=converted_mean, sigma=sigma, size=size)


def evaluate_probabilities(
    probabilities: np.ndarray,
    seed: np.random.Generator,
) -> np.ndarray:
    """Evaluate probabilities.

    Evaluate probabilities by assigning to each element a random value
    between 0 and 1 and comparing it with the corresponding probability.
    If the randomly generated number is smaller than the probability the
    boolean value True is returned.

    Args:
        probabilities: Probabilities to be evaluated.
        seed: Random number generator with specified seed.

    Returns:
        An array containing evaluated probabilities in form of boolean
        values.
    """
    return seed.random(size=len(probabilities)) < probabilities


def normalize_dictionary_of_arrays(dictionary: dict) -> dict:
    """Normalize the dictionary elements of arrays with dtype=float.

    Convert the dictionary of arrays to an ndarray and assert that each
    element array has the same length. For a given index, normalize all
    the corresponding array elements of dtype float by dividing each of
    them by their common sum.

    >> In [20]:normalize_dictionary_of_arrays({'a': [1, 2], 'b': [2, 3], 'c': [1, 3]})
    >> Out[21]:
        {'a': array([0.25, 0.25]),
        'b': array([0.5  , 0.375]),
        'c': array([0.25 , 0.375])}

    more symbolic:
    in [20]:    a   b   c
            0   1   2   1
            1   2   3   3
    Out[21]:     a      b      c
            0  0.25  0.500  0.250
            1  0.25  0.375  0.375


    Args:
        dictionary: Dictionary of arrays with dtype=float.

    Returns:
        A dictionary with normalized array elements.

    Raises:
        AssertionError: If not all arrays have the same size
    """
    arrays = list(dictionary.values())
    iterator = iter(arrays)
    array_length = len(next(iterator))
    if not all(len(array) == array_length for array in iterator):
        raise ValueError("All arrays must have the same size.")

    summed_up_arrays = np.sum(np.array(arrays), axis=0)
    return {key: key_value / summed_up_arrays for key, key_value in dictionary.items()}
