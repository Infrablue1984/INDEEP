#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Manage public data in ClassPeople und implement interventions.

Iterate interventions in each time step and call responsible classes to implement the intervention.
Set status of public attributes in ClassPeople (e.g. diagnosed, quarantined).

Typical usage example:
    pub_reg = PublicHealthRegulator(
        epi_user_input, INTERVENTION_LIST, people, networks, start_date
    )
    pub_reg.regulate(()
"""

__author__ = "Inga Franzen"
__created__ = "2020"
__date_modified__ = "2022/06/06"
__version__ = "1.0"

import functools
import operator

import numpy as np
# from line_profiler_pycharm import profile
from calculation.interventions import InterventionMaker as IM
from synchronizer import constants as cs
from synchronizer.synchronizer import Synchronizer as Sync

IS_ON = "is on"
UNDO = "undo"
VALUE = "value"
PERIOD = "period"
TIME_DELAY = "time_delay"


# TODO: think of moving these constants to synchronizer/constants.py
LABORATORY_TEST = "laboratory test"
SENSITIVITY = "sensitivity"
SPECIFICITY = "specificity"
RAPID_TEST = "rapid test"
PERCENTAGE_RAPID_TEST = "percentage_rapid_test"


class PublicHealthRegulator:
    """Change public status of ClassPeople, inform ClassNetworkManager and
    perform interventions.

    Based on epidemic status of ClassPeople decide for the next public status, e.g.
    diagnosed, hospitalized. Further, iterate interventions, perform them itself or
    call other classes, such as ClassPeople or ClassNetworkManager.

    Attributes:
        self.people (ClassPeople): Pointer to ClassPeople object.
        self.net_manager (ClassNetworkManager): Pointer to ClassNetworkManager object.
        self.start_date (datetime.date): Date to start simulation from.
        self.start_weekday(int): from 0 to 7, means Mo-Su.
        self.trace_diary: (dict[int, dict[int, dict[int, dict[str, list[int]]]]]): Organizing traced contacts.
        self._trace (dict[str, int]): Hold information on how to trace.
        self._test_info (dict[str, dict[str, float]]): Hold information on tests.
        self._isolate (dict[str, int/bool]): Hold information on isolation.
        self._quarantine (dict[str, int/bool]): Hold information on quarantine.
        self.hospital_bed_number (int): Number of total available hospital beds.
        self.inter_circle (list(ClassIntervention)): Interventions that need to be performed any time step.
        self.inter_point (list(ClassIntervention)): Interventions that need to activate something one time.
    """

    def __init__(self, intervention_list, people, net_manager, start_date, seed):
        """Initialize.

        Args:
            intervention_list (list(ClassIntervention)): list with Pointer to intervention objects.
            people (ClassPeople): Pointer to ClassPeople object.
            net_manager (ClassNetworkManager): Pointer to ClassNetworkManager object.
            start_date (datetime.date): Date to start simulation from.
        """
        self.seed = seed
        self.people = people
        self.net_manager = net_manager
        self.start_date = start_date
        self.start_weekday = start_date.weekday()
        self.trace_diary = {}  #
        self._trace = {}
        self._test_info = {}
        self._isolate = {}
        self._quarantine = {}
        self.hospital_bed_number = None
        self.inter_circle, self.inter_point = self._help_sort_interventions(
            intervention_list
        )

    # @profile
    def initialize_public_data(self, epi_user_input):
        """Initialize data for time step 0.

        Initialize status in public data (for ClassPeople).
        Update ClassNetworkManager (e.g. for hospitalized).
        Perform interventions (e.g. trace contact status, testing).

        Args:
            epi_user_input: User input to specify the disease.
        """
        public_data = self.people.get_public_data()
        epi_data = self.people.get_epidemic_data()
        self._initialize_diagnosed(epi_user_input, public_data, epi_data)
        self._initialize_vaccinated(epi_user_input, public_data)
        self._initialize_hospitalized(epi_user_input, public_data, epi_data)
        self._initialize_intervention_base_values()
        self._perform_one_time_interventions(public_data, self.start_date)
        self._perform_cyclic_interventions(public_data, epi_data, 0, self.start_date)
        self._do_isolate_people(public_data, 0)
        self.people.update_public_data(self, public_data)

    def _help_sort_interventions(self, intervention_list):
        """Sort intervention objects by type.

        Point interventions are performed one time at start/end. They set status in ClassPeople,
        ClassNetworkManager or variables. Cyclic interventions are performed in any time step while activated.
        Limit Event Size is put to the end of the list due to run time issues.

        Args:
            intervention_list (list(ClassIntervention)): list with Pointer to intervention objects.
        """
        intervention_list = IM.sort_intervention_list_by_date(intervention_list)
        inter_cyclic = []
        inter_point = []
        inter_point_special = []
        for i, inter in enumerate(intervention_list):
            if inter.get_one_time():
                # fetch to put to the end
                if inter.get_name() == cs.LIMIT_EVENT_SIZE:
                    inter_point_special.append(inter)
                else:
                    inter_point.append(inter)
            else:
                inter_cyclic.append(inter)
        if inter_point_special:  # guarantee special to be last
            for i in range(len(inter_point_special)):
                inter_point.append(inter_point_special.pop())
        return inter_cyclic, inter_point

    def _initialize_intervention_base_values(self):
        """Initialize base values used to perform interventions.

        The values written in the Sync.INTERVENTION_USER_INPUT could be
        integrated into the tool GUI and added to the ClassIntervention.
        """
        self._test_info = Sync.INTERVENTION_USER_INPUT[cs.TEST_INFO]
        self._quarantine = Sync.INTERVENTION_USER_INPUT[cs.QUARANTINED]
        self._isolate = {IS_ON: False, VALUE: 0}
        self._trace = {
            **{IS_ON: 0},
            **Sync.INTERVENTION_USER_INPUT[cs.MANUAL_CONTACT_TRACING],
        }
        for inter in self.inter_circle:
            if IM.intervention_needs_tracing(inter):
                self._trace[IS_ON] += 1
        if self._trace[IS_ON]:
            self._trace_network_status(0, initialize=True)

    # @profile
    def _initialize_diagnosed(self, epi_user_input, public_data, epi_data):
        """Select people to diagnose and initialize status in public data.

        Ensure all people with severe/critical symptoms are diagnosed independent of
        input data. Also ensure there are not more people diagnosed than being infected.
        Only select people form status "mild symptoms"/"pre-"/"asymptomatic"/"exposed"
        according to the given "initially_infected_percentage".

        Args:
            epi_user_input: User input to specify the disease.
            public_data (dict[str, NumPyArray]): Social data of all agents.
            epi_data (dict[str, NumPyArray]): Epidemic data of all agents.
        """
        # extract data
        reported_infected_percentage = epi_user_input.loc[
            "reported_infected_percentage", cs.VALUE
        ]
        num_to_initially_diagnose = self.people.determine_number_from_percentage(
            reported_infected_percentage
        )
        ind_critical_severe = np.nonzero(epi_data[cs.CRITICAL] | epi_data[cs.SEVERE])[0]
        ind_mild = np.nonzero(epi_data[cs.MILD])[0]
        num_infected = np.count_nonzero(epi_data[cs.INFECTED])
        num_critical_severe = len(ind_critical_severe)
        num_mild = len(ind_mild)
        assert num_critical_severe + num_mild <= num_infected
        assert num_critical_severe + num_mild == np.count_nonzero(
            epi_data[cs.SYMPTOMATIC]
        )
        # assert all severe critical are diagnosed, but not more than actually infected
        num_to_diagnose = min(
            max(num_to_initially_diagnose, num_critical_severe), num_infected
        )
        ind_to_diagnose = self._determine_ind_to_initially_diagnose(
            epi_data, ind_critical_severe, ind_mild, num_to_diagnose
        )
        ind_feel_bad = self._determine_indices_to_initially_feel_bad(
            ind_critical_severe, ind_mild
        )
        self._set_status_of_initially_diagnosed_and_feel_bad(
            ind_feel_bad, ind_to_diagnose, public_data
        )
        assert np.count_nonzero(public_data[cs.DIAGNOSED]) == num_to_diagnose
        assert np.all(epi_data[cs.INFECTED][np.nonzero(public_data[cs.DIAGNOSED])[0]])

    def _determine_indices_to_initially_feel_bad(self, ind_critical_severe, ind_mild):
        """Determine people with symptoms that also feel bad.

        It is assumed that NOT ALL people with mild symptoms really feel bad. People
        could cough or sneeze, but still feel ok to leave their house. The percentage
        of those who do really feel bad (defined as staying in bed) is determined by
        the ClassSynchronizer, but could be shifted to the GUI as user input.

        Args:
            ind_critical_severe (NumPyArray): Indices of people with severe/critical symptoms.
            ind_mild (NumPyArray): Indices of people with severe or critical symptoms.

        Returns:
            ind_feel_bad (NumPyArray): Indices of people with symptoms that also feel bad.
        """
        mild_feel_bad = Sync.INTERVENTION_USER_INPUT[cs.DIAGNOSED][
            cs.PERCENTAGE_MILD_FEEL_BAD
        ]
        num_mild_feel_bad = round(mild_feel_bad * len(ind_mild))
        ind_mild_feel_bad = self.seed.choice(ind_mild, num_mild_feel_bad, replace=False)
        ind_feel_bad = np.hstack((ind_critical_severe, ind_mild_feel_bad))
        return ind_feel_bad

    def _determine_ind_to_initially_diagnose(
        self, epi_data, ind_critical_severe, ind_mild, num_to_diagnose
    ):
        """Determine people who will be initially diagnosed.

        People with symptoms are prioritized. If the number of people to be diagnosed
        is higher than people with symptoms, then people without symptoms will also be
        diagnosed.

        Args:
             epi_data (dict[str, NumPyArray]): Epidemic data of all agents.
             ind_critical_severe (NumPyArray): Indices of people with severe/critical symptoms.
             ind_mild (NumPyArray): Indices of people with severe or critical symptoms.
             num_to_diagnose(int): Initial number of cases determined by the user.

         Returns:
             ind_to_diagnose (NumPyArray): Indices of people to diagnose.
        """
        # gather indices to diagnose
        num_critical_severe = len(ind_critical_severe)
        num_mild = len(ind_mild)
        if num_to_diagnose > num_critical_severe:
            # case 1: all mild are diagnosed, also part of exposed/pre-/asymptomatic
            if num_to_diagnose > num_critical_severe + num_mild:
                num_extra = num_to_diagnose - num_critical_severe - num_mild
                ind_extra = np.nonzero(
                    epi_data[cs.INFECTED] & (epi_data[cs.SYMPTOMATIC] == 0)
                )[0]
                ind_extra_choice = self.seed.choice(ind_extra, num_extra, replace=False)
                ind_to_diagnose = np.hstack(
                    (ind_critical_severe, ind_mild, ind_extra_choice)
                )
            # case 2: only a part of the mild are diagnosed, exposed/pre-/asymptomatic not
            else:
                num_extra = num_to_diagnose - num_critical_severe
                ind_extra_choice = self.seed.choice(ind_mild, num_extra, replace=False)
                ind_to_diagnose = np.hstack((ind_critical_severe, ind_extra_choice))
        else:
            ind_to_diagnose = ind_critical_severe
        return ind_to_diagnose

    def _set_status_of_initially_diagnosed_and_feel_bad(
        self, ind_feel_bad, ind_to_diagnose, public_data
    ):
        """Sample time step from the past and set status in public data.

        Distinguish people with status "diagnosed", "feel_bad" or both. Sample time
        step at which people were diagnosed (must not necessarily be 0). To simplify
        it is assumed that time step of being diagnosed is same to feeling bad.
        Finally, set status in public data.

        Args:
            public_data (dict[str, NumPyArray]): Public data of all agents.
            ind_feel_bad (NumPyArray): Indices of people that feel bad.
            ind_to_diagnose (NumPyArray): Indices of people to be diagnosed.
        """
        # set status and sample time steps
        ind_only_to_diagnose = np.setdiff1d(ind_to_diagnose, ind_feel_bad)
        ind_only_feel_bad = np.setdiff1d(ind_feel_bad, ind_to_diagnose)
        ind_both = np.intersect1d(ind_to_diagnose, ind_feel_bad)
        # TODO: actually must check that time step is not before time step infected.
        time_steps_1 = self.seed.integers(-4, 0, ind_only_to_diagnose.size)
        time_steps_2 = self.seed.integers(-4, 0, ind_only_feel_bad.size)
        time_steps_3 = self.seed.integers(-4, 0, ind_both.size)
        set_peoples_status(
            ind_only_to_diagnose, public_data, cs.DIAGNOSED, True, time_steps_1
        )
        set_peoples_status(
            ind_only_feel_bad, public_data, cs.FEEL_BAD, True, time_steps_2
        )
        set_peoples_status(ind_both, public_data, cs.DIAGNOSED, True, time_steps_3)
        set_peoples_status(ind_both, public_data, cs.FEEL_BAD, True, time_steps_3)

    def _initialize_vaccinated(self, epi_user_input, public_data):
        """Select people that are vaccinated and set status in public data.

        Select people randomly uniformly distributed.

        Args:
            epi_user_input: User input to specify the disease.
            public_data (dict[str, NumPyArray]): Public data of all agents.
        """
        agents_data = self.people.get_agents_data()
        # TODO(Inga): maybe sample dates from bigger array with negative values
        time_step = -1
        vaccinated_percentage = epi_user_input.loc[
            "initially_vaccinated_percentage", cs.VALUE
        ]
        number_vaccinated = self.people.determine_number_from_percentage(
            vaccinated_percentage,
        )
        vaccinated_ids = self.seed.choice(
            agents_data["id"],
            number_vaccinated,
            replace=False,
        ).tolist()
        vaccinated_indices = self.people.get_ind_from_multiple_agent_ids(vaccinated_ids)
        set_peoples_status(
            vaccinated_indices, public_data, cs.VACCINATED, True, time_step
        )

    def _initialize_hospitalized(self, epi_user_input, public_data, epi_data):
        """Initialize hospitalization and set relevant variables.

        Args:
            epi_user_input: User input to specify the disease.
            public_data (dict[str, NumPyArray]): Social data of all agents.
            epi_data (dict[str, NumPyArray]): Epidemic data of all agents.
        """
        hospital_percentage = epi_user_input.loc["hospital_percentage", cs.VALUE]
        self.hospital_bed_number = self.people.determine_number_from_percentage(
            hospital_percentage,
        )
        self._do_hospitalize_people(public_data, epi_data, time_step=0)

    def _perform_one_time_interventions(self, public_data, date_of_time_step):
        """Iterate interventions, update its status and perform if status
        changed.

        Point interventions are performed one time at start/end. The performance is
        either done by itself or delegated to another class (e.g. ClassNetworkManager).

        Args:
            public_data (dict[str, NumPyArray]): Public data of all agents.
            date_of_time_step (datetime.date): Date of time step.
        """
        for intervention in self.inter_point:
            intervention.update_status(date_of_time_step)
            if intervention.status_activated_changed():
                my_name = intervention.get_name()
                if cs.HOME_DOING in my_name:
                    function = self.net_manager.organize_home_doing
                    self._perform_specific(
                        my_name, cs.HOME_DOING, function, intervention
                    )
                elif cs.SPLIT_GROUPS in my_name:
                    function = self.net_manager.organize_split_groups
                    self._perform_specific(
                        my_name, cs.SPLIT_GROUPS, function, intervention
                    )
                elif cs.LIMIT_EVENT_SIZE in my_name:
                    function = self.net_manager.organize_limit_event_size
                    self._perform_specific(
                        my_name, cs.LIMIT_EVENT_SIZE, function, intervention
                    )
                elif cs.SHUT_INSTITUTIONS in my_name:
                    function = self.net_manager.organize_shut_institution
                    self._perform_specific(
                        my_name, cs.SHUT_INSTITUTIONS, function, intervention
                    )
                elif my_name in (cs.MASK, cs.SOCIAL_DISTANCING):
                    self._organize_social_behavior(intervention, public_data)
                elif my_name == cs.SELF_ISOLATION:
                    self._activate_deactivate_isolation(intervention)

    def _perform_specific(self, full_name, super_name, function, intervention):
        """Interventions organized by self.net_manager can be further
        specified.

        The interventions can be performed solely for each net_type or for all net_types
        together. This can be selected in the GUI and goes with the intervention name.
        Each net_type goes with a digit, the net_manager needs the specific digit
        to know net_type.

        Args:
            full_name (str): Full name of the intervention containing the specification.
            super_name (str): Name of the general type of the intervention.
            function (ClassMethod): Pointer to the method, which performs the intervention.
            intervention (ClassIntervention): intervention to perform.
        """
        net_type = full_name.split(super_name)[1].strip()
        if not net_type:
            for digit in Sync.PUBLIC_DIGITS:
                function(self, intervention, digit)
        else:
            digit = Sync.DIGIT_TRANSLATE_2[net_type]
            function(self, intervention, digit)

    def _perform_cyclic_interventions(
        self, public_data, epi_data, time_step, date_of_time_step
    ):
        """Iterate interventions, update its status and perform.

        Cyclic interventions are performed in any time step. The check de-/activated
        is done within the _organize methods. The tracing of network status is done
        a couple of time_steps in advance. Release people from quarantine or receive test results
        needs always be done, even if the intervention is de-activated.

        Args:
            public_data (dict[str, NumPyArray]): Public data of all agents.
            epi_data (dict[str, NumPyArray]): Epidemic data of all agents.
            time_step (int): current time_step
            date_of_time_step (datetime.date): Date of time step.
        """
        # trace is done, if at least one intervention is selected
        if self._trace[IS_ON]:
            self._trace_network_status(time_step)
        # regulate according to interventions
        for intervention in self.inter_circle:
            intervention.update_status(date_of_time_step)
            my_name = intervention.get_name()
            if my_name == cs.TESTING:
                self._organize_testing(intervention, public_data, epi_data, time_step)
            elif my_name == cs.MANUAL_CONTACT_TRACING:
                self._organize_trace_contacts(intervention, public_data, time_step)
            elif my_name == cs.CLUSTER_ISOLATION:
                self._organize_cluster_isolation(intervention, public_data, time_step)
            else:
                print("This is not the name of a cyclic intervention.")
        self._do_un_quarantine_people(public_data, time_step)
        self._do_receive_test_results(public_data, epi_data, time_step)

    # @profile
    def _trace_network_status(self, time_step, initialize=False):
        """Get changed network objects from NetworkManager and store
        information.

        The original format is network id (key) and network object (value). Extract
        information about the contacts of each agent and store it in the
        trace_diary. Finally, time steps older than the trace period are dropped.

        Args:
            time_step (int): current time_step
            initialize (boolean): True for time step 0, else False.
        """
        networks_changed = self.net_manager.get_changed_networks()
        for net_id, network in networks_changed.items():
            if initialize:
                self.trace_diary[net_id] = {}
            self.extract_and_store_data_in_trace_diary(net_id, network, time_step)
        self._drop_outdated_data_from_trace_diary(time_step)

    def _drop_outdated_data_from_trace_diary(self, time_step):
        """Drop outdated contact data from trace_diary.

        The trace_diary is of the form:
        trace_diary[net_id][time_step][agent_id][primary/cluster] = list(agent_ids)
        Data from time steps out of the trace period are dropped. But only, if there
        is a newer status, as network status is only traced in the case of change.
        EXAMPLE: trace_dict[net_id] with traced time_steps 8 and 15 means that 8 is also
        representative for time_steps 9 till 14. Accordingly, if time_step_out_of_trace
        _period is 13, then 8 can not be dropped, because it also represents 13 and 14.

        Args:
            time_step (int): current time_step
        """
        time_step_out_of_trace_period = time_step - self._trace[PERIOD]
        if time_step_out_of_trace_period >= 0:
            # iterate all time steps out of trace period
            for older_time_step in range(0, time_step_out_of_trace_period + 1):
                # iterate all net_ids and belonging time_step_dict:
                for net_id, time_step_dict in self.trace_diary.items():
                    # check if time step was tracked for net_id due to changes
                    if (older_time_step in time_step_dict) & (
                        time_step_out_of_trace_period + 1 in time_step_dict
                    ):
                        time_step_dict.pop(older_time_step)

    def extract_and_store_data_in_trace_diary(self, net_id, network, time_step):
        """Extract contacts of each agent from network object.

        Not the whole network object needs to be saved. Only the individual contacts
        for each agent is appended to the trace_diary. Extract contacts both 'primary'
        and 'cluster' and store them in a nested dict of the form:

        trace_diary[net_id][time_step][agent_id][primary/cluster] = list(agent_ids)

        Args:
            net_id (int): The id of the network.
            network (ClassNetwork): Pointer to network object.
            time_step (int): current time_step
        """
        self.trace_diary[net_id][time_step] = {}
        # see Synchronizer.DIGIT_TRANSLATE to understand digits
        digit = network.get_first_digit()
        weekday = (time_step + self.start_weekday) % 7
        active_members = network.get_active_members()
        for agent_id in active_members:
            primary_contacts = network.get_primary_contacts(agent_id, weekday)
            if digit != 7:
                cluster_contacts = network.get_cluster_contacts(agent_id, weekday)
            else:
                cluster_contacts = np.arange(0)
            if primary_contacts or cluster_contacts:
                self.trace_diary[net_id][time_step][agent_id] = {}
                self.trace_diary[net_id][time_step][agent_id][
                    cs.PRIMARY
                ] = primary_contacts
                self.trace_diary[net_id][time_step][agent_id][
                    cs.CLUSTER
                ] = cluster_contacts

    def _organize_testing(self, test_intervention, public_data, epi_data, time_step):
        """Organize testing.

        If intervention is not activated --> Return.
        Else --> Sample indices of agents from target group according to intervention input.
        Then split indices into groups 'rapid test' and 'laboratory test' according to
        _test_info. Finally, implement testing.

        Args:
            test_intervention (ClassIntervention): Pointer to intervention object.
            public_data (dict[str, NumPyArray]): Public data of all agents.
            epi_data (dict[str, NumPyArray]): Epidemic data of all agents.
            time_step (int): current time_step
        """
        if not test_intervention.is_activated():
            return
        # extract data
        agents_data = self.people.get_agents_data()
        attributes = test_intervention.get_target_group()
        interval = test_intervention.get_intervall()
        # set conditions
        is_tested = public_data[cs.TESTED]
        is_diagnosed = public_data[cs.DIAGNOSED]
        is_test_relevant = ~is_tested & ~is_diagnosed
        if attributes == cs.SYMPTOMATIC:
            to_test_indices = np.nonzero(epi_data[cs.SYMPTOMATIC] & is_test_relevant)[0]
        elif (attributes == cs.RANDOM) & (time_step % interval == 0):
            to_test_indices = np.nonzero(is_test_relevant)[0]
        elif attributes == cs.PRIMARY_CONTACT:
            my_sources = agents_data[cs.ID][
                public_data[f"date_{cs.DIAGNOSED}"] == time_step
            ]
            array_of_tuples = self._get_network_tuples_from_agents(my_sources, cs.ALL)
            target_list = self._trace_agent_contacts_from_tuples(
                array_of_tuples, cs.PRIMARY
            )
            to_test_indices = np.nonzero(
                np.isin(agents_data[cs.ID], target_list) & is_test_relevant
            )[0]
        elif (attributes == cs.BIGGER_65) & (time_step % interval == 0):
            to_test_indices = np.nonzero(agents_data[cs.AGE] > 65 & is_test_relevant)[0]
        elif (attributes == cs.PUPIL_TEACHER) & (time_step % interval == 0):
            cond2 = agents_data[cs.OCCUPATION] == "p"
            cond3 = agents_data[cs.OCCUPATION] == "t"
            to_test_indices = np.nonzero((cond2 | cond3) & is_test_relevant)[0]
        else:
            return
        ind_tested = self._sample_percentage_of_array_elements(
            test_intervention, to_test_indices
        )
        num_rapid = int(len(ind_tested) * self._test_info[cs.PERCENTAGE_RAPID_TEST])
        indices_rapid = self.seed.choice(ind_tested, num_rapid, replace=False)
        indices_lab = ind_tested[np.where(~indices_rapid)]

        self._implement_testing(indices_rapid, public_data, time_step, cs.RAPID_TEST)
        self._implement_testing(indices_lab, public_data, time_step, cs.LABORATORY_TEST)

    def _organize_trace_contacts(self, trace_intervention, public_data, time_step):
        """Organize contact tracing.

        Trace primary contacts of freshly diagnosed agents and send percentage of
        these into quarantine (according to intervention input).

        Args:
            trace_intervention (ClassIntervention): Pointer to intervention object.
            public_data (dict[str, NumPyArray]): Public data of all agents.
            time_step (int): current time_step
        """
        if trace_intervention.is_activated():
            my_sources = self._get_freshly_diagnosed_agents(public_data, time_step)
            if not my_sources.size:
                return
            array_of_tuples = self._get_network_tuples_from_agents(my_sources, "all")
            target_list = self._trace_agent_contacts_from_tuples(
                array_of_tuples, cs.PRIMARY
            )
            traced_ids = self._sample_percentage_of_array_elements(
                trace_intervention, target_list
            )
            self._do_quarantine_people(traced_ids, public_data, time_step)
        else:
            if trace_intervention.status_activated_changed():
                self._trace[IS_ON] -= 1

    def _organize_social_behavior(self, intervention, public_data):
        """Organize either social distancing or mask wearing.

        Sample percentage of people being committed according to intervention input
        and change attribute in public data of ClassPeople.

        Args:
            intervention (ClassIntervention): Pointer to intervention object.
            public_data (dict[str, NumPyArray]): Public data of all agents.
        """
        if intervention.get_name() == cs.SOCIAL_DISTANCING:
            behavior = cs.DO_SOCIAL_DISTANCING
        elif intervention.get_name() == cs.MASK:
            behavior = cs.MASK_WEARING
        else:
            raise ValueError("This is not a valid intervention name")

        if intervention.is_activated():
            intervention_commitment = intervention.get_value()
            number_of_committed_agents = self.people.determine_number_from_percentage(
                intervention_commitment,
            )
            agents_data = self.people.get_agents_data()
            committed_ids = self.seed.choice(
                agents_data["id"],
                number_of_committed_agents,
                replace=False,
            ).tolist()
            committed_indices = self.people.get_ind_from_multiple_agent_ids(
                committed_ids
            )
            public_data[behavior][committed_indices] = True

        else:
            public_data[behavior].fill(False)

    def _organize_cluster_isolation(self, cluster_intervention, public_data, time_step):
        """Organize cluster isolation (= cluster tracing).

        Trace cluster contacts of freshly diagnosed agents and send percentage of
        these clusters (=all members) into quarantine (according to intervention input).
        A cluster is the whole group belonging to a network where one of the freshly
        diagnosed agents was in.

        Args:
            cluster_intervention (ClassIntervention): Pointer to intervention object.
            public_data (dict[str, NumPyArray]): Public data of all agents.
            time_step (int): current time_step
        """
        # extract data
        if cluster_intervention.is_activated():
            my_sources = self._get_freshly_diagnosed_agents(public_data, time_step)
            if not my_sources.size:
                return
            array_of_tuples = self._get_network_tuples_from_agents(
                my_sources, "public localized"
            )
            array_of_tuple_choice = self._sample_percentage_of_array_elements(
                cluster_intervention, array_of_tuples
            )
            target_list = self._trace_agent_contacts_from_tuples(
                array_of_tuple_choice, cs.CLUSTER
            )
            self._do_quarantine_people(target_list, public_data, time_step)
        else:
            if cluster_intervention.status_activated_changed():
                self._trace[IS_ON] -= 1

    def _activate_deactivate_isolation(self, iso_intervention):
        """Organize isolation by setting constants in the _isolate dict.

        Args:
            iso_intervention (ClassIntervention): Pointer to intervention object.
        """

        if iso_intervention.is_activated():
            self._isolate[IS_ON] = True
            self._isolate[VALUE] = iso_intervention.get_value() / 100
        else:
            self._isolate[IS_ON] = False
            self._isolate[VALUE] = 0

    def _do_diagnose_people(self, public_data, epi_data, time_step):
        """Determine new "diagnosed"/"feel bad", set status and date_{status}
        in public_data.

        Decide who to diagnose and who to set "feel bad" according to current epidemic status. "Feel bad" indicates
        people that apart from the symptoms feel sick. It leaves space to distinguish
        between symptomatic people that are able to move and those lying in bed. It will be further progressed
        in the isolation method. Secondly this field becomes relevant for diseases other than COVID-19. It has the
        potential to be placeholder and be plotted as "diagnosed" at the end of the simulation.
        1. People will be diagnosed, if:
            - they have severe/critical symptoms or
            - they have a save positive test result.
            This is leaned on the German practice during the COVID-19 pandemic.
        2. People will be set "feel bad", if
            - they have severe/critical symptoms or
            - they have mild symptoms and are sampled from the pool according to a defined percentage.
            The percentage of mild that will feel sick can be defined by the user in the scenario data.

        Args:
            public_data (dict[str, array[int/bool]]): Data in people class that hold public status for each agent.
            epi_data (dict[str, array[int/bool]]): Data in people class that hold epidemic status for each agent.
            time_step (int): current time_step
        """
        # officially diagnose (only relevant for SarsCov2)
        is_severe = epi_data[f"date_{cs.SEVERE}"] == time_step
        is_critical = epi_data[f"date_{cs.CRITICAL}"] == time_step
        is_un_diagnosed = public_data[cs.DIAGNOSED] == 0
        is_save_positive = public_data[cs.POS_TESTED] > 1
        ind_to_diagnose = np.where(
            (is_severe | is_critical | is_save_positive) & is_un_diagnosed
        )[0]
        public_data[cs.POS_TESTED][is_save_positive] = 0
        set_peoples_status(ind_to_diagnose, public_data, cs.DIAGNOSED, True, time_step)
        # set status feel bad (could be alternative measure for diagnosed)
        is_mild = epi_data[f"date_{cs.MILD}"] == time_step
        ind_new_mild = np.where(is_mild)[0]
        # percentage of people getting mild symptoms that also feel bad.
        mild_feel_bad = Sync.INTERVENTION_USER_INPUT[cs.DIAGNOSED][
            cs.PERCENTAGE_MILD_FEEL_BAD
        ]
        num_mild_feel_bad = round(mild_feel_bad * len(ind_new_mild))
        ind_mild_feel_bad = self.seed.choice(
            ind_new_mild, num_mild_feel_bad, replace=False
        )
        ind_new_severe = np.where(is_severe | is_critical)[0]
        ind_new_feel_bad = np.hstack((ind_new_severe, ind_mild_feel_bad))
        set_peoples_status(ind_new_feel_bad, public_data, cs.FEEL_BAD, 1, time_step)

    def _do_un_diagnose_people(self, public_data, epi_data, time_step):
        """Determine who to offset "diagnosed"/"feel bad", set status and
        date_{status} in public_data.

        The conditions to be offset diagnosed are:
            - not having symptoms and
            - considered duration of disease is over
        The conditions to be offset "feel bad" are:
            - not having symptoms and
            - been set "feel bad" in the past

        Args:
            public_data (dict[str, array[int/bool]]): Data in people class that hold public status for each agent.
            epi_data (dict[str, array[int/bool]]): Data in people class that hold epidemic status for each agent.
            time_step (int): current time_step
        """
        cond1 = (
            public_data[f"date_{cs.DIAGNOSED}"]
            <= time_step - Sync.INTERVENTION_USER_INPUT[cs.DIAGNOSED][cs.PERIOD]
        )
        cond2 = epi_data[cs.SYMPTOMATIC] == 0
        cond3 = public_data[cs.FEEL_BAD] == 1
        cond_un_diagnose = cond1 & cond2
        set_peoples_status(cond_un_diagnose, public_data, cs.DIAGNOSED, False, np.nan)
        cond_un_feel_bad = cond2 & cond3
        set_peoples_status(cond_un_feel_bad, public_data, cs.FEEL_BAD, False, np.nan)

    def _do_hospitalize_people(self, public_data, epi_data, time_step):
        """Determine new hospitalized, set status and date_hospitalized in
        public_data.

        Decide who to hospitalize according to current epidemic status.
        There is a hierarchy between hospitalized, isolated and quarantined. Quarantined and isolated people lose their
        status to false and switch to being hospitalized.

        Args:
            public_data (dict[str, array[int/bool]]): Data in people class that hold public status for each agent.
            epi_data (dict[str, array[int/bool]]): Data in people class that hold epidemic status for each agent.
            time_step (int): current time_step
        """
        ids = self.people.get_data_for("id")
        # TODO: _check_for_vacant_hospital_beds
        occupied_hospital_beds = np.count_nonzero(public_data[cs.HOSPITALIZED])
        vacant_hospital_beds = self.hospital_bed_number - occupied_hospital_beds
        if vacant_hospital_beds > 0:
            cond1 = epi_data[cs.SEVERE]
            cond2 = epi_data[cs.CRITICAL]
            cond3 = np.invert(public_data[cs.HOSPITALIZED])
            indices_without_hospital_bed = np.where((cond1 | cond2) & cond3)[0]
            if vacant_hospital_beds < indices_without_hospital_bed.size:
                to_hospitalize_indices = self.seed.choice(
                    indices_without_hospital_bed, vacant_hospital_beds, replace=False
                )
            else:
                to_hospitalize_indices = indices_without_hospital_bed
            set_peoples_status(
                to_hospitalize_indices, public_data, cs.HOSPITALIZED, True, time_step
            )
            set_peoples_status(
                to_hospitalize_indices, public_data, cs.ISOLATED, False, np.nan
            )
            set_peoples_status(
                to_hospitalize_indices, public_data, cs.QUARANTINED, False, np.nan
            )
            cond4 = public_data[cs.QUARANTINED][to_hospitalize_indices]
            ind_not_quarantined = to_hospitalize_indices[~cond4]
            self.net_manager.remove_quarantined_from_networks(
                self, ids[ind_not_quarantined], households=True
            )

    def _do_un_hospitalize_people(self, public_data, epi_data, time_step):
        """Determine who to offset hospitalized, set status and date_{status}
        in public_data.

        The conditions to be offset hospitalized are:
            - currently hospitalized
            - being recovered or
            - being dead

        Args:
            public_data (dict[str, array[int/bool]]): Data in people class that hold public status for each agent.
            epi_data (dict[str, array[int/bool]]): Data in people class that hold epidemic status for each agent.
            time_step (int): current time_step
        """
        # Todo: add hospital
        ids = self.people.get_data_for("id")
        cond1 = epi_data[f"date_{cs.RECOVERED}"] == time_step
        cond2 = epi_data[f"date_{cs.DEAD}"] == time_step
        cond3 = public_data[cs.HOSPITALIZED] == 1
        to_un_hospitalize_indices = np.where((cond1 | cond2) & cond3)
        back_to_network_ids = ids[cond1 & cond3]
        set_peoples_status(
            to_un_hospitalize_indices, public_data, cs.HOSPITALIZED, False, np.nan
        )
        self.net_manager.add_quarantined_back_to_networks(
            self, back_to_network_ids, households=True
        )

    def _do_quarantine_people(self, agent_ids, public_data, time_step):
        """Quarantine agent ids given to the method.

        Remove agent ids that have already been quarantined before. For the residual
        change status in public data (from ClassPeople) and advice ClassNetworkManager
        object to remove quarantined agents from their networks.

        Args:
            agent_ids (list[int]): The ids of agents to be quarantined.
            public_data (dict[str, NumPyArray]): Public data of all agents.
            time_step (int): current time_step
        """
        ids = self.people.get_data_for("id")
        agent_ids_filtered = self._drop_members_already_quarantined(
            ids, agent_ids, public_data
        )
        indices = self.people.get_ind_from_multiple_agent_ids(agent_ids_filtered)
        set_peoples_status(indices, public_data, cs.QUARANTINED, True, time_step)
        self.net_manager.remove_quarantined_from_networks(
            self, agent_ids_filtered, households=True
        )

    def _do_un_quarantine_people(self, public_data, time_step):
        """Un-quarantine agents that satisfy the conditions.

        Change status in public data (from ClassPeople) and advice ClassNetworkManager
        object to add quarantined agents back to their networks.
        The conditions to offset quarantined are:
            - either time has run out
            - or save negative tested

        Args:
            public_data (dict[str, NumPyArray]): Public data of all agents.
            time_step (int): current time_step
        """
        if public_data[cs.QUARANTINED].sum() == 0:
            return
        ids = self.people.get_data_for("id")
        cond1 = (
            public_data[f"date_{cs.QUARANTINED}"]
            <= time_step - self._quarantine[PERIOD]
        )
        cond2 = public_data[cs.NEG_TESTED] == 2
        ind_to_un_quarantine = np.where(cond1 | cond2)[0]
        set_peoples_status(
            ind_to_un_quarantine, public_data, cs.QUARANTINED, False, np.nan
        )
        self.net_manager.add_quarantined_back_to_networks(
            self, ids[ind_to_un_quarantine], households=True
        )
        public_data[cs.NEG_TESTED][cond2] = 0

    def _do_isolate_people(self, public_data, time_step):
        """Determine new isolated, set status and date_isolated in public_data.

        Decide who to isolate according to public status.
        1. People will isolate themselves voluntarily, if:
            - they feel bad
            - not yet isolated or hospitalized
            ATTENTION: this is independent of governmental intervention
        2. People will be forced to isolate, if:
            - they are diagnosed or positive tested.
            - not yet isolated or hospitalized
            - sampled by the algorithm according to the percentage given in the dict value
            ATTENTION: this is due to governmental intervention
        There is a hierarchy between isolated and quarantined. Quarantined people lose their
        status of being quarantined and switch to being isolated.

        Args:
            public_data (dict[str, NumPyArray]): Data in people class that hold public status for each agent.
            time_step (int): current time_step
        """
        ids = self.people.get_data_for("id")
        cond1 = public_data[cs.HOSPITALIZED] == 0
        cond2 = public_data[cs.ISOLATED] == 0
        cond3 = public_data[cs.POS_TESTED] > 0
        if time_step == 0:
            cond4 = public_data[f"date_{cs.DIAGNOSED}"] <= time_step
            cond5 = public_data[f"date_{cs.FEEL_BAD}"] <= time_step
        else:
            cond4 = public_data[f"date_{cs.DIAGNOSED}"] == time_step
            cond5 = public_data[f"date_{cs.FEEL_BAD}"] == time_step
        if self._isolate[IS_ON]:
            ind_with_criteria_to_isolate = np.where(cond1 & cond2 & (cond3 | cond4))[0]
            num_to_force_isolate = round(
                ind_with_criteria_to_isolate.size * self._isolate[VALUE]
            )
            ind_to_force_isolate = self.seed.choice(
                ind_with_criteria_to_isolate, num_to_force_isolate, replace=False
            )
        else:
            ind_to_force_isolate = np.arange(0)
        ind_feel_bad = np.where(cond1 & cond2 & cond5)[0]
        ind_to_isolate = np.unique(np.hstack((ind_to_force_isolate, ind_feel_bad)))
        if not ind_to_isolate.size:
            return
        ind_not_quarantined = ind_to_isolate[
            public_data[cs.QUARANTINED][ind_to_isolate] == 0
        ]
        set_peoples_status(ind_to_isolate, public_data, cs.ISOLATED, True, time_step)
        set_peoples_status(ind_to_isolate, public_data, cs.QUARANTINED, False, np.nan)
        self.net_manager.remove_quarantined_from_networks(
            self, ids[ind_not_quarantined], households=True
        )

    def _do_de_isolate_people(self, public_data):
        """Determine offset isolated, set status and date_isolated in
        public_data.

        The conditions to offset isolated are:
            - not yet positive tested
            - not diagnosed
            - not feel bad

        Args:
            public_data (dict[str, NumPyArray]): Public data of all agents.
        """
        ids = self.people.get_data_for("id")
        cond1 = public_data[cs.POS_TESTED] == 0
        cond2 = public_data[cs.DIAGNOSED] == 0
        cond3 = public_data[cs.FEEL_BAD] == 0
        cond4 = public_data[cs.ISOLATED] == 1
        ind_to_de_isolate = np.where(cond1 & cond2 & cond3 & cond4)
        set_peoples_status(ind_to_de_isolate, public_data, cs.ISOLATED, False, np.nan)
        self.net_manager.add_quarantined_back_to_networks(
            self, ids[ind_to_de_isolate], households=True
        )

    def _do_receive_test_results(self, public_data, epi_data, time_step):
        """Check test results and set status in public data.

        Check conditions for having negative or positive test results and shuffle random numbers.
        According to these set status in public data (from ClassPeople). The field 'pos_tested'
        is set to 1 for indices with positive 'rapid test' (2 for 'laboratory test'). The latter will
        be reset by another method after progressing. The former indices are tested again with
        'laboratory test'. The 1 in the field 'pos_tested' stays for a couple of time_steps until the
        new test result is either negative ('pos_tested' then set to 0) or positive (set to 2). The field
        'neg_tested' is set to 1 for indices with negative results. The inverted indices are set 0 and
        therefore reset in any time step.

        Args:
            public_data (dict[str, NumPyArray]): Public data of all agents.
            epi_data (dict[str, NumPyArray]): Epidemic data of all agents.
            time_step (int): current time_step
        """
        if public_data[cs.TESTED].sum() == 0:
            return
        has_results = public_data[f"date_{cs.TEST_RESULTS}"] == time_step
        is_infected = epi_data[cs.INFECTED]
        is_dead = epi_data[cs.DEAD]
        is_un_detectable = epi_data[f"date_{cs.EXPOSED}"] <= time_step - 2
        is_severe = epi_data[cs.SEVERE]
        is_critical = epi_data[cs.CRITICAL]
        # set random numbers to decide True/False from sensitivity
        my_random = self.seed.integers(
            0, 101, len(public_data[f"date_{cs.TEST_RESULTS}"])
        )
        is_sensitive = public_data[cs.SENSITIVITY] - my_random > 0
        is_unspecific = public_data[cs.SPECIFICITY] - my_random <= 0
        is_save = public_data[cs.LABORATORY_TEST] == 1
        positive_cond = (
            (is_infected & ~is_un_detectable & is_sensitive)
            | is_severe
            | is_critical
            | (~is_infected & ~is_dead & is_unspecific)
        )
        is_positive = has_results & positive_cond
        is_negative = has_results & ~positive_cond & ~is_dead
        assert np.count_nonzero(has_results) == np.count_nonzero(
            is_positive
        ) + np.count_nonzero(is_negative) + np.count_nonzero(is_dead & has_results)
        # reset base data
        public_data[cs.TESTED][has_results] = False
        public_data[f"date_{cs.TESTED}"][has_results] = np.nan
        public_data[f"date_{cs.TEST_RESULTS}"][has_results] = np.nan
        # proceed positive
        public_data[cs.POS_TESTED][is_positive & is_save] = 2
        public_data[cs.POS_TESTED][is_positive & ~is_save] = 1
        to_test_again_indices = np.where(is_positive & ~is_save)
        self._implement_testing(
            to_test_again_indices, public_data, time_step, cs.LABORATORY_TEST
        )
        # proceed negative
        public_data[cs.NEG_TESTED][is_negative & is_save] = 2
        public_data[cs.NEG_TESTED][is_negative & ~is_save] = 1
        # reset data
        public_data[cs.NEG_TESTED][~is_negative] = 0
        public_data[cs.POS_TESTED][is_negative & is_save] = 0

    def _drop_members_already_quarantined(self, ids, agent_ids, public_data):
        """Filter those not quarantined/isolated/hospitalized from given ids.

        Args:
            ids (list[int]): Ids of all agents.
            agent_ids (list[int]): Ids of agents to filter from.
            public_data (dict[str, NumPyArray]): Public data of all agents.
        Returns:
            list_filtered_ids (list[int]): Contains filtered agent_ids.
        """
        # drop members already isolated/quarantined/hospitalized to not overwrite date
        cond1 = public_data[cs.QUARANTINED] == 0
        cond2 = public_data[cs.ISOLATED] == 0
        cond3 = public_data[cs.HOSPITALIZED] == 0
        cond4 = np.isin(ids, agent_ids)
        filtered_ids = ids[cond1 & cond2 & cond3 & cond4]
        list_filtered_ids = list(filtered_ids)
        return list_filtered_ids

    def _get_network_tuples_from_agents(self, agent_ids, category):
        """Iterate agent_ids, get their network ids and built tuples of
        agent_id and net_id.

        Args:
            agent_ids (list[int]): Ids of agents to filter from.
            category (str): Determine network category to get net_ids from.
        Returns:
            array_of_all_tuples (NumPyArray[int]): 2D array containing pairs of agent_id and net_id.
        """
        list_of_all_tuples = []
        # collect targets and isolate clusters
        for agent_id in agent_ids:
            if category == cs.ALL:
                net_ids = self.net_manager.get_all_network_ids_from_agent(agent_id)
            elif category == "localized":
                net_ids = self.net_manager.get_localized_network_ids_from_agent(
                    agent_id
                )
            elif category == "public localized":
                net_ids = self.net_manager.get_localized_public_network_ids_from_agent(
                    agent_id
                )
            else:
                raise ValueError(
                    "This is not a valid category. Use 'public localized' or"
                    " 'localized' or 'all."
                )
            for net_id in net_ids:
                list_of_all_tuples.append((agent_id, net_id))
        array_of_all_tuples = np.array(list_of_all_tuples)
        return array_of_all_tuples

    def _trace_agent_contacts_from_tuples(self, network_tuples, contact_type):
        """Iterate network_tuples and store contacts in a list.

        Ensure contacts in the list are unique.

        Args:
            network_tuples (NumPyArray[int]): 2D array containing pairs of agent_id and net_id.
            contact_type (str): Determine type to trace (primary/cluster).
        Returns:
            target_list (list[int]): Contain all contacts in for of agent_ids.
        """
        target_list = []
        for agent_id, net_id in network_tuples:
            for time_step in self.trace_diary[net_id]:
                my_contacts = self._get_contacts_from_trace_diary(
                    agent_id, net_id, time_step, contact_type
                )
                if my_contacts:
                    target_list.append(my_contacts)
        # flatten list of lists
        target_list = functools.reduce(operator.iconcat, target_list, [])
        target_list = np.unique(target_list)
        return target_list

    def _get_contacts_from_trace_diary(self, agent_id, net_id, time_step, contact_type):
        """Load contacts from _trace_diary.

        Args:
            agent_id (int): id of an agent.
            net_id (int): id of a network.
            time_step (int): current time_step
            contact_type (str): Determine type to trace (primary/cluster).
        Returns:
            my_contacts (list[int]): Contain contacts from agent in network.
        """
        if (contact_type != cs.PRIMARY) & (contact_type != cs.CLUSTER):
            raise ValueError(
                f"This is not a valid contact type. Use {cs.PRIMARY} or {cs.CLUSTER}."
            )
        if agent_id in self.trace_diary[net_id][time_step]:
            my_contacts = self.trace_diary[net_id][time_step][agent_id][contact_type]
        else:
            my_contacts = np.arange(0)
        return my_contacts

    def _sample_percentage_of_array_elements(self, intervention, my_array):
        """Sample a percentage from array elements.

        Args:
            intervention (ClassIntervention): Pointer to intervention object.
            my_array (NumPyArray[int]): Contain elements to sample from.

        Returns:
            array_choice (NumPyArray[int]): Contain sampled elements.
        """
        percentage = intervention.get_value() / 100
        num = round(len(my_array) * percentage)
        array_choice = self.seed.choice(
            a=my_array, size=num, replace=False, shuffle=False
        )
        return array_choice

    def _get_freshly_diagnosed_agents(self, public_data, time_step):
        """Filter agent_ids which are freshly diagnosed.

        Args:
            public_data (dict[str, NumPyArray]): Public data of all agents.
            time_step (int): current time_step

        Returns:
            my_sources (NumPyArray[int]): Contain agent:ids freshly diagnosed.
        """
        ids = self.people.get_data_for("id")
        cond = (
            public_data[f"date_{cs.DIAGNOSED}"] + self._trace[TIME_DELAY] == time_step
        )
        my_sources = ids[cond]
        return my_sources

    def _implement_testing(self, indices, public_data, time_step, test_type):
        """Determine date to have results and set fields in public data.

        Args:
            indices (NumPyArray[int]): indices of agents to determine date for and set fields.
            public_data (dict[str, NumPyArray]): Public data of all agents.
            time_step (int): current time_step
            test_type (str): 'rapid' / 'laboratory'
        """
        if test_type == cs.LABORATORY_TEST:
            # no official reference used for data
            days_to_wait = [1, 2, 3, 4, 5]
            prob_days = np.array([10, 20, 40, 20, 10]) / 100
            date_results = (
                self.seed.choice(days_to_wait, len(indices), p=prob_days) + time_step
            )
            public_data[cs.LABORATORY_TEST][indices] = 1
        elif test_type == cs.RAPID_TEST:
            date_results = time_step
        else:
            raise ValueError("test type not defined")
        public_data[cs.TESTED][indices] = True
        public_data[f"date_{cs.TESTED}"][indices] = time_step
        public_data[f"date_{cs.TEST_RESULTS}"][indices] = date_results
        public_data[cs.SENSITIVITY][indices] = self._test_info[test_type][
            cs.SENSITIVITY
        ]
        public_data[cs.SPECIFICITY][indices] = self._test_info[test_type][
            cs.SPECIFICITY
        ]

    # @profile
    def regulate(self, time_step, date_of_time_step):
        """Regulate diagnose/hospitalize/interventions/isolate and update
        public data.

        Diagnose and hospitalize has to be done at the beginning, because the interventions
        built on that information. Isolation has to be done last, because it builds on both:
        diagnose/hospitalize/testing.

        Args:
            time_step (int): current time_step
            date_of_time_step (datetime.date): Date of time step.
        """

        # take public data from people class
        if time_step > 0:
            public_data = self.people.get_public_data()
            epi_data = self.people.get_epidemic_data()
            self._do_diagnose_people(public_data, epi_data, time_step)
            self._do_un_diagnose_people(public_data, epi_data, time_step)
            self._do_hospitalize_people(public_data, epi_data, time_step)
            self._do_un_hospitalize_people(public_data, epi_data, time_step)
            self._perform_one_time_interventions(public_data, date_of_time_step)
            self._perform_cyclic_interventions(
                public_data, epi_data, time_step, date_of_time_step
            )
            self._do_isolate_people(public_data, time_step)
            self._do_de_isolate_people(public_data)
            self.people.update_public_data(self, public_data)


def set_peoples_status(my_ind, public_data, status, value, time_step):
    """Set status of given indices to value and date_{status} to time_step in
    public data.

    Args:
        my_ind (NumPyArray[int]): The indices of agents to set status for.
        public_data (dict[str, NumPyArray]): Public data of all agents.
        status (str): name of the field to set value for
        value (int/bool): value to set
        time_step (int): current time_step
    """
    public_data[f"{status}"][my_ind] = value
    public_data[f"date_{status}"][my_ind] = time_step
