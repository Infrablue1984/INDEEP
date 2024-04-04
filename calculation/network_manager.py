#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Manage all networks and it's members.

Typical usage example:

net_manager = ClassNetworkManager()
primary_contacts = net_manager.get_primary_contacts_from_agent_in_network()
"""

__author__ = "Inga Franzen"
__created__ = "2020"
__date_modified__ = "2023/05/31"
__version__ = "1.0"

import functools
import operator
import time
from copy import deepcopy as deep

import numpy as np
from line_profiler_pycharm import profile
from numpy import random as rm

from calculation import networks
from synchronizer import constants as cs
from synchronizer.synchronizer import FileLoader
from synchronizer.synchronizer import Synchronizer as Sync


class NetworkManager:
    """Organize objects of type ClassNetwork and manage calls from other
    classes.

    Sort ClassNetwork objects in a dictionary to ensure easy access. The dictionary is 3-times nested:
    [digit][region][ClassNetwork id]. The digits stand for a specific network type:
        1: "households",
        2: "workplaces",
        3: "kitas",
        4: "schools",
        5: "unis",
        6: "activities",
        7: "geographical".
    Communicate with other classes and encapsulate ClassNetwork. Other classes can give
    network id and NetworkManager searches proper results. Also organize interventions that effect ClassNetwork.

    Attributes:
        self.regions (array[int]): 1000 -17000, standing for german region.
        self.scale (int): Describe the ratio between number of agents in model and reality.
        self.people (ClassPeople): Pointer to ClassPeople object.
        self.weekday (int): from 0 to 7, means Mo - Su.
        self.change_dict (dict[int, ClassNetwork]): Track ClassNetwork objects that change within time step.
        self.agents_network_ids (dict[int, array[int]]): Holding network ids from each agent, where he/she participates.
        self.networks (dict[int, dict[int, dict[int, ClassNetwork]]]): Organizing ClassNetwork objects.
        self.parent_ids (dict[int, dict[int, array[int]]): Organizing parent ids (e.g. school id for multiple classes)
        self.event_limit (boolean): Determine, if intervention limit events is on or off
    """

    def __init__(self, people, regions, scale, start_date, seed):
        """Initialize ClassNetworkManager.

        Set variables and load existing data for network structures from file.
        Initialize ClassNetwork objects and assign to organized dictionary.

        Args:
            people (ClassPeople): Pointer to ClassPeople object.
            regions (array[int]): 1000 -17000, standing for german region.
            scale (int): Describe the ratio between number of agents in model and reality.
            start_date (datetime.date): Date to start simulation from.
        """
        self.seed = seed
        self.regions = regions
        self.scale = scale
        self.people = people
        self.weekday = start_date.weekday()
        self.time_step = 0
        self._change_dict = {}
        self.agents_network_ids = FileLoader.load_agents_networks(regions, scale)
        self.networks = {}
        self.parent_ids = {}
        self.event_limit = {key: 10000 for key in Sync.PUBLIC_DIGITS}
        self._create_networks()

    def _create_networks(self):
        """Set data for self.networks and self.parent_ids."""
        for digit in [1, 2, 3, 4, 5, 6]:
            p1, p2 = self._create_institutional_networks(digit)
            self.networks[digit] = p1
            self.parent_ids[digit] = p2
        self.networks[7] = self._create_geographical_networks()

    def _create_institutional_networks(self, digit):
        """Load data from file, extract values and initialize ClassNetwork
        objects.

        Args:
            digit (int): digit standing for network type.

        Returns:
            locations (dict[int, dict[int, ClassNetwork]]): Organized dict for one network type.
            parent_ids (dict[int, array[int]]): Organized dict for one network type.
        """

        parent_ids = {}
        locations = {}
        df_location = FileLoader.load_network_data(digit, scale=self.scale)
        members_dict = FileLoader.load_members(digit, self.scale)
        for region in self.regions:
            filt = df_location["region"] == region
            region_arr = df_location.loc[
                filt,
            ].to_numpy()
            locations[region] = {}
            if digit in range(2, 6):
                parent_ids[region] = np.unique(region_arr[:, 0])
            for idx in range(len(region_arr)):
                id = region_arr[idx, 1]
                info = region_arr[idx]
                members = members_dict[id]
                network = self._instantiate_network_object(digit, members, info)  # type: ignore
                locations[region][id] = network
                self._add_to_change_dict(id, network)
        return locations, parent_ids

    def _instantiate_network_object(self, digit, members, local_info):
        """Create instances of ClassNetwork objects for localized areas.

        Args:
            digit (int): digit standing for network type.
            members (array[int]): Hold ids of agents.
            local_info (array[int]): Hold information on one network.
        """
        members = list(members)
        if digit == 1:
            return networks.Household(members, local_info, self.seed)
        if digit == 2:
            return networks.WorkPlace(members, local_info, self.seed)
        if digit == 3:
            return networks.KitaPlace(members, local_info, self.seed)
        if digit == 4:
            return networks.SchoolPlace(members, local_info, self.seed)
        if digit == 5:
            return networks.UniPlace(members, local_info, self.seed)
        if digit == 6:
            return networks.Activity(members, local_info, self.seed)
        else:
            BaseException("The given net type is not valid")

    def _create_geographical_networks(self):
        """Create instances of ClassNetwork objects for geographic areas.

        Extract information from people class and filter agent ids by geographic area.
        1. There is one geographic area for each region.
        2. If multiple regions are chosen, there is another geographic area for the federal state of these regions.
        3. If regions from multiple federal states are chosen, there is another geographic area for agents from
        all regions.
        The method determines a percentage of people from each geographic area that participates in geographic movement.
        The other people are assumed to only interact in localized areas. The percentage always adds up to 35 %.
        The members will be shuffled each time step.

        Returns:
            random_networks (dict[int, dict[int, ClassNetwork]]): Organized dict for geographic networks.
        """

        random_networks = {}
        base_id = 7 * 10**5
        z_codes = self.people.get_data_for(cs.Z_CODE)
        ids = self.people.get_data_for("id")
        assert len(z_codes) == len(ids)
        my_states = (self.regions / 1000).astype(
            int
        ) * 10**3  # calculate state from z_code
        my_states = np.unique(my_states, axis=0)  # remove duplicates
        # 1. organize contacts on regional level
        for region in self.regions:
            if len(self.regions) == 1:
                p_pop1 = 0.35
            elif (region == 2000) | (region == 11000):  # region equals state
                p_pop1 = 0.30
            else:
                p_pop1 = 0.20
            random_networks[region] = {}
            my_id = base_id + region
            my_agents = ids[z_codes == region]
            for agent in my_agents:
                self.agents_network_ids[agent].append(my_id)
            network = networks.GeographicArea(my_agents, my_id, p_pop1, self.seed)
            random_networks[region][my_id] = network
            self._add_to_change_dict(my_id, network)
        # 2. organize contacts on level of federal states
        if len(self.regions) > 1:  # at least 2 regions
            if len(my_states) > 1:  # at least 2 states
                p_pop2 = 0.1
            else:
                p_pop2 = 0.15
            for state in my_states:  # iterate federal states
                if (state == 2000) or (state == 11000):  # region equals state
                    continue
                random_networks[state] = {}
                my_id = base_id + state
                my_agents = ids[(z_codes / 1000).astype(int) * 10**3 == state]
                for agent in my_agents:
                    self.agents_network_ids[agent].append(my_id)
                # assume 10 % people have contacts from other region
                network = networks.GeographicArea(my_agents, my_id, p_pop2, self.seed)
                random_networks[state][my_id] = network
                self._add_to_change_dict(my_id, network)
            # 3. organize contacts on all levels
            if len(my_states) > 1:  # at least 2 different states
                p_pop3 = 0.05  # assume 5 % people have contacts from other state
                random_networks[0] = {}
                for agent in ids:
                    self.agents_network_ids[agent].append(base_id)
                network = networks.GeographicArea(ids, base_id, p_pop3, self.seed)
                random_networks[0][base_id] = network
                self._add_to_change_dict(base_id, network)
        return random_networks

    def _search_network_object(self, net_id):
        """Search network object for net_id.

        Args:
            net_id (int):

        Returns:
            network (ClassNetwork): pointer to network object.
        """
        region = Sync.get_region_from_net_id(net_id)
        digit = Sync.get_first_digit(net_id)
        if net_id not in self.networks[digit][region]:
            raise ValueError(f"{net_id} is not the id of a location")
        else:
            network = self.networks[digit][region][net_id]
        return network

    def organize_home_doing(self, client, intervention, digit):
        """Organize home doing for the given network type (digit).

        Take all members from network type and select a percentage according
        to intervention value. Iterate network objects and trigger to remove
        the members from networks. Add network to change_dict only if change
        has taken place. If intervention is not activated do the opposite,
        because this is the case, when the intervention ends.

        Args:
            client (ClassX): type of the class calling the method.
            intervention (ClassIntervention): Pointer to intervention object.
            digit (int): stand for network type: 2: "workplaces", 3: "kitas", 4: "schools", 5: "unis", 6: "activities".
        """
        if type(client).__name__ != "PublicHealthRegulator":
            raise AssertionError(
                "Only the Public Health Regulator is allowed to trigger home doing."
            )
        assert digit in Sync.PUBLIC_DIGITS
        # case one: the home doing starts
        if intervention.is_activated():
            percentage = intervention.get_value() / 100
            for region in self.regions:
                my_networks = self.networks[digit][region]
                all_members = [
                    network.get_original_members() for network in my_networks.values()
                ]
                all_members = np.unique(np.concatenate(all_members, axis=0))
                num = round(len(all_members) * percentage)
                my_agents = self.seed.choice(all_members, num, replace=False)
                for net_id, network in my_networks.items():
                    if network.do_send_home(my_agents):
                        self._add_to_change_dict(net_id, network)

        # case two: the home doing finish and people are send back to work
        else:
            for region in self.regions:
                for net_id, network in self.networks[digit][region].items():
                    if network.do_un_send_home():
                        self._add_to_change_dict(net_id, network)

    def organize_shut_institution(self, client, intervention, digit):
        """Organize shut institution for the given network type (digit).

        Take all institutions from network type and select a percentage according to
        intervention value. Iterate network objects and trigger closing. Institution
        is superior and comprises multiple ClassNetwork objects, and contains e.g.
        school for classes/free/mensa or work for office/meeting/mensa. Activities
        are an exception, they are both ClassNetwork object and institution. If
        intervention is not activated do the opposite, because this is the case,
        when the intervention ends. Add network to change_dict only if change
        has taken place.

        Args:
            client (ClassX): type of the class calling the method.
            intervention (ClassIntervention): Pointer to intervention object.
            digit (int): stand for network type: 2: "workplaces", 3: "kitas", 4: "schools", 5: "unis", 6: "activities".
        """
        if type(client).__name__ != "PublicHealthRegulator":
            raise AssertionError(
                "Only the Public Health Regulator is allowed to trigger the method."
            )
        assert digit in Sync.PUBLIC_DIGITS
        # case one: intervention starts
        if intervention.is_activated():
            percentage = intervention.get_value() / 100
            for region in self.regions:
                # sample networks without parent institution --> sample from ids
                if digit == 6:
                    ids = list(self.networks[digit][region].keys())
                    parent = False
                # sample networks with parent institution --> sample from parent_ids
                else:
                    # sample networks with parent institution
                    ids = self.parent_ids[digit][region]
                    parent = True
                num = round(len(ids) * percentage)
                sample = self.seed.choice(ids, num, replace=False)
                for net_id, network in self.networks[digit][region].items():
                    if network.get_net_id(parent=parent) in sample:
                        network.do_shut_location()
                        self._add_to_change_dict(net_id, network)

        # case two: intervention stops
        else:
            for region in self.regions:
                for net_id, network in self.networks[digit][region].items():
                    if network.is_shut():
                        network.do_open_location()
                        self._add_to_change_dict(net_id, network)

    def organize_limit_event_size(self, client, intervention, digit):
        """Organize limit event size for the given network type (digit).

        Iterate network objects and trigger limit event size. The network object
        will decide internally if and what to do. If intervention is not activated
        do the opposite, because this is the case, when the intervention ends. Add
        network to change_dict only if change has taken place.

        Args:
            client (ClassX): type of the class calling the method.
            intervention (ClassIntervention): Pointer to intervention object.
            digit (int): stand for network type: 2: "workplaces", 3: "kitas", 4: "schools", 5: "unis", 6: "activities".
        """
        if type(client).__name__ != "PublicHealthRegulator":
            raise AssertionError(
                "Only the Public Health Regulator is allowed to trigger the method."
            )
        assert digit in Sync.PUBLIC_DIGITS
        # case one: the limitation of event size starts
        if intervention.is_activated():
            my_size = intervention.get_value()
            self.event_limit[digit] = my_size
            for region in self.regions:
                for net_id, network in self.networks[digit][region].items():
                    if network.do_limit_events(my_size):
                        self._add_to_change_dict(net_id, network)
        # case two: the limitation of event size finishes
        else:
            self.event_limit[digit] = 10000
            for region in self.regions:
                for net_id, network in self.networks[digit][region].items():
                    if network.do_un_limit_event_size():
                        self._add_to_change_dict(net_id, network)

    def organize_split_groups(self, client, intervention, digit):
        """Organize split groups for the given network type (digit).

        Iterate network objects and trigger split groups. The network object
        will decide internally if and what to do. If intervention is not activated
        do the opposite, because this is the case, when the intervention ends. Add
        network to change_dict only if change has taken place.

        Args:
            client (ClassX): type of the class calling the method.
            intervention (ClassIntervention): Pointer to intervention object.
            digit (int): stand for network type: 2: "workplaces", 3: "kitas", 4: "schools", 5: "unis", 6: "activities".
        """
        if type(client).__name__ != "PublicHealthRegulator":
            raise AssertionError(
                "Only the Public Health Regulator is allowed to regulate quarantine."
            )
        assert digit in Sync.PUBLIC_DIGITS
        # case one: the splitting of groups starts
        if intervention.is_activated():
            degree = intervention.get_value()  # degree can be 2 or 4
            for region in self.regions:
                for net_id, network in self.networks[digit][region].items():
                    if network.do_split_location(degree):
                        self._add_to_change_dict(net_id, network)
        # case two: the limitation of event size finishes
        else:
            for region in self.regions:
                for net_id, network in self.networks[digit][region].items():
                    if network.do_union_location():
                        self._add_to_change_dict(net_id, network)

    def _add_to_change_dict(self, net_id, network):
        self._change_dict[net_id] = network

    def _shuffle_active_and_passive_members(self, digit):
        """Iterate network objects, check conditions and trigger shuffle.

        Meant for intervention limit event size. Active and passive members shall occasionally
        be exchanged. The network object will decide internally if and what to do.

        Args:
            digit (int): stand for network type: 2: "workplaces", 3: "kitas", 4: "schools", 5: "unis", 6: "activities".
        """
        # exchange active and passive
        for region in self.regions:
            for net_id, network in self.networks[digit][region].items():
                if network.is_localized() & network.check_weekday(self.weekday):
                    if network.do_shuffle_active_and_passive_members():
                        self._add_to_change_dict(net_id, network)

    # @profile
    def _shuffle_public_un_localized_networks(self, digit):
        """Iterate un-localized network objects and trigger shuffle.

        Meant for network objects such as corridors or school yards. Un-localized means there
        is no specific location/room. It only represents a network of face-to-face contacts,
        which shall occasionally be exchanged. The network object will decide internally if
        and what to do.

        Args:
            digit (int): stand for network type: 2: "workplaces", 3: "kitas", 4: "schools", 5: "unis", 6: "activities".
        """
        assert digit in Sync.PUBLIC_DIGITS
        for region in self.regions:
            un_localized = dict(
                filter(
                    lambda x: not x[1].is_localized(),
                    self.networks[digit][region].items(),
                )
            )
            for net_id, network in un_localized.items():
                if not network.is_small():
                    network.do_shuffle_contacts()
                    self._add_to_change_dict(net_id, network)

    def _shuffle_geographical_areas(self):
        """Iterate geographical areas and trigger shuffle.

        Geographical areas (digit 7) represent a supra-regional network
        of face-to-face contacts, which shall be regularly exchanged.
        """
        for area in self.networks[7]:
            for net_id, network in self.networks[7][area].items():
                network.do_shuffle_contacts()
                self._add_to_change_dict(net_id, network)

    def add_quarantined_back_to_networks(self, client, agent_ids, households=False):
        """Add agents back to each of their networks.

        Iterate agents and their networks and trigger un-quarantine. The network object
        will decide internally what to do. Add network to change_dict only if change
        has taken place.

        Args:
            client (ClassX): type of the class calling the method.
            agent_ids (list[int]): List with ids of agents.
            households (boolean): if True, un-quarantine from households. If False, do not.
        """
        if type(client).__name__ != "PublicHealthRegulator":
            raise AssertionError(
                "Only the Public Health Regulator is allowed to regulate quarantine."
            )
        # release people from quarantine
        for agent in agent_ids:
            my_net_ids = self.agents_network_ids[agent]
            for net_id in my_net_ids:
                network = self._search_network_object(net_id)
                network_type = network.get_local_type()
                if (network_type == cs.LIVING_ROOM) & (not households):
                    continue
                else:
                    if network.do_un_quarantine_member(agent):
                        self._add_to_change_dict(net_id, network)

    def remove_quarantined_from_networks(self, client, agent_ids, households=False):
        """Remove agents from each of their networks.

        Iterate agents and their networks and trigger quarantine. The network object
        will decide internally what to do. Add network to change_dict only if change
        has taken place.

        Args:
            client (ClassX): type of the class calling the method.
            agent_ids (list[int]): List with ids of agents.
            households (boolean): if True, un-quarantine from households. If False, do not.
        """
        if type(client).__name__ != "PublicHealthRegulator":
            raise AssertionError(
                "Only the Public Health Regulator is allowed to regulate quarantine."
            )
        # quarantine people
        for agent in agent_ids:
            my_net_ids = self.agents_network_ids[agent]
            for net_id in my_net_ids:
                network = self._search_network_object(net_id)
                network_type = network.get_local_type()
                if (network_type == cs.LIVING_ROOM) & (not households):
                    continue
                else:
                    if network.do_quarantine_member(agent):
                        self._add_to_change_dict(net_id, network)

    # @profile
    def update_networks(self, client, time_step, date_of_time_step):
        """Update networks = shuffle members and contacts, where necessary.

        Update time_step and weekday. Iterate digits and shuffle un-localized
        networks such as corridors. Shuffle active and passive members in public
        locations, only if intervention LIMIT_EVENT_SIZE is active. Always shuffle
        geographical areas.

        Args:
            client (ClassX): type of the class calling the method.
            time_step (int): current time_step
            date_of_time_step (datetime.date): Date of time step.
        """
        if (type(client).__name__ != "SimRunner") and (
            type(client).__name__ != "TestNetworksCase1"
        ):
            raise AssertionError("Only the Simulator is allowed to enter that field.")
        self.time_step = time_step
        self.weekday = date_of_time_step.weekday()
        for digit in Sync.PUBLIC_DIGITS:
            if self.event_limit[digit] < 10000:
                self._shuffle_active_and_passive_members(
                    digit
                )  # exchange active and passive members
            self._shuffle_public_un_localized_networks(digit)
        self._shuffle_geographical_areas()

    # @profile
    def get_changed_networks(self):
        """Get copy of dictionary with network objects that changed within last
        time step.

        Returns:
            changed_dict (dict[int, ClassNetwork]): Hold net_ids (key) and network objects (value).
        """
        change_dict = self._change_dict
        self._change_dict = {}
        return change_dict

    def get_localized_network_ids_from_agent(self, agent_id):
        """Get all localized network ids from agent.

        Localized networks are those, that symbolize a room/location,
        e.g. households, offices, classrooms or activities.
        The network ids are constant during the simulation and independent of
        agents status within the network such as active/passive/quarantined.

        Args:
            agent_id (int): The id of one agent.

        Returns:
            my_locations (List[int]): The list with network ids from the agent.
        """
        net_ids = self.agents_network_ids[agent_id]
        my_locations = []
        for net_id in net_ids:
            network = self._search_network_object(net_id)
            if network.is_localized():
                my_locations.append(net_id)
        return my_locations

    def get_localized_public_network_ids_from_agent(self, agent_id):
        """Get all public network ids from agent.

        Localized public networks are those, that symbolize a room/location in a public area,
        e.g. offices, classrooms or activities (no households!).
        The network ids are constant during the simulation and independent of
        agents status within the network such as active/passive/quarantined.

        Args:
            agent_id (int): The id of one agent.

        Returns:
            my_locations (List[int]): The list with network ids from the agent.
        """
        net_ids = self.agents_network_ids[agent_id]
        my_locations = []
        for net_id in net_ids:
            network = self._search_network_object(net_id)
            if network.is_localized():
                my_locations.append(net_id)
        return my_locations

    def get_all_network_ids_from_agent(self, agent_id):
        """Get all network ids from agent.

        All network ids include:
            - localized networks (e.g. households, offices, classrooms or activities).
            - un-localized networks (unspecific 1:1 contacts between people).
        The network ids are constant during the simulation and independent of
        agents status within the network such as active/passive/quarantined.

        Args:
            agent_id (int): The id of one agent.

        Returns:
            my_locations (List[int]): The list with network ids from the agent.
        """
        my_locations = self.agents_network_ids[agent_id]
        return my_locations

    def get_primary_contacts_from_agent_in_network(self, net_id, agent_id):
        """Get 1:1 contacts from agent in network at current weekday.

        The agent's status within the network such as active/passive/quarantined
        will be considered for both agent and primary contacts.

        Args:
            net_id (int): The id of the network.
            agent_id (int): The id of one agent.

        Returns:
            contact_ids (List[int]): The list with agent_ids that are primary contact.
        """
        network = self._search_network_object(net_id)
        contact_ids = network.get_primary_contacts(agent_id, self.weekday)
        return contact_ids

    def get_members_from_localized_network(self, net_id, agent_id):
        """Get all active members within network at current weekday.

        This is to model aerosol transmission. The agent_id is needed due to possible
        group splitting. The network_id stays the same, but the subgroup can change.

        Args:
            net_id (int): The id of the network.
            agent_id (int): The id of one agent.

        Returns:
            cluster_members (List[int]): agent_ids from active cluster members.
        """
        network = self._search_network_object(net_id)
        if not network.is_localized():
            raise ValueError("This net_id is not localized")
        cluster_members = network.get_cluster_contacts(agent_id, self.weekday)
        return cluster_members

    def get_type_of_location(self, net_id):
        """Get type of net_id. E.g. office, class, living-room, un-specific.

        Args:
            net_id (int): id of specific network

        Returns:
            (str): Type of location.
        """
        network = self._search_network_object(net_id)
        info = network.get_local_type()
        return info

    def get_non_localized_network_ids_from_agent(self, agent_id):
        """UNUSED METHOD!!! Get all none-localized network ids from agent.

        None-localized are those, that do not symbolize a room/location.
        Instead, they symbolize unspecific 1:1 contacts between people.
        Each contact takes place at a different and unspecified place.
        The network ids are constant during the simulation and independent of
        agents status within the network such as active/passive/quarantined.

        Args:
            agent_id (int): The id of one agent.

        Returns:
            my_locations (List[int]): The list with network ids from the agent.
        """
        net_ids = self.agents_network_ids[agent_id]
        my_locations = []
        for net_id in net_ids:
            network = self._search_network_object(net_id)
            if not network.is_localized():
                my_locations.append(net_id)
        return my_locations

    def get_all_primary_contacts_from_one_agent(self, agent_id, digit=0):
        """ONLY USED INTERNALLY BY CONTROLLER!!!

        Get 1:1 contacts from agent from all of his networks at current weekday.
        The agent's status within the network such as active/passive/quarantined
        will be considered for both agent and primary contacts.

        Args:
            agent_id (int): The id of one agent.
            digit (int): digit standing for network type. If 0, all network types.

        Returns:
            my_primary (NumPyArray[int]): The array with agent_ids that are primary contact.
        """
        my_primary = []
        my_nets = self.agents_network_ids[agent_id]
        if digit != 0:
            my_nets = Sync.help_filter_network_ids_by_digit(my_nets, digit)
        for net_id in my_nets:
            net_primary = self.get_primary_contacts_from_agent_in_network(
                net_id, agent_id
            )
            my_primary.append(net_primary)
        my_primary = functools.reduce(operator.iconcat, my_primary, [])
        my_primary = np.unique(my_primary)
        return my_primary
