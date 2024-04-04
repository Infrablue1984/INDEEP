#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Create single network institutions such as a household, a school or a
classroom.

The Network class manages all the single networks. Every single network
is a class with attributes such as type, size_total, status open/closed,
the ids of its members and the information about members staying absent
voluntary or due to quarantine. It also has functions to change its own
status or handle the members inside.
"""

__author__ = "Inga Franzen"
__created__ = "2020"
__date_modified__ = "2022/06/06"
__version__ = "1.0"

import math
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
# from line_profiler_pycharm import profile
# from numpy import random as rm

from calculation import graphs as graphs
from synchronizer import constants as cs
from synchronizer.synchronizer import Synchronizer as Sync

warnings.simplefilter(action="ignore", category=FutureWarning)

# TODO: the graph function can be modified to create a graph from scratch.
#  \ It is possible to consider personal attributes such as extro-/introvert
#  \ and associate probabilities to connect. Also an attr time_spent=10min
#  \could be added. Or an attr "sports", "singing", "outdoor" can be
#  \ added to the graph. The method could be overwritten in any \
#  subclass to consider graph types.
#  \ It would be necessary to study contact pattern e.g. at school, at work \
#  in more detail. Also the method could grow more time consuming, \
#  if more complex.


class Location(metaclass=ABCMeta):
    """Locations can be rooms (localized) or bigger areas (un-localized).

    Locations hold information about themselves, such as unique id or type,
    but also information about their members and a reference to a ClassGraph,
    symbolizing the contact pattern. Locations need to be able to give the
    contacts of an agent during the time step. Therefore, they have to manage
    current status of all members, which can change due to interventions.

    Attributes:
        self.id (int): unique id of the location
        self.type (str): type of location, such as office or class.
        self.localized (boolean): True, if room based. False, if bigger area.
        self.net_info (dict[str, int]): Hold information, such as id or room volume.
    """

    def __init__(self, uid=0, my_type=cs.UNSPECIFIC):
        """Initialize basic properties of location.

        Args:
            uid (int): unique id of the location.
            my_type (str): type of location, such as office, class or unspecific.
        """
        self.id = uid
        self.type = my_type
        self.localized = self._help_set_localized()

    def _help_set_localized(self):
        """Set value for being localized.

        Returns:
            (boolean): True, if type is not unspecific. False, else.
        """
        if (self.type == cs.UNSPECIFIC) or (self.type == cs.GEOGRAPHIC):
            return False
        else:
            return True

    def get_local_type(self):
        """Get type of the location. E.g. office, class, living-room, un-
        specific.

        Returns:
            (str): Type of location.
        """
        return self.type

    @abstractmethod
    def get_original_members(self):
        """Get all members of the location, independent of its status.

        Returns:
            members (list[int]): list with agent_ids from location.
        """
        pass

    @abstractmethod
    def get_active_members(self):
        """Get active members of the location.

        Some members are registered at the location, but might not actively
        participate within the current time step.

        Returns:
            members (list[int]): list with active agent_ids from location.
        """
        pass

    @abstractmethod
    def get_status(self, agent_id):
        """UNUSED METHOD Get status from agent.
        Possible statuses can be:
            0 : active --> member does actively participate
            99 : passive --> member does not actively participate
        Other statuses possible to organize location.

        Returns:
            status (int): status from agent.
        """
        pass

    @abstractmethod
    def get_net_id(self, *args):
        """Get unique id of the location.

        Returns:
            id (int): unique id of location or superior institution.
        """
        pass

    @abstractmethod
    def check_weekday(self, weekday):
        """Check, if location takes place on this weekday.

        Args:
            weekday (int): 0-7 (Mo-Su)

        Returns:
            (boolean): True, if location takes place. False, else.
        """
        pass

    def is_localized(self):
        """Check, if location is localized.

        Localized means, the location represents a room. Un-localized means, the
        location represents a bigger area, such as a city, region, federal state,
        building corridor or yard.

        Returns:
            (boolean): True, if localized. False, else.
        """
        return self.localized

    @abstractmethod
    def get_primary_contacts(self, agent_id, weekday):
        """Get primary contacts from agent at specified weekday.

        Only consider active members and active contacts.

        Args:
            agent_id (int): id of agent to get contacts from.
            weekday (int): from 0 to 7, means Mo - Su.

        Returns:
            primary contacts (list[int]): List with agent_ids of primary contacts.
        """
        pass

    @abstractmethod
    def do_quarantine_member(self, agent_id):
        """Remove agent from location due to
        quarantine/isolation/hospitalization.

        Ensure agent is registered as quarantined and no longer appears in active
        members. The information will be used in the get methods.

        Args:
            agent_id (int): id of agent to be quarantined.

        Returns:
            (boolean): True, if agent was removed. False, else.
        """
        pass

    @abstractmethod
    def do_un_quarantine_member(self, agent_id):
        """Add agent back to location when quarantined.

        Ensure agent is registered back from quarantine. The information will be
        used in the get methods.

        Returns:
            (boolean): True, if agent is sent back. False, else.
        """
        pass


class Household(Location):
    """A Household symbolizes a room or a flat and is therefore always
    localized.

    Households are not affected by interventions such as LIMIT_EVENT_SIZE, SPLIT_GROUPS,
    HOME_DOING or being SHUT. They only have to manage QUARANTINE, ISOLATION and
    HOSPITALIZATION.

    Attributes:
        self.size_total (int): number of total members
        self.mean_contacts (int): number of mean contacts within a graph
        self.members_all (list[int]): list with agent_ids of all members from the location
        self.members_active (list[int]): list with members actively participating (e.g. not quarantined).
        self.agents_info (dict[int, dict[str, int]]): Dict to register status for each member.
        self.members_graph (ClassGraph): Symbolizing contact patterns (members = nodes, contacts = edges).
    """

    def __init__(self, members, local_info, seed):
        """Initialize properties of the location and fields to store
        information in.

        Important are the fields to handle information of the agent's status within
        the location. This is to manage quarantined, isolated or hospitalized. And also
        the contact pattern among agents.

        Args:
            members (list[int]): List with ids of agents that are part of the household.
            local_info (NumPyArray[int/str]: contains information on the location
                1 --> NET_ID = unique ID of location
                4 --> MEAN_CONTACTS
        """
        assert local_info[3] == len(members)
        super(Household, self).__init__(uid=local_info[1], my_type=cs.LIVING_ROOM)
        self.seed = seed
        self.size_total = len(members)
        self.mean_contacts = local_info[4]
        self.members_all = members
        self.members_active = members.copy()
        self.agents_info = self._initialize_status(members)
        self.members_graph = self._create_graph_from_all()
        assert len(self.members_all) == self.size_total

    def _initialize_status(self, members):
        """Create dict with info about status for each member.

        By default, the status is 0, which means active member.
        Possible statuses can be:
            0 : active --> member does actively participate
            97 : quarantined --> member is absent due to quarantine/isolation/hospitalization

        Args:
            members (list[int]): List with ids of agents.

        Returns:
            agents_info (dict[int, dict[str, int]]): info about each member.
        """
        agents_info = {}
        for member in members:
            agents_info[member] = {cs.STATUS: 0}
        return agents_info

    def _create_graph_from_all(self):
        """Create graph from all members in this location.

        If the network is small, which means every member has contact to
        every other member, create GraphSmall. Otherwise, create GraphMedium.
        The graph will stay constant with all members during the simulation.
        If a member gets passive, quarantined or so, this will be checked and
        filtered by the get methods.

        Returns:
            members_graph (ClassGraph): Graph with members as nodes and edges as contacts.
        """
        if self.size_total == self.mean_contacts + 1:
            members_graph = graphs.GraphSmall(
                self.members_all, self.mean_contacts, self.id
            )
        else:
            members_graph = graphs.GraphMedium(
                self.members_all, self.mean_contacts, self.id, self.seed
            )
        return members_graph

    def get_original_members(self):
        """Get all members of the location, independent of its status.

        Returns:
            self.members_all (list[int]): list with agent_ids from location.
        """
        return self.members_all

    def get_active_members(self):
        """Get all active members of the location.

        Active members are those not quarantined/isolated/hospitalized.

        Returns:
            active_members (list[int]): list with active agent_ids from location.
        """
        return self.members_active

    def get_first_digit(self):
        """Get digit standing for network type.

        Returns:
            digit (int): digit of network type.
        """
        return 1

    def get_status(self, agent_id):
        """UNUSED METHOD Get status from agent.

        0 : active --> member does actively participate
        97 : quarantined --> member is quarantined/isolated/hospitalized.

        Returns:
            status (int): status from agent.
        """
        return self.agents_info[agent_id][cs.STATUS]

    def get_net_id(self):
        """Get unique id of the location.

        Returns:
            id (int): unique id of location.
        """
        return self.id

    def check_weekday(self, _):
        """Check, if location takes place on this weekday.

        Returns:
            (boolean): True, if location takes place. False, else.
        """
        return True

    # @profile
    def do_quarantine_member(self, agent_id):
        """Remove agent from location due to
        quarantine/isolation/hospitalization.

        Ensure agent is no longer in lists active and set its
        new status in agents_info. The information will be used in the get methods.
        Possible shifts can be:
            0 : active --> 97 : quarantined --> remove member from active list.

        Args:
            agent_id (int): id of agent to be quarantined.

        Returns:
            (boolean): True, if agent was removed. False, else.
        """
        # return, if agent not in network or is already quarantined
        if (agent_id not in self.members_all) or (
            self.agents_info[agent_id][cs.STATUS] == 97
        ):
            return False
        # quarantine agent
        self.agents_info[agent_id][cs.STATUS] = 97
        self.members_active.remove(agent_id)
        return True

    # @profile
    def do_un_quarantine_member(self, agent_id):
        """Add agent back to location due to
        quarantine/isolation/hospitalization.

        Ensure agent is added back to list active and status set in agents_info.
        Possible shifts can be:
            97 : quarantined --> 0 : active --> add member to active list.

        Returns:
            (boolean): True, if agent is sent back. False, else.
        """
        if self.agents_info[agent_id][cs.STATUS] == 0:
            return False
        self.agents_info[agent_id][cs.STATUS] = 0
        self.members_active.append(agent_id)
        return True

    def get_primary_contacts(self, agent_id, _):
        """Get primary contacts from agent.

        Only consider active members and active contacts.

        Args:
            agent_id (int): id of agent to get contacts from.
            _ (int): from 0 to 7, means Mo - Su.

        Returns:
            primary contacts (list[int]): List with agent_ids of primary contacts.
        """
        if agent_id not in self.agents_info:
            return []
        if self.agents_info[agent_id][cs.STATUS] == 0:
            graph = self.members_graph
            primary_contacts = graph.extract_agents_neighbours(agent_id)
            # Graphs are static now, filter active contacts
            primary_active = [
                a_id
                for a_id in primary_contacts
                if self.agents_info[a_id][cs.STATUS] == 0
            ]
            return primary_active
        else:
            return []

    def get_cluster_contacts(self, agent_id, _):
        """Get cluster contacts from agent.

        Cluster contacts are all members of the location.
        Only consider active members and active contacts.

        Args:
            agent_id (int): id of agent to get contacts from.
            _ (int): from 0 to 7, means Mo - Su.

        Returns:
            cluster contacts (list[int]): List with agent_ids of cluster contacts.
        """
        if agent_id in self.members_active:
            members_active_copy = self.members_active.copy()
            members_active_copy.remove(agent_id)
            return members_active_copy
        else:
            return []


class PublicLocation(Location):
    """A PublicLocation is associated to a public building or room.

    It can be localized, such as a class, a meeting room, a mensa or an office,
    or un-localized, such as a building corridor or a yard. Public locations
    (despite ActivityPlaces) belong to a superior public institution, such as a
    school or a workplace. The institution is not an object, but symbolized by
    a unique id (here called parent_id). Most interventions affect public locations
    or institutions. Therefore, the ClassPublicLocation has to manage a lot of
    overlaps between different interventions, such as LIMIT_EVENT_SIZE, SPLIT_GROUPS,
    HOME_DOING, being SHUT or QUARANTINE.

    The following public locations (institution) are possible:
        office (workplace)
        meeting (workplace)
        class (school, kita)
        lecture (uni)
        free = e.g. extra activity, break room (school, kita, uni)
        mensa (workplace, school, kita, uni)
        unspecific = e.g. corridor, yard (workplace, school, kita, uni)
        activity = e.g. sports, singing, cinema (None)

    Attributes:
        self.size_total (int): number of total members
        self.parent_id (int): id of superior institution
        self.first_digit (int): digit standing for type of superior institution
        self.mean_contacts (int): number of mean contacts within a graph
        self.weekdays (list[int]): weekdays, where meetings at the location take place.
        self.num_groups (int): number of subgroups due to intervention SPLIT_GROUPS
        self.shut (boolean): weather location is shut or not due to shut of superior institution.
        self.event_limit (int): max. number of people due to intervention LIMIT_EVENT_SIZE, 10000 as default value
        self.size_active (list[int]): number of active members within subgroup (intervention SPLIT_GROUPS)
        self.members_all (list[int]): list with agent_ids of all members from the location
        self.members_active (list[list[int]]): list with members per subgroup actively participating
        self.members_passive (list[list[int]]): list with members per subgroup not participating due to LIMIT_EVENT_SIZE
        self.members_do_home(list[int]): list with members not participating due to intervention HOME_DOING
        self.agents_info (dict[int, dict[str, int]]): Dict to register status and subgroup for each member.
        self.members_graph (ClassGraph): Symbolizing contact patterns (members = nodes, contacts = edges).
    """

    def __init__(self, members, local_info, seed):
        """Initialize properties of the location and fields to store
        information in.

        Important are the fields to handle information of the agent's status within
        the location. This is to manage different interventions in parallel. And also
        the contact pattern among agents.

        Args:
            members (list[int]): List with ids of agents that will participate in the network.
            local_info (NumPyArray[int/str]): contains information on the location
                0 --> PARENT_ID = ID of superior institution
                1 --> NET_ID = ID of location
                2 --> TYPE of location
                3 --> SIZE of location
                4 --> MEAN_CONTACTS
                5 --> YEAR/AGE_GR
                6 --> list of WEEKDAY's
                7 --> LANDSCAPE (30 urban, 60 rural)
                8 --> TYPE of superior institution
                9 --> SIZE of superior institution
                10 --> REGION
        """
        super(PublicLocation, self).__init__(
            uid=local_info[1], my_type=Sync.code_to_type(local_info[2])
        )
        # Todo: see why size_total in the table differs to len(members)
        # assert(local_info[3] == len(members))
        self.seed = seed
        self.size_total = len(members)
        self.parent_id = local_info[0]
        self.first_digit = Sync.get_first_digit(local_info[1])
        self.mean_contacts = local_info[4]
        self.weekdays = Sync.extract_weekdays(local_info[6])
        self.num_groups = 1
        self.shut = 0
        self.event_limit = 10000  # default value
        self.size_active = [len(members)]
        self.members_all = members
        self.members_active = [members.copy()]
        self.members_passive = [[]]
        self.members_do_home = []
        self.agents_info = self._initialize_status_and_sub_gr(members)
        self.members_graph = [self._create_graph_from_members(members)]
        assert len(self.members_all) == self.size_total

    def _initialize_status_and_sub_gr(self, members):
        """Create dict with info about status and sub_gr for each member.

        By default, the sub_gr is 0, it can go 0-3 due to intervention SPLIT_GROUPS.
        By default, the status is 0, which means active member.
        Possible statuses can be:
            0 : active --> member does actively participate
            99 : passive --> member is absent due to intervention LIMIT_EVENT_SIZE
            98 : do home --> member is absent due to intervention HOME_DOING
            97 : quarantined --> member is absent due to interventions with TRACING
            100 : do home + quarantined --> member is absent due to both

        Args:
            members (list[int]): List with ids of agents.

        Returns:
            agents_info (dict[int, dict[str, int]]): info about each member.
        """
        agents_info = {}
        for member in members:
            agents_info[member] = {cs.STATUS: 0, cs.SUB_GR: 0}
        return agents_info

    def _create_graph_from_members(self, members):
        """Create graph from all members in this location.

        If the network is small, which means every member has contact to
        every other member, create GraphSmall. Otherwise, create GraphMedium.
        The graph will stay constant with all members during the simulation.
        If a member gets passive, quarantined or so, this will be checked and
        filtered by the get methods.

        Returns:
            members_graph (ClassGraph): Graph with members as nodes and edges as contacts.
        """
        if len(members) == self.mean_contacts + 1:
            members_graph = graphs.GraphSmall(members, self.mean_contacts, self.id)

        else:
            members_graph = graphs.GraphMedium(
                members, self.mean_contacts, self.id, self.seed
            )
        return members_graph

    def _limit_reached(self, sub_gr):
        """Check, if the limit for active people is reached for the sub_gr.

        Important for intervention LIMIT_EVENT_SIZE.

        Args:
            sub_gr (int): 0 - 3 (results from intervention SPLIT_GROUPS)
        """
        if self.size_active[sub_gr] >= self.event_limit:
            return True
        else:
            return False

    def _help_extract_my_members(self, agent_ids):
        """Extract those agent_ids from list, that are members in this
        location.

        Args:
            agent_ids (list[int]): List with ids of agents.

        Returns:
            members (list[int]): List with ids of members.
        """
        my_members = agent_ids[np.isin(agent_ids, self.members_all)]
        return list(my_members)

    # @profile
    def _remove_member_from_active(self, member, sub_gr, new_status):
        """Remove member from active.

        Set members new status in agents_info, remove member from list members_active.
        Also manage the size of members_active and where required, move a member from
        passive to active and in turn, set its status in agents_info.

        Args:
            member (int): id of a member.
            sub_gr (int): 0 - 3 (results from intervention SPLIT_GROUPS)
            new_status (int): see list of statuses
        """
        self.agents_info[member][cs.STATUS] = new_status
        self.members_active[sub_gr].remove(member)
        if self.members_passive[sub_gr]:
            move_up = self.members_passive[sub_gr].pop()
            self.agents_info[move_up][cs.STATUS] = 0
            self.members_active[sub_gr].append(move_up)
        else:
            self.size_active[sub_gr] -= 1

    def _remove_member_from_passive(self, member, sub_gr, new_status):
        """Remove member from passive.

        Set members new status in agents_info, remove member from list members_passive.

        Args:
            member (int): id of a member.
            sub_gr (int): 0 - 3 (results from intervention SPLIT_GROUPS)
            new_status (int): see list of statuses
        """
        self.agents_info[member][cs.STATUS] = new_status
        self.members_passive[sub_gr].remove(member)

    # @profile
    def _add_member_to_active(self, member, sub_gr):
        """Add member to active.

        Set members status to 0 in agents_info, add member to list members_active.
        Also manage the size of members_active.

        Args:
            member (int): id of a member.
            sub_gr (int): 0 - 3 (results from intervention SPLIT_GROUPS)
        """
        self.agents_info[member][cs.STATUS] = 0
        self.members_active[sub_gr].append(member)
        self.size_active[sub_gr] += 1

    def _add_member_to_passive(self, member, sub_gr):
        """Add member to passive.

        Set members status to 99 in agents_info, add member to list members_passive.

        Args:
            member (int): id of a member.
            sub_gr (int): 0 - 3 (results from intervention SPLIT_GROUPS)
        """
        self.agents_info[member][cs.STATUS] = 99
        self.members_passive[sub_gr].append(member)

    def do_limit_events(self, size):
        """Organize intervention LIMIT_EVENT_SIZE within the location.

        Iterate subgroups of active members and check, if the limit within
        the subgroup is reached. If so, shift the overspill of people from
        active to passive and change their status to 99.

        Args:
            size (int): max. size for groups of active members.

        Returns:
            changed (boolean): True, if change has taken place. False else.
        """
        changed = False
        if not self.localized:
            return changed
        self.event_limit = size
        for sub_gr in range(self.num_groups):
            if self.size_active[sub_gr] > self.event_limit:
                self.size_active[sub_gr] = self.event_limit
                my_active = self.members_active[sub_gr][: self.event_limit]
                my_passive = self.members_active[sub_gr][self.event_limit :]
                self.members_active[sub_gr] = my_active
                self.members_passive[sub_gr] = my_passive
                for member in my_passive:
                    self.agents_info[member][cs.STATUS] = 99
                changed = True
        return changed

    def do_un_limit_event_size(self):
        """Organize undo intervention LIMIT_EVENT_SIZE within the location.

        Iterate subgroups and check, if there are any passive members.
        If so, move them back to active  and empty list of passive members.

        Returns:
            changed (boolean): True, if change has taken place. False else.
        """
        changed = False
        if not self.localized:
            return changed
        # iterate relevant events
        for sub_gr in range(self.num_groups):
            if self.members_passive[sub_gr]:
                assert len(self.members_active[sub_gr]) == self.event_limit
                for member in self.members_passive[sub_gr]:
                    self._add_member_to_active(member, sub_gr)
                self.members_passive[sub_gr] = []
                assert self.size_active[sub_gr] == len(self.members_active[sub_gr])
                changed = True
        self.event_limit = 10000
        return changed

    def do_shuffle_active_and_passive_members(self):
        """Shuffle active and passive members due to intervention
        LIMIT_EVENT_SIZE.

        Iterate subgroups and check, if there are any passive members.
        If so, implement an algorithm to randomly exchange active
        for passive members. Ensure members are removed from/to list
        of active/passive members and their status is changed accordingly.

        Returns:
            changed (boolean): True, if change has taken place. False else.
        """
        assert len(self.members_all) == self.size_total
        assert self.localized
        changed = False
        for sub_gr in range(self.num_groups):
            if self.members_passive[sub_gr]:
                assert len(self.members_active[sub_gr]) == self.event_limit
                changed = True
                num_passive = len(self.members_passive[sub_gr])
                num_active = self.size_active[sub_gr]
                if num_passive:
                    pos = self.seed.integers(0, num_passive)
                    for idx_act in range(0, num_active, 2):
                        idx_pas = (idx_act + pos) % num_passive
                        passive = self.members_passive[sub_gr][idx_pas]
                        active = self.members_active[sub_gr][idx_act]
                        self.agents_info[active][cs.STATUS] = 99
                        self.members_passive[sub_gr][idx_pas] = active
                        self.agents_info[passive][cs.STATUS] = 0
                        self.members_active[sub_gr][idx_act] = passive
            assert len(self.members_active[sub_gr]) == self.size_active[sub_gr]
        return changed

    # @profile
    def do_shuffle_contacts(self):
        """Shuffle contact pattern within the location.

        Iterate subgroups and shuffle associated graph.
        """
        for sub_gr in range(self.num_groups):
            self.members_graph[sub_gr].shuffle_member()

    def do_split_location(self, degree):
        """Split location and its members into subgroups.

        Groups can be split once (degree 2) or twice (degree 4). Even though
        the input degree is limited to the values 2 or 4, the group can still
        be split into 3 groups. Also groups with size lower than 5 will not be
        split. This is to ensure that subgroups have a minimum size of 3-4 people,
        but will be split at a size bigger 5. The method guaranties that every
        member is registered in a subgroup, even though he/she might be in
        quarantine. Also, the intervention LIMIT_EVENT_SIZE will be re-considered
        after the group was split.

        Args:
            degree (int): number of subgroups, must be 2 or 4.

        Returns:
            (boolean): True, if group was split. False else.
        """
        if self.num_groups > 1:
            raise AssertionError("Group is already split, can only be split one time")
        if (degree != 2) and (degree != 4):
            raise AssertionError("Groups can only split by degree 2 or 4")
        if (not self.localized) or (self.size_total < 6):
            return False
        if self.size_total < 11:  # if groups smaller 11, split by 2
            self.num_groups = 2
        elif self.size_total < 16:  # if 11 < groups < 16 split by 3 or by degree
            self.num_groups = min(degree, 3)
        else:
            self.num_groups = degree
        # reset group members
        self._help_reset_members_stuff()
        group_sizes = {
            cs.MAX: math.ceil(self.size_total / self.num_groups),
            cs.MIN: math.floor(self.size_total / self.num_groups),
            "num_max": self.size_total % self.num_groups,
        }
        start = 0
        group_size = self._help_update_group_size(group_sizes)
        end = group_size
        for sub_gr in range(self.num_groups):
            group_members = self.members_all[start:end]
            for member in group_members:
                self.agents_info[member][cs.SUB_GR] = sub_gr
                status = self.agents_info[member][cs.STATUS]
                if (status != 0) & (status != 99):
                    continue
                if not self._limit_reached(sub_gr):
                    self._add_member_to_active(member, sub_gr)
                else:
                    self._add_member_to_passive(member, sub_gr)
            graph = self._create_graph_from_members(group_members)
            self.members_graph.append(graph)
            start += group_size
            group_size = self._help_update_group_size(group_sizes)
            end += group_size
            assert self.size_active[sub_gr] == len(self.members_active[sub_gr])
        return True

    def _help_update_group_size(self, group_sizes):
        if group_sizes["num_max"] > 0:
            group_size = group_sizes[cs.MAX]
        else:
            group_size = group_sizes[cs.MIN]
        group_sizes["num_max"] -= 1
        return group_size

    def _help_reset_members_stuff(self):
        self.members_active = []
        self.members_passive = []
        self.size_active = []
        self.members_graph = []
        for i in range(self.num_groups):
            self.members_active.append([])
            self.members_passive.append([])
            self.size_active.append(0)

    def do_union_location(self):
        """Union subgroups within location.

        If the location was split into subgroups, now all its members are sent
        back into one group.The method guaranties that every member is registered
        in subgroup 0, even though he/she might be in quarantine. After union the
        intervention LIMIT_EVENT_SIZE will be re-considered again.

        Returns:
            (boolean): True, if group was split. False else.
        """
        if (not self.localized) or (self.num_groups == 1):
            return False
        active = []
        passive = []
        for sub_gr in range(self.num_groups):
            active += self.members_active[sub_gr]
            passive += self.members_passive[sub_gr]
        self.size_active = [sum(self.size_active)]

        while self.size_active[0] > self.event_limit:
            mover = active.pop()
            passive.append(mover)
            self.agents_info[mover][cs.STATUS] = 99
            self.size_active[0] -= 1

        for agent in self.agents_info:
            self.agents_info[agent][cs.SUB_GR] = 0
        self.members_graph = [self._create_graph_from_members(self.members_all)]
        assert self.members_graph[0].node_num == self.size_total
        self.members_active = [active]
        self.members_passive = [passive]
        self.num_groups = 1
        return True

    def do_shut_location(self):
        """Mark the location as shut.

        The information will be used in the get methods. An active agent
        with active primary contact will still give an empty list of
        contacts due to shut location.
        """
        self.shut = 1

    def do_open_location(self):
        """Mark the location as not shut.

        The information will be used in the get methods.
        """
        self.shut = 0

    def do_send_home(self, members):
        """Remove members from location due to intervention HOME_DOING.

        Extract those members that are members of this location. Ensure they are
        no longer in lists active or passive and set their new status in agents_info.
        The information will be used in the get methods.
        Possible shifts can be:
            0 : active --> 98: do home --> remove member from active list.
            99 : passive --> 98: do home --> remove member from passive list.
            97 : quarantined --> 100 : do home + quarantined.

        Args:
            members (list[int]): List with ids of agents to send home.

        Returns:
            (boolean): True, if some members are sent home. False, else.
        """
        self.members_do_home = self._help_extract_my_members(members)
        if not self.members_do_home:
            return False
        for member in self.members_do_home:
            sub_gr = self.agents_info[member][cs.SUB_GR]
            status = self.agents_info[member][cs.STATUS]
            if status == 0:
                self._remove_member_from_active(member, sub_gr, 98)
            elif status == 99:
                self._remove_member_from_passive(member, sub_gr, 98)
            elif status == 97:
                self.agents_info[member][cs.STATUS] = 100
        return True

    # @profile
    def do_un_send_home(self):
        """Add members back to location due to intervention HOME_DOING.

        Iterate list of members that are in HOME_DOING. Ensure they are
        added to list active or passive and set their status in agents_info.
        When also quarantined, only set their status in agents_info.
        The information will be used in the get methods.
        Possible shifts can be:
            98: do home --> 0 : active --> add member to active list.
            98: do home --> 99 : passive --> add member to passive list.
            100 : do home + quarantined --> 97 : quarantined

        Returns:
            (boolean): True, if some members are sent back. False, else.
        """
        if not self.members_do_home:
            return False
        for member in self.members_do_home:
            sub_gr = self.agents_info[member][cs.SUB_GR]
            status = self.agents_info[member][cs.STATUS]
            if status == 98:
                if self._limit_reached(sub_gr):
                    self._add_member_to_passive(member, sub_gr)
                else:
                    self._add_member_to_active(member, sub_gr)
            elif status == 100:
                self.agents_info[member][cs.STATUS] = 97
        self.members_do_home = []
        return True

    # @profile
    def do_quarantine_member(self, agent_id):
        """Remove agent from location due to
        quarantine/isolation/hospitalization.

        Ensure agent is no longer in lists active or passive and set its
        new status in agents_info. The information will be used in the get methods.
        Possible shifts can be:
            0 : active --> 97 : quarantined --> remove member from active list.
            99 : passive --> 97 : quarantined --> remove member from passive list.
            98 : do home --> 100 : do home + quarantined.

        Args:
            agent_id (int): id of agent to be quarantined.

        Returns:
            (boolean): True, if agent was removed. False, else.
        """
        if agent_id not in self.members_all:
            return False
        sub_gr = self.agents_info[agent_id][cs.SUB_GR]
        status = self.agents_info[agent_id][cs.STATUS]
        if (status == 97) or (status == 100):
            return False
        else:
            if status == 0:
                self._remove_member_from_active(agent_id, sub_gr, 97)
            elif status == 99:
                self._remove_member_from_passive(agent_id, sub_gr, 97)
            elif status == 98:
                self.agents_info[agent_id][cs.STATUS] = 100
            return True

    def do_un_quarantine_member(self, agent):
        """Add agent back to location due to
        quarantine/isolation/hospitalization.

        Ensure agent is added back to list active or passive and set its
        status in agents_info. When in also HOME_DOING, only set its status
        in agents_info. The information will be used in the get methods.
        Possible shifts can be:
            97 : quarantined --> 0 : active --> add member to active list.
            97 : quarantined --> 99 : passive --> add member to passive list.
            100 : do home + quarantined --> 98 : do home

        Returns:
            (boolean): True, if agent is sent back. False, else.
        """
        if agent not in self.members_all:
            return False
        sub_gr = self.agents_info[agent][cs.SUB_GR]
        status = self.agents_info[agent][cs.STATUS]
        if status == 97:
            if self._limit_reached(sub_gr):
                self._add_member_to_passive(agent, sub_gr)
            else:
                self._add_member_to_active(agent, sub_gr)
        elif status == 100:
            self.agents_info[agent][cs.STATUS] = 98
        return True

    def get_original_members(self):
        """Get all members of the location, independent of its status.

        Returns:
            self.members_all (list[int]): list with agent_ids from location.
        """
        return self.members_all

    def get_active_members(self):
        """Get all active members of the location.

        Active members are those currently present. In contrast, passive
        members stay home due to intervention LIMIT_EVENT_SIZE. Others
        stay home due to QUARANTINE or HOME_DOING. Everyone stays home
        when location is shut. The status can change in any time step.

        Returns:
            active_members (list[int]): list with active agent_ids from location.
        """
        if not self.shut:
            active_members = [
                agent_id
                for agent_id in self.agents_info
                if self.agents_info[agent_id][cs.STATUS] == 0
            ]
            return active_members
        else:
            return []

    def get_first_digit(self):
        """Get digit standing for network type.

            2: "workplace",
            3: "kita",
            4: "school",
            5: "uni",
            6: "activity",

        Returns:
            self.first_digit (int): digit of network type.
        """
        return self.first_digit

    def get_sub_gr(self, agent_id):
        """UNUSED METHOD Get subgroup from agent.

        When the intervention SPLIT_NETWORKS is on, there can be subgroups 0-3.

        Returns:
            sub_gr (int): subgroup (0-3) from agent.
        """
        return self.agents_info[agent_id][cs.SUB_GR]

    def get_status(self, agent_id):
        """UNUSED METHOD Get status from agent.

        0 : active --> member does actively participate
        99 : passive --> member is absent due to intervention LIMIT_EVENT_SIZE
        98 : do home --> member is absent due to intervention HOME_DOING
        97 : quarantined --> member is absent due to interventions with TRACING
        100 : do home + quarantined --> member is absent due to both

        Returns:
            status (int): status from agent.
        """
        return self.agents_info[agent_id][cs.STATUS]

    def get_net_id(self, parent=False):
        """Get unique id of the location.

        Args:
            parent (boolean): if True, get id of superior institution.

        Returns:
            id (int): unique id of location or superior institution.
        """
        if parent:
            return self.parent_id
        else:
            return self.id

    def check_weekday(self, weekday):
        """Check, if location takes place on this weekday.

        Args:
            weekday (int): 0-7 (Mo-Su)

        Returns:
            (boolean): True, if location takes place. False, else.
        """
        assert weekday in range(7)
        return weekday in self.weekdays

    def is_shut(self):
        """True, when shut.

        False, else
        """
        return self.shut

    def is_small(self):
        """True, when small. False, else.

        A location is small, when all members have contact to each
        other.
        """
        if self.size_total == self.mean_contacts + 1:
            return True
        else:
            return False

    def get_primary_contacts(self, agent_id, weekday):
        """Get primary contacts from agent.

        Consider all criteria and only return primary contacts, that have
        actively taken place on this weekday. Criteria are:
        Location is not shut, meeting does take place on specified weekday,
        agent is active and primary contacts are active as well.

        Args:
            agent_id (int): id of agent to get contacts from.
            weekday (int): from 0 to 7, means Mo - Su.

        Returns:
            primary contacts (list[int]): List with agent_ids of primary contacts.
        """
        # case 1: general criteria fail
        if (
            self.shut
            or (agent_id not in self.agents_info)
            or (not self.check_weekday(weekday))
        ):
            return []
        # case 2: agent active
        sub_gr = self.agents_info[agent_id][cs.SUB_GR]
        status = self.agents_info[agent_id][cs.STATUS]
        if status == 0:
            graph = self.members_graph[sub_gr]
            primary_contacts = graph.extract_agents_neighbours(agent_id)
            # Graphs are static, filter active contacts
            primary_active = [
                a_id
                for a_id in primary_contacts
                if self.agents_info[a_id][cs.STATUS] == 0
            ]
            return primary_active
        # case 3: agent not active
        else:
            return []

    def get_cluster_contacts(self, agent_id, weekday):
        """Get cluster contacts from agent.

        Cluster contacts are all members of a group, where the agent was a member of.
        Consider all criteria and only return cluster contacts, that have
        actively taken place on this weekday. Criteria are:
        Location is not shut, meeting does take place on specified weekday,
        agent is active and cluster contacts are active as well.

        Args:
            agent_id (int): id of agent to get contacts from.
            weekday (int): from 0 to 7, means Mo - Su.

        Returns:
            cluster contacts (list[int]): List with agent_ids of primary contacts.
        """
        # case 1: general criteria fail
        if (
            self.shut
            or (not self.localized)
            or (agent_id not in self.agents_info)
            or (not self.check_weekday(weekday))
        ):
            return []
        # case 2: agent active
        sub_gr = self.agents_info[agent_id][cs.SUB_GR]
        status = self.agents_info[agent_id][cs.STATUS]
        if status == 0:
            members_active_copy = self.members_active[sub_gr].copy()
            members_active_copy.remove(agent_id)
            return members_active_copy
        # case 3: agent not active
        else:
            return []


class WorkPlace(PublicLocation):
    """Symbolize a location within an institution workplace.

    WorkPlace can be a location of type office, meeting, mensa or
    unspecific.
    """

    def __init__(self, members, local_info, seed):
        super(WorkPlace, self).__init__(members, local_info, seed)


class SchoolPlace(PublicLocation):
    """Symbolize a location within an institution school.

    SchoolPlace can be a location of type class, free, mensa or unspecific.

    Attributes:
        self.year (int): symbolize class level.
    """

    def __init__(self, members, local_info, seed):
        super(SchoolPlace, self).__init__(members, local_info, seed)
        self.year = local_info[5]


class UniPlace(PublicLocation):
    """Symbolize a location within an institution uni.

    UniPlace can be a location of type lecture, free, mensa or
    unspecific.
    """

    def __init__(self, members, local_info, seed):
        super(UniPlace, self).__init__(members, local_info, seed)
        self.year = local_info[5]


class KitaPlace(PublicLocation):
    """Symbolize a location within an institution kita.

    KitaPlace can be a location of type class, free, mensa or
    unspecific.
    """

    def __init__(self, members, local_info, seed):
        super(KitaPlace, self).__init__(members, local_info, seed)


class Activity(PublicLocation):
    """Symbolize an activity such as a sports class, choir or restaurant.

    Activities are what people do in their free time. They are NOT associated
    to a public institution and always localized. There is no classification
    of sports, singing etc. yet, but can easily be added.

    Attributes:
        self.parent_id (int): None (overwrites SuperClass).
        self.age_gr (int): age group of participating members (999 for mixed).
    """

    def __init__(self, members, local_info, seed):
        super(Activity, self).__init__(members, local_info, seed)
        self.parent_id = None
        self.age_gr = local_info[5]


class GeographicArea(Location):
    """A geographic area symbolizes face-to-face contacts within a defined
    territory.

    The territory can be a region, city or federal state. Geographic ares are always
    un-localized, which means, the location is unspecified and differs from
    face-to-face to face-to-face contact pair. Geographic areas are not affected by
    interventions such as LIMIT_EVENT_SIZE, SPLIT_GROUPS, HOME_DOING or being SHUT.
    They only have to manage QUARANTINE, ISOLATION and HOSPITALIZATION.

    Attributes:
        self.size_total (int): number of total members
        self.size_active (int): number of active members
        self.agents_info (dict[str, NumPyArray[int]]): Dict to register id and status for each member.
        self.id_to_idx (dict[int, int]): Hold translation between agent_id and index.
        self.members_graph (ClassGraph): Symbolizing contact patterns (members = nodes, contacts = edges).
    """

    def __init__(self, members, uid, percentage_active, seed):
        """Initialize properties of the location and contact pattern.

        Important are the fields holding contact pattern, agent's statuses
        and the translation between id and index.

        Args:
            members (NumPyArray[int]): Array with ids of agents that are part of the location.
            uid (int): unique id of the location.
            percentage_active (float): active members related to all members
        """
        super(GeographicArea, self).__init__(uid=uid, my_type=cs.GEOGRAPHIC)
        self.seed = seed
        self.size_total = len(members)
        self.size_active = int(len(members) * percentage_active)
        self.agents_info = self._initialize_status(members)
        self.id_to_idx = dict(zip(self.agents_info[cs.ID], np.arange(self.size_total)))
        self.members_graph = self._create_graph_from_active()
        assert self.members_graph.node_num == self.size_active

    def _initialize_status(self, members):
        """Initialize status for agents and save to dict.

        Sample the active members by size active, create a graph with active members
        as nodes and register all other members as passive.
        Possible statuses can be:
            0 : active --> member does actively participate = node in the graph
            99 : passive --> member does not actively participate.
            97 : quarantined/passive --> member is quarantined and was not actively participating
            96 : quarantined/active --> member is quarantined and was actively participating

        Args:
            members (NumPyArray[int]): Array with ids of agents that are part of the location.

        Returns:
            agents_info (dict[str, NumPyArray[int]]): Hold array with agent_ids and array with agent's status.
        """
        agents_info = {cs.ID: members, cs.STATUS: np.zeros(len(members))}
        ids_active = self.seed.choice(members, self.size_active, replace=False)
        ids_passive = members[np.isin(members, ids_active, invert=True)]
        condition_passive = np.isin(agents_info[cs.ID], ids_passive)
        agents_info[cs.STATUS][condition_passive] = 99
        return agents_info

    def _create_graph_from_active(self):
        """Initialize graph with active members as nodes.

        The number of mean contacts is set to 3.

        Returns:
            graph (ClassGraph): Symbolizing contact patterns (members = nodes, contacts = edges).
        """
        ids_active = self.agents_info[cs.ID][self.agents_info[cs.STATUS] == 0]
        mean_contacts = Sync.LOCATION_TYPE_INFORMATION[self.type][cs.MEAN_CONTACTS]
        graph = graphs.GraphLarge(
            ids_active, mean_contacts=mean_contacts, net_id=self.id, seed=self.seed
        )
        return graph

    # @profile
    def do_quarantine_member(self, agent_id):
        """Remove agent from location due to
        quarantine/isolation/hospitalization.

        Ensure agent is no longer in the graph and set its new status in agents_info.
        The information will be used in the get methods.
        Possible shifts can be:
            0 : active --> 96 : quarantined/active --> remove member from graph.
            99 : passive --> 97 : quarantined/passive --> only set new status.

        Args:
            agent_id (int): id of agent to be quarantined.

        Returns:
            (boolean): True, if agent was removed. False, else.
        """
        if agent_id not in self.id_to_idx:
            return False
        agent_status = self.agents_info[cs.STATUS][self.id_to_idx[agent_id]]
        if (agent_status == 96) or (agent_status == 97):
            return False
        elif agent_status == 0:
            self.agents_info[cs.STATUS][self.id_to_idx[agent_id]] = 96
            self.members_graph.remove_member_from_graph(agent_id)
            return True
        elif agent_status == 99:
            self.agents_info[cs.STATUS][self.id_to_idx[agent_id]] = 97
            return True
        else:
            print(f"{agent_status} not exist.")
            return False

    # @profile
    def do_un_quarantine_member(self, agent):
        """Add agent back to location due to interventions with TRACING.

        Ensure agent is added back to graph (when was active) and/or set its
        status in agents_info. The information will be used in the get methods.
        Possible shifts can be:
            96 : quarantined/active --> 0 : active --> add member to graph and set status.
            97 : quarantined/passive --> 99 : passive --> only set status.

        Returns:
            (boolean): True, if agent is sent back. False, else.
        """
        if agent not in self.id_to_idx:
            return False
        agent_status = self.agents_info[cs.STATUS][self.id_to_idx[agent]]
        if agent_status == 97:
            self.agents_info[cs.STATUS][self.id_to_idx[agent]] = 99
            return True
        elif agent_status == 96:
            self.agents_info[cs.STATUS][self.id_to_idx[agent]] = 0
            self.members_graph.add_member_to_graph(agent)
            return True
        else:
            return False

    def do_shuffle_contacts(self):
        """Exchange active and passive members.

        Take a number of agents according to a defined percentage. Sample the number
        of agents from both, active and passive members. Remove active members from
        graph and set status to 99. Add passive members to graph and set status to 0.
        Due to the exchange, also the contact pattern within the graph will be shuffled.
        Possible shifts can be:
            0 : active --> 99 : passive --> remove member from graph.
            99 : passive --> 0 : active --> add member to graph.
        """
        active_mbr = self.agents_info[cs.ID][self.agents_info[cs.STATUS] == 0]
        passive_mbr = self.agents_info[cs.ID][self.agents_info[cs.STATUS] == 99]
        assert self.members_graph.node_num == len(active_mbr)
        num_active = len(active_mbr)
        per_ex = 0.10
        # limit to 1000 due to time consumption
        num_ex = min(int(per_ex * num_active), 1000)
        if num_ex > 0:
            exchange_active = self._sample_exchanger(active_mbr, num_ex)
            for agent in exchange_active:
                self.members_graph.remove_member_from_graph(agent)
                self.agents_info[cs.STATUS][self.id_to_idx[agent]] = 99
            assert self.members_graph.node_num == len(active_mbr) - num_ex
            exchange_passive = self._sample_exchanger(passive_mbr, num_ex)
            for agent in exchange_passive:
                self.members_graph.add_member_to_graph(agent)
                self.agents_info[cs.STATUS][self.id_to_idx[agent]] = 0

    def _sample_exchanger(self, members, num_ex):
        """Sample the given number of members.

        The sampling algorithm is faster than numpy.random.choice, which
        is an advantage for higher number of members.

        Args:
            members (NumPyArray[int]): array with agent ids.
            num_ex (int): number of exchangers to sample from members.

        Returns:
            exchanger (NumPyArray[int]): array with agent ids to exchange.
        """
        num = len(members)
        step = int(num / num_ex)
        start = self.seed.integers(0, step - 1)
        end = start + step * (num_ex - 1) + 1
        exchanger = members[start:end:step]
        assert num_ex == len(exchanger)
        return exchanger

    def get_primary_contacts(self, agent_id, _):
        """Get primary contacts from agent.

        Only active members have primary contacts.

        Args:
            agent_id (int): id of agent to get contacts from.
            _ : weekday is not considered as geographic areas are every day.

        Returns:
            primary contacts (list[int]): List with agent_ids of primary contacts.
        """
        status = self.agents_info[cs.STATUS][self.id_to_idx[agent_id]]
        if status == 0:
            return self.members_graph.extract_agents_neighbours(agent_id)
        else:
            return []

    def get_original_members(self):
        """Get all members of the location, independent of its status.

        Returns:
            (list[int]): list with agent_ids from location.
        """
        return self.agents_info[cs.ID]

    def get_active_members(self):
        """Get active members of the location.

        Active members are those having contacts within the geographical area.
        In contrast, passive do not. The status can change in any time step.

        Returns:
            active_members (NumPyArray[int]): array with active agent_ids from location.
        """
        return self.agents_info[cs.ID][self.agents_info[cs.STATUS] == 0]

    def get_first_digit(self):
        """Get digit standing for network type.

        Returns:
            self.first_digit (int): digit of network type.
        """
        return 7

    def get_status(self, agent_id):
        """UNUSED METHOD Get status from agent.

        Possible statuses can be:
            0 : active --> member does actively participate = node in the graph
            99 : passive --> member does not actively participate.
            97 : quarantined/passive --> member is quarantined and was not actively participating
            96 : quarantined/active --> member is quarantined and was actively participating

        Returns:
            status (int): status from agent.
        """
        index = self.id_to_idx[agent_id]
        return self.agents_info[cs.ID][index]

    def get_net_id(self):
        """Get unique id of the location.

        Returns:
            id (int): unique id of location.
        """
        return self.id

    def check_weekday(self, _):
        """Check, if location takes place on this weekday.

        Returns:
            (boolean): True, if location takes place. False, else.
        """
        return True


class Events:
    """NOT IMPLEMENTED YET."""

    def __init__(self):
        pass
