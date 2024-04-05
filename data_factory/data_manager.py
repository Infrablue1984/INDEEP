#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Update keydata if new my_regions were choosen or new federal statistics were
uploaded.

At the moment create network objects and organize the saving of net_ids
to people class object. Call Transformer, Controller and Network
classes.
"""

__author__ = "Inga Franzen"
__created__ = "2020"
__date_modified__ = "2022/09/30"
__version__ = "1.0"

import os
import threading
import time

import numpy as np
from line_profiler_pycharm import profile

import data_factory.generator.people as p
from data_factory.generator.activities import Activities
from data_factory.generator.households import Households
from data_factory.generator.networks_d import Institution
from data_factory.generator.schools import Schools
from data_factory.generator.works import Workplaces
from synchronizer import constants as cs
from synchronizer.config import Config
from synchronizer.synchronizer import FileLoader, FileSaver
from synchronizer.synchronizer import PathManager as PM
from synchronizer.synchronizer import Synchronizer as Sync

"""Call all methods in script to download and update data.

Specifically save agents_data of people class including network ids in ndarray/csv"""


class DataManager:
    @staticmethod
    def organize_generation_of_key_data_for_selected_regions(
        regions, scale, sim_progress=False
    ):
        """Organize the generation of key data for the selected regions.

        People, Household and School objects etc. are created by the
        implementation of algorithms on different statistical data. In
        order to speed up the simulation, these calculations are done
        only one time for each region and only, if the region was
        selected by the user. The results are saved in a modified data
        file. This method checks for new regions to be calculated and
        if, calls the DataManager class to perform the data generation.
        """
        (creation_needed, regions_new,) = DataManager.check_for_new_regions_to_create(
            regions,
            scale,
        )
        if creation_needed:
            DataManager.create_people_and_network_files(
                regions_new,
                scale,
                sim_progress,
            )
            time.sleep(0.5)

    @staticmethod
    # @profile
    def create_people_and_network_files(regions, scale, sim_progress):
        my_regions = regions
        slice_max = 30
        total_num_regions = len(my_regions)
        while len(my_regions) > 0:
            t3 = time.time()
            if len(my_regions) > slice_max:
                my_slice = slice_max
            else:
                my_slice = len(my_regions)
            print(
                f"Create key data for regions {my_regions[:my_slice]} at scale {scale}:"
            )
            if sim_progress:
                sim_progress["text"] = "Generate key data for simulation"
            t1 = time.time()
            people = p.People(my_regions[:my_slice], scale)
            people.initialize_data()
            t2 = time.time()
            print(f"    people class: {round(t2 - t1, 2)}s")
            if sim_progress:
                sim_progress["progress"] += 5 * my_slice / total_num_regions
            # Todo: do not make corridor/ random contacts < 2
            t1 = time.time()
            households = Households(people, scale)
            t2 = time.time()
            print(f"    {type(households).__name__} class: {round(t2 - t1, 2)}s")
            if sim_progress:
                sim_progress["progress"] += 70 * my_slice / total_num_regions
            t1 = time.time()
            schools = Schools(people, scale)
            t2 = time.time()
            print(f"    {type(schools).__name__} class: {round(t2 - t1, 2)}s")
            if sim_progress:
                sim_progress["progress"] += 10 * my_slice / total_num_regions
            t1 = time.time()
            works = Workplaces(people, scale)
            t2 = time.time()
            print(f"    {type(works).__name__} class: {round(t2 - t1, 2)}s")
            if sim_progress:
                sim_progress["progress"] += 15 * my_slice / total_num_regions
            t1 = time.time()
            activities = Activities(people, scale)
            t2 = time.time()
            print(f"    {type(activities).__name__} class: {round(t2 - t1, 2)}s")
            if sim_progress:
                sim_progress["progress"] += 10 * my_slice / total_num_regions
            t1 = time.time()
            agents_networks = Transformer.network_ids_to_people(
                people, households, schools, works, activities
            )
            t2 = time.time()
            print(f"    Transformer: {round(t2 - t1, 2)}s")
            savingThread = threading.Thread(
                target=DataManager.save_generated_data_to_file,
                args=(
                    agents_networks,
                    people,
                    scale,
                    households,
                    schools,
                    works,
                    activities,
                ),
            )
            savingThread.start()
            savingThread.join()
            t4 = time.time()
            print(f"Create key data: {round(t4 - t3, 2)}s")
            my_regions = np.delete(my_regions, np.s_[:my_slice])

    @staticmethod
    def save_generated_data_to_file(my_dict, people, scale, *networks):
        savingLock = threading.Lock()
        savingLock.acquire()
        Transformer.save_agents_networks_to_file(my_dict, scale)
        people.save_agents_data_to_file()
        for network in networks:
            network.save_networks_to_file()
        DataManager.save_generated_regions_to_info_file(people.get_regions(), scale)
        savingLock.release()

    @staticmethod
    def save_generated_regions_to_info_file(new_regions, scale):
        old_regions = DataManager.get_old_regions(scale)
        all_regions = np.append(new_regions, old_regions)
        assert len(np.unique(all_regions)) == len(all_regions)
        abs_file_name = PM.get_path_info_file(scale)
        np.save(abs_file_name, all_regions)

    @staticmethod
    def adopt_network_data(net_type):
        scale = 100
        DataManager.reset_network_data(net_type, scale)
        my_regions = DataManager.get_old_regions(scale)
        # TODO: iterate all scales
        scale = 100
        while len(my_regions) > 0:
            if len(my_regions) > 30:
                my_slice = 30
            else:
                my_slice = len(my_regions)
            t3 = time.time()
            people = p.People(my_regions[:my_slice], scale=scale)
            people.reload_data()
            if net_type == cs.HOUSEHOLDS:
                net_object = Households(people, scale)
            elif net_type == cs.WORKPLACES:
                net_object = Workplaces(people, scale)
            elif net_type == cs.SCHOOLS:
                net_object = Schools(people, scale)
            elif net_type == cs.ACTIVITIES:
                net_object = Activities(people, scale)
            else:
                print("no valid network name")
                return
            Transformer.add_network_ids(scale, net_object)
            net_object.save_networks_to_file()
            t4 = time.time()
            print(f"total time: {t4-t3}")
            my_regions = np.delete(my_regions, np.s_[:my_slice])
        print(
            f"Recreate network data {net_type} for regions {my_regions} at scale"
            f" {scale}"
        )

    @staticmethod
    def reset_network_data(net_type, scale):
        if net_type == cs.HOUSEHOLDS:
            digits = [1]
        elif net_type == cs.WORKPLACES:
            digits = [2]
        elif net_type == cs.SCHOOLS:
            digits = [3, 4, 5]
        elif net_type == cs.ACTIVITIES:
            digits = [6]
        else:
            digits = Sync.BUILDING_DIGITS
        Institution.reset_data(digits, scale)
        Transformer.delete_network_ids(digits, scale)

    @staticmethod
    def reset_data(scale):
        path_list = [
            Config.AGENT_FILE_DIR,
            Config.NETWORK_FILE_DIR,
            Config.INFO_FILE_DIR,
        ]
        for my_path in path_list:
            for file in os.listdir(my_path):
                if str(scale) in file:
                    os.remove(os.path.join(my_path, file))

    @staticmethod
    def check_for_new_regions_to_create(regions, scale):
        old_regions = DataManager.get_old_regions(scale)
        indices = np.nonzero(np.isin(regions, old_regions, invert=True))[0]
        new_regions = regions[indices]
        if len(new_regions) > 0:
            create_regions = True
        else:
            create_regions = False
        return create_regions, new_regions

    @staticmethod
    def get_old_regions(scale):
        abs_file_name = PM.get_path_info_file(scale)
        if not os.path.exists(abs_file_name):
            old_regions = np.arange(0)
        else:
            old_regions = np.load(abs_file_name, allow_pickle=True)
        return old_regions


class Transformer:
    """Translate net_manager to people."""

    def __init__(self):
        pass

    @staticmethod
    def network_ids_to_people(people, *networks):
        ids = people.get_data_for(cs.ID)
        my_dict = {}
        for a_id in ids:
            my_dict[a_id] = []
        Transformer.add_network_ids_to_agents(my_dict, *networks)
        return my_dict

    @staticmethod
    def delete_network_ids(digits, scale):
        abs_file_name = PM.get_path_agents_networks(scale)
        if os.path.exists(abs_file_name):
            my_dict = FileLoader.load_dict_from_npy(abs_file_name)
        else:
            my_dict = {}
        Transformer.delete_network_ids_from_agents(my_dict, digits)
        FileSaver.save_dict_to_npy(abs_file_name, my_dict)

    @staticmethod
    def add_network_ids(scale, *networks):
        abs_file_name = PM.get_path_agents_networks(scale)
        agents_nets = FileLoader.load_dict_from_npy(abs_file_name)
        Transformer.add_network_ids_to_agents(agents_nets, *networks)
        FileSaver.save_dict_to_npy(abs_file_name, agents_nets)

    @staticmethod
    def add_network_ids_to_agents(agents_nets, *networks):
        for network in networks:
            for net_id, agent_ids in network.members_groups.items():
                for a_id in agent_ids:
                    agents_nets[a_id].append(net_id)

    @staticmethod
    def delete_network_ids_from_agents(agents_nets, digits):
        for key, net_ids in agents_nets.items():
            new_net_ids = []
            for idx, net_id in enumerate(net_ids):
                if int(str(net_id)[0]) not in digits:
                    new_net_ids.append(net_id)
            agents_nets[key] = new_net_ids

    @staticmethod
    def find_digits(*networks):
        digits = {
            "households": 1,
            "workplaces": 2,
            "schools": [3, 4, 5],
            "activities": 6,
        }
        my_digits = []
        for network in networks:
            net_name = type(network).__name__.lower()
            digit = digits[net_name]
            my_digits.append(digit)
            return my_digits

    @staticmethod
    def save_agents_networks_to_file(my_dict, scale):
        abs_file_name = PM.get_path_agents_networks(scale)
        if os.path.exists(abs_file_name):
            agents_nets = FileLoader.load_dict_from_npy(abs_file_name)
            agents_nets = {**agents_nets, **my_dict}
        else:
            agents_nets = my_dict
        FileSaver.save_dict_to_npy(abs_file_name, agents_nets)


test = 1
if __name__ == "__main__":
    # Todo: file too large (1.5 MB)
    DataManager.adopt_network_data(cs.HOUSEHOLDS)
    DataManager.adopt_network_data(cs.SCHOOLS)
    DataManager.adopt_network_data(cs.WORKPLACES)
    DataManager.adopt_network_data(cs.ACTIVITIES)
