#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Docstring."""

__author__ = "Inga Franzen"
__created__ = "2021"
__date_modified__ = "2023/05/31"
__version__ = "1.0"

import pathlib

import numpy as np
import pandas as pd
from numpy import random as rm

from data_factory.generator.networks_d import Institution
from synchronizer import constants as cs
from synchronizer.synchronizer import PathManager as PM
from synchronizer.synchronizer import Synchronizer as Sync

# 01	Schleswig-Holstein
# 02	Hamburg
# 03	Niedersachsen
# 04	Bremen
# 05	Nordrhein-Westfalen
# 06	Hessen
# 07	Rheinland-Pfalz
# 08	Baden-Württemberg
# 09	Bayern
# 10	Saarland
# 11	Berlin
# 12	Brandenburg
# 13	Mecklenburg-Vorpommern
# 14	Sachsen
# 15	Sachsen-Anhalt
# 16	Thüringen

abs_path = pathlib.Path(__file__).parent.parent.absolute()


class Schools(Institution):
    def __init__(self, people, scale=None):
        """Initialize schools with id and information on location, size and
        people.

        :param people: people class object
        :param scale: 10 or 100
        """
        super(Schools, self).__init__(people, scale)
        self.abs_path = pathlib.Path(__file__).parent.parent.absolute()
        all_agents_data = self.extract_pupils_and_edus_from_people_class(people)

        self.school_data = pd.read_excel(
            PM.get_path_school_table(), sheet_name="type_sizes"
        )
        # create medium school sizes for each federal called via index 1-16,
        # Source: Statistisches Bundesamt, Schulen auf einen Blick, 2018, p. 39

        fed_sc_size = pd.read_excel(
            PM.get_path_school_table(),
            sheet_name="federal_sizes",
        )["sizes"].values
        self.sc_size_factors = fed_sc_size / fed_sc_size[0]  # divide by mean

        # load data for secondary schools
        self.p_5_10 = pd.read_excel(PM.get_path_school_table(), sheet_name="pupil_5_10")
        self.id = int(10**10 / self.scale)
        # iterate regions
        for region in self.regions:
            agents_data = self.filter_agents_data_by_subdata(
                cs.Z_CODE, region, all_agents_data
            )
            self.make_kitas_for_one_region(region, agents_data)
            self.make_schools_for_one_region(region, agents_data)
            self.make_colleges_for_one_region(region, agents_data)
        self.add_mean_contacts()
        # self.determine_mean_number_of_contacts()
        # TODO:(Jan) ich hoffe, ich habe den Sinn der beiden folgen Methoden richtig verstanden
        self.determine_mean_contacts_region()
        self.save_current_mean_contacts()

    def make_kitas_for_one_region(self, region, agents_data):
        num_kita = self.filter_ages(agents_data, 0, 6).sum()
        # TOOD: (Jan) Bedeutung von 3?
        id_count = 3 * self.id + int(region * 10**5 / self.scale)
        # TODO:(Jan) ich sehe nicht, wiviele Kitas erzeugt werden?
        _ = self.make_one_school_type(region, agents_data, "kita", num_kita, id_count)
        # TODO:(Jan) sicher, dass where benötigt wird und nicht (agents_data[cs.OCCUPATION] == "k").sum() funktioniert?
        assert np.where(agents_data[cs.OCCUPATION] == "k", 1, 0).sum() == 0

    def make_schools_for_one_region(self, region, agents_data):
        """# Sources: Statistisches Bundesamt, Schulen auf einen Blick, 2018,
        p.

        38 # 21111-0011.xlsx, Statistik der allgemeinbildenden Schulen,
        Statistisches Bundesamt (Destatis), 2021 | Stand: 21.07.2021 #
        to simplify the data it is assumed to have 4 different
        structural types # Type1: year 1-4 (Grundschulen) # Type2: year
        5-10, size small (Förderschulen + Hauptschulen) # Type3: year
        5-10, size medium (Realschulen + Schularten mit mehreren
        Bildungsgängen) # Type4: year 5-13, (Gymnasien +
        Schulartunabhängige Orientierungsstufen + Integrierte
        Gesamtschulen # + Freie Waldorfschulen)
        """

        federal_key = int(region / 1000)
        # get real number of people
        num_primary = self.filter_ages(agents_data, 7, 10).sum()
        num_secondary = self.filter_ages(agents_data, 11, 19).sum()

        # case 1: scale = 100 or small regions
        if num_secondary < sum(self.school_data["size"][1:4]):
            # TODO:(Jan) war erst verwundert, dass dreimal "all_in" -> if/else tauschen?
            num_big = "all_in"
            num_medium = "all_in"
            num_small = "all_in"
        # case 2: scale = 10 or scale = 100 and metropole
        else:
            p_big = self.p_5_10.at[federal_key, "mit Oberstufe"]
            num_big = int(num_secondary * p_big)
            p_medium = self.p_5_10.at[federal_key, "ohne Oberstufe, groß"]
            num_medium = int(num_secondary * p_medium)
            num_small = num_secondary - num_big - num_medium

        # TODO:(Jan) 4 heißt Schule?
        id_count = 4 * self.id + int(region * 10**5 / self.scale)
        id_count = self.make_one_school_type(
            region, agents_data, "primary", num_primary, id_count
        )
        id_count = self.make_one_school_type(
            region, agents_data, "secondary big", num_big, id_count
        )
        id_count = self.make_one_school_type(
            region, agents_data, "secondary medium", num_medium, id_count
        )
        _ = self.make_one_school_type(
            region, agents_data, "secondary small", num_small, id_count
        )
        # TODO:(Jan) agents_data wird doch nicht geändert? Prüfung am Anfang der Methode?
        # assert success
        cond1 = agents_data[cs.AGE] >= 0
        cond2 = agents_data[cs.AGE] <= 19
        assert np.where(cond1 & cond2, 1, 0).sum() == 0

        return

    def make_one_school_type(self, region, agents_data, sc_type, num_pupil, id_count):
        """Create array from table.

        Args:
            agents_data(dict(ndarry)): Hold ids, ages, sexes, z_codes, landscapes for people.

        Returns:
        """
        (
            cl_dist,
            cl_probs,
            mean_cl_size,
            last,
            max_age,
            min_age,
            num_schools,
            p_per_t,
        ) = self.determine_school_data(num_pupil, region, sc_type)
        if self.filter_ages(agents_data, min_age, max_age).sum() == 0:
            return
        for num in range(0, num_schools):
            sc_id = id_count = self.register_school(id_count, region, sc_type)
            # sample from class dist
            mean_cl_num = int(rm.choice(cl_dist, 1, p=cl_probs))
            last = self.check_last(last, num, num_schools)
            # iterate age groups
            for age in range(min_age, max_age + 1):
                year = self.check_year(age, sc_type)
                people_available = self.check_people_available(age, agents_data)
                # if no people available, continue with next age group
                if people_available == 0:
                    continue
                # adopt class size for that age group according to real number
                # of people in that age group
                cl_size, cl_num = self.adopt_class_structure_to_real_number_of_people(
                    mean_cl_size, mean_cl_num, people_available, last
                )
                # iterate classes
                for class_num in range(0, cl_num):
                    id_count += 1
                    self.register_class(cl_size, id_count, sc_id, year=year)
                    self.add_pupil(age, agents_data, id_count, sc_id, cl_size, year)
                    # add educators, at kita the educators are for one group
                    self.add_educators(
                        id_count, sc_id, sc_type, agents_data, cl_size, p_per_t
                    )

            size = self.get_school_size(sc_id)
            # create other groups for school such as cafeteria or free activity
            id_count = self.create_other_activities_at_institution(
                sc_id, id_count, size, sc_type
            )

            # add teachers, at school the teachers are for several classes and activity
            self.add_teachers(sc_id, id_count, sc_type, agents_data, size, p_per_t)
            self.register_value(sc_id, self.get_school_size(sc_id))

        return id_count

    def make_colleges_for_one_region(self, region, agents_data):
        # extract data
        ids = agents_data[cs.ID]
        occ = agents_data[cs.OCCUPATION]
        assert self.filter_ages(agents_data, 0, 16).sum() == 0
        # get and create sizes
        students = ids[occ == "p"]
        num_students = len(students)
        min_size = 5
        max_size = min(num_students, 1750)  # 1750 mean size of big lecture room
        mean_size = (max_size + min_size) / 2
        # TODO:(Jan) jeder Student hat als 5 Vorlesungen pro Woche? Wenn ja, dann scheint die Zahl sehr klein.
        num_p_week = 5
        # TODO:(Jan) die Näherung num_of_events über diese mean_size zu berechnen wirkt falsch. Kann nicht überblicken, wie falsch
        num_of_events = int(num_students * num_p_week / mean_size)
        # TODO:(Jan) 5 ist die Zahl fpr collage?
        id_count = 5 * self.id + int(region * 10**5 / self.scale)
        # create id --> it is assumed 1 college/university per region
        sc_id = self.register_school(id_count, region, "college", len(ids))
        self.members_institutions[sc_id] = ids
        size = self.get_school_size(sc_id)
        # create classes
        event_ids = np.arange(id_count + 1, id_count + 1 + num_of_events)
        sizes = rm.choice(range(min_size, max_size + 1), num_of_events)
        # iterate events and save to dataframe and dict
        for idx, event in enumerate(event_ids):
            id_count += 1
            st_ids = rm.choice(students, sizes[idx], replace=False)
            # TODO:(jan) was sagt mir das?
            times_p_week = rm.choice([1, 2], 1, p=[0.4, 0.6])
            wd = rm.choice([0, 1, 2, 3, 4], times_p_week)
            self.register_class(sizes[idx], id_count, sc_id, type="lecture", weekday=wd)
            self.add_students_and_remove(st_ids, id_count, agents_data)
        # add teachers
        ids = agents_data[cs.ID]
        teachers = self.sample_teachers_and_remove(ids, "all_in", agents_data)
        # self.add_teachers_to_school(sc_id, teachers)
        # TODO:(Jan) 2 Lehrer?
        self.add_teachers_to_classes(id_count, sc_id, teachers, 2)
        # create other groups for school such as cafeteria or free activity
        _ = self.create_other_activities_at_institution(
            sc_id, id_count, size, "college"
        )
        self.register_value(sc_id, size)
        return

    def add_mean_contacts(self):
        # TODO: (Jan) Haben die Zahlen einen Hintergrund oder sind sie gut geschätzt?
        mean_contacts = {
            "class": 3,
            "lecture": 2,
            "mensa": 3,
            "free": 3,
            "unspecific": 1,
        }
        super(Schools, self).add_mean_contacts(**mean_contacts)

    def get_school_size(self, sc_id):
        size = len(self.members_institutions[sc_id])
        return size

    def filter_ages(self, agents_data, min, max):
        cond1 = agents_data[cs.AGE] >= min
        cond2 = agents_data[cs.AGE] <= max
        # TODO:(Jan) maybe "return min <= agents_data[cs.AGE] <= max" not tested if it works with pdDF(?)
        return np.where(cond1 & cond2, 1, 0)

    def check_last(self, last, num, num_schools):
        # TODO:(Jan) die Methode returd also immer >0 also True, solange last nicht 0 ist? Das ergibt wenig sinn
        # Bzw. falls last iterable sein sollte(?) immer dann, wenn liste nicht lehr
        # "return num == num_schools -1" ?
        if num == num_schools - 1:  # note if last school is iterated
            last = True
        return last

    def check_people_available(self, age, agents_data):
        # check number of people available
        ages = agents_data[cs.AGE]  # extract data
        # TODO:(Jan) people_available = (ages == age).sum()
        # Bzw.  people_available = (agents_data[cs.AGE] == age).sum()
        # TODO:(Jan) hier "where" letztes mal angemerkt
        people_available = np.where(ages == age, 1, 0).sum()
        return people_available

    def check_year(self, age, sc_type):
        # TODO: (Jan) option: year = np.nan if sc_type == "kita" else age-6
        # TODO:(jan) methoden-name. year is offensichtlich class?
        if sc_type == "kita":
            year = np.nan
        else:
            year = age - 6
        return year

    def determine_school_data(self, num_pupil, region, sc_type):
        federal_key = int(region / 1000)

        # extract general school data
        filt = self.school_data["type"] == sc_type
        size = self.school_data.loc[filt, "size"].values[0]
        mean_sc_size = int(size * self.sc_size_factors[federal_key])

        # medium school size for federal
        min_age = self.school_data.loc[filt, "min_age"].values[0]
        max_age = self.school_data.loc[filt, "max_age"].values[0]
        mean_cl_size = self.school_data.loc[filt, "class_size"].values[0]
        p_per_t = self.school_data.loc[filt, "pupil_per_teacher"].values[0]

        # because number of age 0 is so low and average numbers are distortet
        if sc_type == "kita":
            years = max_age - min_age
        else:
            years = max_age - min_age + 1
        # this is to evenly distribute class sizes
        (
            cl_dist,
            cl_probs,
            num_s,
        ) = self.determine_sc_and_cl_num(mean_cl_size, num_pupil, mean_sc_size, years)
        # iterate schools
        last = False
        # if no pupils left
        return cl_dist, cl_probs, mean_cl_size, last, max_age, min_age, num_s, p_per_t

    def extract_pupils_and_edus_from_people_class(self, people):
        all_agents_data = people.get_agents_data()
        occ = people.get_data_for(cs.OCCUPATION)
        # TODO:(Jan) hab ne ahnung, was p,e,k abfrage macht
        cond1 = occ == "p"
        cond2 = occ == "e"
        cond3 = occ == "k"
        idx = np.argwhere(
            np.where(
                cond1 | cond2 | cond3,
                1,
                0,
            )
        )
        all_agents_data = self.extract_data_of_specified_indices(idx, all_agents_data)
        return all_agents_data

    def determine_sc_and_cl_num(self, class_size, num_pupil, sc_size, years):
        # TODO:(jan) unklare methode; besonders methodenname
        # determine classes per year
        if num_pupil == "all_in" or num_pupil < sc_size:
            num_schools = 1
        else:
            num_schools = int(np.ceil(num_pupil / sc_size))
        temp = sc_size / (class_size * years)
        lower = np.floor(temp)
        upper = np.ceil(temp)
        prob = temp - lower
        # cl_per_year will be randomly choosen for each school
        if lower != upper:
            cl_per_year_choice = [lower, upper]
            cl_per_year_probs = [round(1 - prob, 2), round(prob, 2)]
        else:
            cl_per_year_choice = [temp]
            cl_per_year_probs = [1]

        return cl_per_year_choice, cl_per_year_probs, num_schools

    def adopt_class_structure_to_real_number_of_people(
        self, cl_size, cl_per_year, people_available, last=False
    ):
        temp_cl_size = cl_size
        # case 1: not enough people to fill number of schools from statistics
        if people_available < temp_cl_size * cl_per_year:
            while people_available < temp_cl_size * cl_per_year:
                temp_cl_size -= 1
                if temp_cl_size < cl_size * 0.80:
                    cl_per_year = max(1, cl_per_year - 1)
                    if cl_per_year == 1:
                        temp_cl_size = people_available
                    elif cl_size * cl_per_year < people_available:
                        temp_cl_size = int(np.ceil(people_available / cl_per_year))
                    else:
                        temp_cl_size = cl_size
            if people_available % (temp_cl_size * cl_per_year) != 0:
                temp_cl_size += 1  # because of broken numbers rounded to integer
        # case 2: too many people left when last school from statistic is filled
        elif (people_available > temp_cl_size * cl_per_year) & (last is True):
            while people_available > temp_cl_size * cl_per_year:
                temp_cl_size += 1
                if temp_cl_size > cl_size * 1.20:
                    cl_per_year += 1
                    if cl_size * cl_per_year > people_available:
                        temp_cl_size = int(np.ceil(people_available / cl_per_year))
                    else:
                        temp_cl_size = cl_size
        else:
            pass

        return temp_cl_size, cl_per_year

    def register_value(self, sc_id, value):
        # now we know real school size
        self.networks[self.networks[:, 0] == sc_id, 2] = value

    def register_school(self, id_count, region, sc_type, size=np.nan):
        # only not the case for first school
        if not int(np.ceil(id_count / 1000) * 1000) == id_count:
            id_count += 1
        sc_id = int(np.ceil(id_count / 10) * 10)
        self.members_institutions[sc_id] = np.arange(0)
        super(Schools, self).append_institution_data(sc_id, sc_type, size)
        return sc_id

    def register_class(
        self, size, id_count, sc_id, type="class", year=np.nan, weekday=np.nan
    ):
        super(Schools, self).append_group_data(
            sc_id, id_count, type=type, size=size, year=year, weekday=weekday
        )

    def add_pupil(self, age, agents_data, id_count, sc_id, cl_size, year):
        members = self.sample_pupil_and_remove(age, agents_data, cl_size)
        self.add_pupil_to_school(members, sc_id)
        self.add_pupil_to_class(members, id_count)

    def add_pupil_to_class(self, cl_members, id_count):
        # add pupil to dictionary and class data to Dataframe
        self.members_groups[id_count] = cl_members

    def add_pupil_to_school(self, cl_members, sc_id):
        # add pupil to school
        self.members_institutions[sc_id] = np.hstack(
            (self.members_institutions[sc_id], cl_members)
        )

    def sample_pupil_and_remove(self, age, agents_data, cl_size):
        # extract data
        ids = agents_data[cs.ID]
        ages = agents_data[cs.AGE]
        # sample from pool
        my_ids = ids[(ages == age)]
        if len(my_ids) >= cl_size:
            ids_sampled = rm.choice(my_ids, cl_size, replace=False)
        else:
            # will only be the case due to broken numbers taken to integer from method
            # "adopt_class_structure_to_real_number_of_people"
            people_available = np.where(ages == age, 1, 0).sum()
            ids_sampled = rm.choice(my_ids, people_available, replace=False)
        # remove agents from pool and extract data again for next loop
        agents_data, _ = super(Schools, self).split_sampled_data_from_agents_data(
            ids, ids_sampled, agents_data
        )
        return ids_sampled

    def add_students_and_remove(self, st_ids, id_count, agents_data):
        self.add_pupil_to_class(st_ids, id_count)
        # remove agents from pool
        ids = agents_data[cs.ID]
        agents_data, _ = super(Schools, self).split_sampled_data_from_agents_data(
            ids, st_ids, agents_data
        )
        return

    def add_teachers(self, sc_id, id_count, sc_type, agents_data, size=0, p_per_t=1):
        if sc_type == "kita":
            return
        elif sc_type == "college":
            # number of teachers
            num_p_cl = 2
            num_p_sc = "all_in"
        else:
            # number of teachers
            num_p_cl = 3
            # 1.5 is estimated adaptation
            num_p_sc = int(np.ceil(size / p_per_t * 1.5))
        ids = agents_data[cs.ID]
        teachers = self.sample_teachers_and_remove(ids, num_p_sc, agents_data)
        self.add_teachers_to_school(sc_id, teachers)
        self.add_teachers_to_classes(id_count, sc_id, teachers, num_p_cl)
        return

    def add_teachers_to_classes(self, id_count, sc_id, teachers, num_t):
        for cl_id in range(sc_id + 1, id_count):
            assert cl_id in self.members_groups
            # very rare case, but sometimes happen
            if len(teachers) < num_t:
                num_t = len(teachers)
            cl_t = rm.choice(teachers, num_t, replace=False)
            self.members_groups[cl_id] = np.hstack((self.members_groups[cl_id], cl_t))

    def add_teachers_to_school(self, sc_id, teachers):
        self.members_institutions[sc_id] = np.hstack(
            (self.members_institutions[sc_id], teachers)
        )

    def sample_teachers_and_remove(self, ids, num_t, agents_data):
        if len(ids) == 0:
            # actually this shoul never happen, but happen with some seed
            # print("No IDs left")
            return ids
        occ = agents_data[cs.OCCUPATION]
        if num_t == "all_in":
            # print(f"region: {int(ids[0]/10**5)}, total edus: {len(occ)}, number to sample: {num_t}")
            teachers = ids[(occ == "e")]
        # sample teachers
        else:
            if len(ids[(occ == "e")]) < num_t:
                num_t = len(ids[(occ == "e")])
            teachers = rm.choice(ids[(occ == "e")], num_t, replace=False)
        # remove agents from pool
        agents_data, _ = super(Schools, self).split_sampled_data_from_agents_data(
            ids, teachers, agents_data
        )
        return teachers

    def add_educators(self, cl_id, sc_id, sc_type, agents_data, cl_size, p_per_t):
        if sc_type != "kita":
            return
        # number of educators
        num_e = int(np.ceil(cl_size / p_per_t))
        # extract data
        ids = agents_data[cs.ID]
        edus = self.sample_teachers_and_remove(ids, num_e, agents_data)
        self.add_teachers_to_school(sc_id, edus)
        self.add_educators_to_class(cl_id, edus)
        return

    def add_educators_to_class(self, cl_id, edus):
        self.members_groups[cl_id] = np.hstack((self.members_groups[cl_id], edus))

    def save_networks_to_file(self):
        # save dict to npy file
        self.save_school_type_to_file(cs.KITAS, 3)
        self.save_school_type_to_file(cs.SCHOOLS, 4)
        self.save_school_type_to_file(cs.UNIVERSITIES, 5)

    def save_school_type_to_file(self, net_type, short_id):
        members = self.filter_members_by_net_type(short_id, self.scale)
        self.save_members_to_file(members, net_type)
        # TODO:(Jan) warum hier nicht einfach members verwendet?
        # TODO: (Jan) für folgende 6 Zeilen for loop "for sub in [False, True]"
        my_networks = self.filter_networks_by_net_type(short_id, self.scale)
        my_networks = self.numpy_to_dataframe(my_networks)
        self.save_dataframe_to_file(my_networks, net_type)
        # TODO:(Jan) warum hier nicht einfach members verwendet?
        my_sub_nets = self.filter_subnets_by_net_type(short_id, self.scale)
        my_sub_nets = self.numpy_to_dataframe(my_sub_nets, sub=True)
        self.save_dataframe_to_file(my_sub_nets, net_type, sub=True)

    def reset_data(self):
        path = f"{abs_path}/data_formatted"
        # save empty data
        df_net = pd.DataFrame(columns=Sync.NETWORK_COLUMNS.keys())
        df_sub = pd.DataFrame(columns=Sync.SUB_NET_COLUMNS.keys())
        empty_dict = {}
        for net_type in [cs.SCHOOLS, cs.KITAS, cs.UNIVERSITIES]:
            df_net.to_csv(f"{path}/{net_type}_{self.scale}.csv")
            df_sub.to_csv(f"{path}/{net_type}_{self.scale}_subnet.csv")
            np.save(f"{path}/{net_type}_networks_{self.scale}_subnet.npy", empty_dict)

    def filter_subnets_by_net_type(self, short_id, scale):
        cond1 = self.sub_nets[:, 0] >= short_id * 10**10 / scale
        cond2 = self.sub_nets[:, 0] < (short_id + 1) * 10**10 / scale
        my_sub_nets = self.sub_nets[cond1 & cond2]
        return my_sub_nets

    def filter_networks_by_net_type(self, short_id, scale):
        cond1 = self.networks[:, 0] >= short_id * 10**10 / scale
        cond2 = self.networks[:, 0] < (short_id + 1) * 10**10 / scale
        return self.networks[cond1 & cond2]

    def filter_members_by_net_type(self, short_id, scale):
        return dict(
            filter(
                lambda x: int(x[0] / 10**10 * scale) == short_id,
                self.members_groups.items(),
            )
        )

    def create_other_activities_at_institution(self, sc_id, id_count, size, sc_type):
        id_count = super(Schools, self).create_unspecific_contacts(
            sc_id, id_count, size
        )
        if sc_type != "kita":
            id_count = super(Schools, self).create_mensa(sc_id, id_count, size)
            if sc_type == "college":
                mean_p_agent = 0.3
            else:
                mean_p_agent = 0.5
        else:
            mean_p_agent = 0.2

        id_count = super(Schools, self).create_other_groups(
            sc_id, id_count, size, "free", mean_p_agent=mean_p_agent
        )
        return id_count
