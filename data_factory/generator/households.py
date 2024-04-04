#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Docstring."""

__author__ = "Inga Franzen"
__created__ = "2021"
__date_modified__ = "2023/05/31"
__version__ = "1.0"

import math
import pathlib
import time

import numpy as np
import pandas as pd
# from line_profiler_pycharm import profile

from data_factory.generator.networks_d import Institution
from synchronizer import constants as cs
from synchronizer.synchronizer import FileLoader
from synchronizer.synchronizer import PathManager as PM


class Households(Institution):
    # @profile
    def __init__(self, people, scale=None):
        """Initialize households with id and information on location, size and
        inhabitants (people)

        :param regions: list with regional id
        :param scale: 10 or 100
        """
        t1 = time.time()
        super(Households, self).__init__(people, scale)
        self.abs_path = pathlib.Path(__file__).parent.parent.absolute()
        self.ids = np.zeros(0, dtype=np.int64)
        self.z_codes = np.zeros(0, dtype=int)
        self.sizes = np.zeros(0, dtype=int)
        self.landsc = np.zeros(0, dtype=int)
        self.contacts = np.zeros((0, 25), dtype=int)
        # only for testing
        self.contacts1 = np.zeros(0, dtype=int)
        self.scale = scale
        # set attributes for household object
        all_agents_data = people.get_agents_data()
        occupations = people.get_data_for(cs.OCCUPATION)
        # TODO:(jan) warum wird hier np.where benötigt?
        indices = np.argwhere(np.where(occupations == "in need of care", 1, 0))
        all_agents_data = super(Households, self).delete_indices(
            indices, all_agents_data
        )
        self.base_id = np.int64(1 * 10**11 / self.scale)
        for region in self.regions:
            t2 = time.time()
            # get copy of filtered agents data from people class
            agents_data = self.filter_agents_data_by_subdata(
                cs.Z_CODE, region, all_agents_data
            )
            del agents_data[cs.OCCUPATION]

            hh_data = self.make_household_data(region, agents_data)
            # pack all household data to one array (z_codes not needed here)
            hh_data, contacts = self.initialize_household_contacts(
                region, agents_data, hh_data
            )
            # set ids
            start_id = self.base_id + np.int64(region * np.int64(10**6) / self.scale)
            end_id = start_id + len(hh_data[0])
            ids = np.arange(start_id, end_id)
            self.z_codes = np.hstack((self.z_codes, hh_data[0]))
            self.sizes = np.hstack((self.sizes, hh_data[1]))
            self.landsc = np.hstack((self.landsc, hh_data[2]))
            # self.contacts is only for the tests
            self.contacts = np.vstack((self.contacts, contacts))
            self.ids = np.hstack((self.ids, ids))
        self.sizes = np.count_nonzero(self.contacts, axis=1)
        # self.members_groups is for saving the data effectively
        self.members_groups = {}
        for idx in range(len(self.ids)):
            temp = set(self.contacts[idx][:])
            temp.remove(0)
            self.members_groups[self.ids[idx]] = np.array(list(temp))
        self.transfer_data_to_sub_nets()
        self.determine_mean_contacts_region()
        self.save_current_mean_contacts()

    def check_time(self, t1):
        # TODO: (jan) print(f"time: {time.time() - t1} seconds")
        t5 = time.time()
        t6 = t5 - t1
        print("time: {} seconds".format(t6))

    def make_household_data(self, region, agents_data):
        """read."""
        t_2 = time.time()
        # read files
        df_states = FileLoader.read_excel_table(PM.get_path_states())
        df_states = df_states.drop(df_states.index[-1])
        df_states.set_index("Schlüsselnummer", inplace=True)
        df_counties = FileLoader.read_excel_table(PM.get_path_counties())
        df_counties.set_index("Schlüsselnummer", inplace=True)
        t_3 = time.time()

        county_pop = int(df_counties.loc[region, "Bevölkerung"] / self.scale)
        county_urban_pop = int(df_counties.loc[region, "städtisch"] / self.scale)
        county_rural_pop = int(df_counties.loc[region, "ländlich"] / self.scale)
        assert (
            county_pop - county_urban_pop
            <= county_rural_pop + 1 | county_pop - county_urban_pop
            >= county_rural_pop - 1
        )

        # extract number of federal population
        federal_key = int(region / 1000)
        federal_pop = int(df_states.at[federal_key, "Bevölkerung"] / self.scale)

        # get percentage of county population to federal pop
        per_county_to_federal = county_pop / federal_pop

        # get percentage urban and rural population
        per_urban = county_urban_pop / (county_rural_pop + county_urban_pop)
        # TODO: (jan) Vorschlag: per_rural = 1-per_urban
        per_rural = county_rural_pop / (county_rural_pop + county_urban_pop)

        federal_pop = int(df_states.at[federal_key, "Bevölkerung"]) / self.scale
        #
        # extract federal household data
        t_2 = time.time()
        df_households = FileLoader.read_excel_table(
            PM.get_path_household_table(), str(federal_key)
        )
        df_households.replace("-", np.nan, inplace=True)
        df_households = df_households.fillna(0)
        df_households.set_index("Haushaltsgröße", inplace=True)
        df_households.rename(
            index={
                df_households.index[-1]: df_households.index[-1].translate(
                    {ord(">"): None}
                )
            },
            inplace=True,
        )
        t_3 = time.time()

        # iterate householde sizes
        nd_sizes = np.arange(0)
        nd_struct = np.arange(0)
        num_of_ids = 0
        for size in df_households.index:
            # determine number of households for rural and urban
            # TODO: (Jan) ist richtig. Sortiere in beiden Fällen die Rechenoperationen gleich;
            # TODO:(Jan) GGF: (damit direkt Unterschied zw number_rual und number_urban ersichtlich wird): x = df_[hosuholtst](..) * per_country/self.scale // number_rual = x*per_rural, //number_urban = x*per_urban
            # TODO: (Jan) das hier abgerundet wird ist gewollt?
            number_rural_size = (
                df_households.loc[size, "Gesamt"]
                * per_rural
                * per_county_to_federal
                / self.scale
            ).astype(int)
            number_urban_size = (
                df_households.loc[size, "Gesamt"]
                * per_urban
                / self.scale
                * per_county_to_federal
            ).astype(int)

            # create array with structural data
            # TODO: (Jan) was ist 60/30? Kann sich der Wert ändern? Dann in Variable, damit er dann nicht überall geändet werden muss
            rural_struct = np.full_like(
                np.arange(number_rural_size), 60, dtype=np.int64
            )
            urban_struct = np.full_like(
                np.arange(number_urban_size), 30, dtype=np.int64
            )
            nd_struct_temp = np.hstack((urban_struct, rural_struct))

            # create array with size
            nd_sizes_temp = np.full_like(nd_struct_temp, size, dtype=np.int64)

            # stack eith other sizes
            nd_sizes = np.hstack((nd_sizes, nd_sizes_temp))
            nd_struct = np.hstack((nd_struct, nd_struct_temp))

            # counter for households
            num_of_ids += number_urban_size + number_rural_size

        # TODO:(Jan) modify comment; z_code is 10**11; housholds are region*10**6?
        # TODO:(Jan) irgenwo cKommentar, wie sich die ids zusammensetzen
        # create array for z_code and household ids
        nd_regions = np.full_like(np.arange(num_of_ids), region)
        nd_ids = (
            np.arange(num_of_ids, dtype="int64")
            + np.int64(1 * 10**11) / self.scale
            + region * np.int64(10**6) / self.scale
        )
        """Export data."""
        # TODO: not logical to transfer from ndarray to dataframe and back
        df_hhids = pd.DataFrame()
        df_hhids[cs.Z_CODE] = nd_regions
        df_hhids["hh_size"] = nd_sizes
        df_hhids["hh_landscape"] = nd_struct

        # TODO: rewrite method for ndarray instead of dataframe
        df_hhids = self.delete_too_many_household_ids(df_hhids, agents_data, 30)
        df_hhids = self.delete_too_many_household_ids(df_hhids, agents_data, 60)

        # add region to all
        z_codes = df_hhids[cs.Z_CODE].to_numpy()
        hh_sizes = df_hhids["hh_size"].to_numpy()
        landscapes = df_hhids["hh_landscape"].to_numpy()

        return np.vstack((z_codes, hh_sizes, landscapes))

    def initialize_household_contacts(self, region, agents_data, hh_data):
        """Distribute people from people class (ids) into households according
        to age /sex sensitive algorithms.

        Copy of array "agents_data" is passed through all methods.
        People that are distributed are deleted from the array.

        :param region: code for region
        :param agents_data (ndarray): information on agents (packed)
        :param hh_data (ndarry): infromation on households (packed)
        :return: contacts (ndarray): hold members_institutions (peoples
            id) for each hh
        """
        # TODO:(jan) hier könnte es um 50% reduziert werden wenn man "for landscape in [30,60]: call-all-methods(..., landscape) machen würde"
        agents_data, hh_data = self.initialize_1_size_hh(
            region, agents_data, hh_data, 30
        )
        agents_data, hh_data = self.initialize_1_size_hh(
            region, agents_data, hh_data, 60
        )
        # TODO:(Jan): spaces~rooms? But what means contacts?
        # 25 potential spaces for each household
        contacts = np.zeros((len(hh_data[0]), 25), dtype=int)

        # print('3-size hh')
        agents_data = self.initialize_3_size_hh(
            region, agents_data, hh_data, contacts, 30
        )
        agents_data = self.initialize_3_size_hh(
            region, agents_data, hh_data, contacts, 60
        )
        # print('4-size hh')
        agents_data = self.initialize_4_size_hh(
            region, agents_data, hh_data, contacts, 30
        )
        agents_data = self.initialize_4_size_hh(
            region, agents_data, hh_data, contacts, 60
        )

        # print('5-size hh')
        agents_data = self.initialize_5_size_hh(
            region, agents_data, hh_data, contacts, 30
        )
        agents_data = self.initialize_5_size_hh(
            region, agents_data, hh_data, contacts, 60
        )

        # TODO:(jan) Is there are reasen, that 2 after 1,3,4,5? Dann müsste das dokumenterit werden; sonst Reihenfolge anpassen
        # print('2-size hh')
        agents_data = self.initialize_2_size_hh(
            region, agents_data, hh_data, contacts, 30
        )
        agents_data = self.initialize_2_size_hh(
            region, agents_data, hh_data, contacts, 60
        )

        agents_data = self.initialize_bigger_5_size_hh(
            region, agents_data, hh_data, contacts, 30
        )
        _ = self.initialize_bigger_5_size_hh(region, agents_data, hh_data, contacts, 60)

        return hh_data, contacts

    def initialize_1_size_hh(self, region, agents_data, hh_data, landsc):
        """Select members_institutions for 1 size households according to age
        distribution table male /female."""
        # unpack multi-dim array to 1-dim
        _, hh_sizes, hh_landsc = self.extract_hh_data(hh_data)
        # get sum of 1-size hh split into city (c) and rural (r)
        sum_hh = np.sum((hh_sizes == 1) & (hh_landsc == landsc))
        if sum_hh == 0:
            return (
                agents_data,
                hh_data,
            )
        # TODO: upload household tables in bigauge_tool
        hh_data = hh_data[:, sum_hh:]
        df_age_gr = FileLoader.read_excel_table(
            PM.get_path_household_age_table(), "1_Person"
        )  # get age groups and distribution from data table
        age_grs = np.arange(len(df_age_gr))
        age_dist_f = np.array(df_age_gr["female"])
        age_dist_m = np.array(df_age_gr["male"])
        # only if number of households is bigger 0
        """
        # TODO: (jan) simplyfy; maybe somthing like this:
        # this one can maybe also easyer?
        if sum__hh==1:
            tup = ((age_dist_m, sum_hh, 0),)
        else:
            sum_hh_f = int(sum_hh * 0.5)
            sum__hh_m = sum_hh - sum__hh_f
            tup = ((age_dist_f, sum_hh_f, 1), ((age_dist_m, sum_hh_m, 0))

        ids = []
        for age_dist, sum_hh, sex in tup:
            loop_ids, agents_data = super(
                Households, self
            ).determine_age_groups_and_sample_member(
                age_grs,
                age_dist,
                region,
                landsc,
                sum_hh,
                df_age_gr,
                agents_data,
                sex,
            )
            ids.append(lopp_ids[cs.ID])
        if len(ids)>0:
            ids = np.concatenate(ids, axis=0)
            self.contacts1 = np.append(self.contacts1, ids)
        """
        if sum_hh == 1:
            ids, agents_data = super(
                Households, self
            ).determine_age_groups_and_sample_member(
                age_grs,
                age_dist_m,
                region,
                landsc,
                sum_hh,
                df_age_gr,
                agents_data,
                0,
            )
            # contacts[(hh_sizes == 1) & (hh_landsc == landsc), 0] = ids
        else:
            sum_hh_f = int(sum_hh * 0.5)
            data_hh_f, agents_data = super(
                Households, self
            ).determine_age_groups_and_sample_member(
                age_grs,
                age_dist_f,
                region,
                landsc,
                sum_hh_f,
                df_age_gr,
                agents_data,
                1,
            )
            sum_hh_m = sum_hh - sum_hh_f
            data_hh_m, agents_data = super(
                Households, self
            ).determine_age_groups_and_sample_member(
                age_grs,
                age_dist_m,
                region,
                landsc,
                sum_hh_m,
                df_age_gr,
                agents_data,
                0,
            )
            # Do not register single households
            ids = np.concatenate((data_hh_f[cs.ID], data_hh_m[cs.ID]), axis=0)
            self.contacts1 = np.append(self.contacts1, ids)

        return agents_data, hh_data

    def initialize_2_size_hh(self, region, agents_data, hh_data, contacts, landsc):
        """2 size households, almost evenly distributed among ages and sexes.

        # TODO:(jan) docstring wrong? The constellation of couples and
        families regarding age and sex follows the rule not the
        exception. In reality there are couples with huge age difference
        or homosexual couples. These constellations are not considered
        in the model due to simplification.
        """
        hhids, hh_sizes, hh_landsc = self.extract_hh_data(hh_data)
        df_age_gr = FileLoader.read_excel_table(
            PM.get_path_household_age_table(), "Mehrpersonen"
        )

        # sum of 2-size households
        sum_hh = np.sum((hh_sizes == 2) & (hh_landsc == landsc))
        if sum_hh == 0:
            return agents_data
        # sum of households with childless couples 83 %, single parent with child 16,5% and other 0,5%
        # TODO:(jan) Prozentzahlen irgendwo in variablen sammeln; Ist Transparenter und man will nicht den code ändern, wenn man eine andere Gesselschaft betrachtet
        sum_couple = np.rint(sum_hh * 0.83).astype(int)
        sum_single_p = np.rint(sum_hh * 0.165).astype(int)
        sum_o = sum_hh - sum_couple - sum_single_p

        if sum_single_p > 0:
            ids_single_p, agents_data = self.sample_family(
                1, 1, region, landsc, sum_single_p, df_age_gr, agents_data
            )
            ids = ids_single_p

        # sample members_institutions of households with childless couples
        if sum_couple > 0:
            ids_couple, data_mbr_1, agents_data = self.sample_couple(
                18, region, landsc, sum_couple, df_age_gr, agents_data
            )
            ids = np.concatenate((ids_single_p, ids_couple), axis=1)

        if sum_o > 0:
            ids_o, agents_data = self.sample_shared_flat(
                2, region, landsc, sum_o, df_age_gr, agents_data
            )
            ids = np.concatenate((ids, ids_o), axis=1)

        # add to household ids
        # TODO:(jan) see initialize_3_size_hh
        contacts[(hh_sizes == 2) & (hh_landsc == landsc), 0] = ids[0]
        contacts[(hh_sizes == 2) & (hh_landsc == landsc), 1] = ids[1]

        return agents_data

    def initialize_3_size_hh(self, region, agents_data, hh_data, contacts, landsc):
        """3 size households.

        The constellation of couples and families regarding age and sex
        follows the rule not the exception. In reality there are couples
        with huge age difference or homosexual couples. These
        constellations are not considered in the model due to
        simplification.
        """
        # sum of 3-size households
        _, hh_sizes, hh_landsc = self.extract_hh_data(hh_data)
        sum_hh = np.sum((hh_sizes == 3) & (hh_landsc == landsc))
        if sum_hh == 0:
            return agents_data

        df_age_gr = FileLoader.read_excel_table(
            PM.get_path_household_age_table(), "Mehrpersonen"
        )

        # sum of households two parents with child 88 %, single parent with two children 9 % and other 3 %
        sum_two_p = np.rint(sum_hh * 0.88).astype(int)
        sum_single_p = np.rint(sum_hh * 0.09).astype(int)
        sum_o = sum_hh - sum_two_p - sum_single_p

        # sample member of households
        if sum_two_p > 0:
            ids_two_p, agents_data = self.sample_family(
                2, 1, region, landsc, sum_two_p, df_age_gr, agents_data
            )
            ids = ids_two_p
        # tODO:(jan) wird sichergestellt, dass eltern erwachsen sind? Bei Paaren wird explizit 18 angegeben
        if sum_single_p > 0:
            ids_single_p, agents_data = self.sample_family(
                1, 2, region, landsc, sum_single_p, df_age_gr, agents_data
            )
            ids = np.concatenate((ids_two_p, ids_single_p), axis=1)
        if sum_o > 0:
            ids_o, agents_data = self.sample_shared_flat(
                3, region, landsc, sum_o, df_age_gr, agents_data
            )
            ids = np.concatenate((ids, ids_o), axis=1)

        # add to household contacts
        for i in range(3):
            contacts[(hh_sizes == 3) & (hh_landsc == landsc), i] = ids[i]

        return agents_data

    def initialize_4_size_hh(self, region, agents_data, hh_data, contacts, landsc):
        """4 size households.

        The constellation of couples and families regarding age and sex
        follows the rule not the exception. In reality there are couples
        with huge age difference or homosexual couples. These
        constellations are not considered in the model due to
        simplification.
        """
        # sum of 4-size households
        hhids, hh_sizes, hh_landsc = self.extract_hh_data(hh_data)
        sum_hh = np.sum((hh_sizes == 4) & (hh_landsc == landsc))
        if sum_hh == 0:
            return agents_data

        df_age_gr = FileLoader.read_excel_table(
            PM.get_path_household_age_table(), "Mehrpersonen"
        )
        # sum of households two parents with children 86 %, single parent with children 9 % and other 5 %
        sum_two_p = np.rint(sum_hh * 0.86).astype(int)
        sum_single_p = np.rint(sum_hh * 0.09).astype(int)
        sum_o = sum_hh - sum_two_p - sum_single_p
        # sample member of households
        if sum_two_p > 0:
            ids_two_p, agents_data = self.sample_family(
                2, 2, region, landsc, sum_two_p, df_age_gr, agents_data
            )
            ids = ids_two_p

        if sum_single_p > 0:
            ids_single_p, agents_data = self.sample_family(
                1, 3, region, landsc, sum_single_p, df_age_gr, agents_data
            )
            ids = np.concatenate((ids_two_p, ids_single_p), axis=1)

        if sum_o > 0:
            ids_o, agents_data = self.sample_shared_flat(
                4, region, landsc, sum_o, df_age_gr, agents_data
            )
            ids = np.concatenate((ids, ids_o), axis=1)

        # add to household contacts
        for i in range(4):
            contacts[(hh_sizes == 4) & (hh_landsc == landsc), i] = ids[i]

        return agents_data

    def initialize_5_size_hh(self, region, agents_data, hh_data, contacts, landsc):
        """5 size households.

        5 size ˜ 68 % and > 5 size ˜ 32 % of 68 % are 77% families and
        of 32 % 47 % families --> (0.68 * 0.77 + 0.32 * 0.47)/ 1.0 ˜
        67,4 % total families

        The constellation of couples and families regarding age and sex
        follows the rule not the exception. In reality there are couples
        with huge age difference or homosexual couples. These
        constellations are not considered in the model due to
        simplification.
        """
        _, hh_sizes, hh_landsc = self.extract_hh_data(hh_data)
        sum_hh = np.sum((hh_sizes == 5) & (hh_landsc == landsc))
        if sum_hh == 0:
            return agents_data

        df_age_gr = FileLoader.read_excel_table(
            PM.get_path_household_age_table(), "Mehrpersonen"
        )

        # sum of households two parents with children 67 % and other 23 %
        sum_two_p = np.rint(sum_hh * 0.67).astype(int)
        sum_o = sum_hh - sum_two_p
        # sample member of households
        if sum_two_p > 0:
            ids_two_p, agents_data = self.sample_family(
                2, 3, region, landsc, sum_two_p, df_age_gr, agents_data
            )
            ids = ids_two_p
        if sum_o > 0:
            ids_o, agents_data = self.sample_shared_flat(
                5, region, landsc, sum_o, df_age_gr, agents_data
            )
            ids = np.concatenate((ids, ids_o), axis=1)

        # add to household contacts
        for i in range(5):
            contacts[(hh_sizes == 5) & (hh_landsc == landsc), i] = ids[i]

        return agents_data

    def initialize_bigger_5_size_hh(
        self, region, agents_data, hh_data, contacts, landsc
    ):
        # TODO:(Jan) ich steig nicht ganz durch. Werden hier jetzt neue Haushalte erzeugt, die größer als 5 sind? Erdne zu besteheneden 5Personen Haushalten neue Personen hinzugefügt? Werden große Haushalte auf 5reduziert?
        # extract number of leftovers and number of households >= 5
        ids, ages, sexes, z_codes, landscapes, _ = super(
            Households, self
        ).extract_agents_data(agents_data)
        hhids, hh_sizes, hh_landsc = self.extract_hh_data(hh_data)
        left_overs = ids[(landscapes == landsc)]
        if len(left_overs) == 0:
            return agents_data
        hh_size_5 = contacts[(hh_sizes == 5) & (hh_landsc == landsc)]

        # 32 % of households with size 5 are actually bigger than 5
        sum_bigger_5 = math.ceil(len(hh_size_5) * 0.32)
        sum_5 = len(hh_size_5) - sum_bigger_5

        # now distribute the leftovers into the households > 5
        i = 5
        while len(left_overs) > 0:
            if len(left_overs) > sum_bigger_5:
                ids_lo = left_overs[:sum_bigger_5]
                add_zeros = np.zeros(sum_5, dtype=int)
                ids_to_add = np.concatenate((add_zeros, ids_lo), axis=0)
                assert len(ids_to_add) == len(hh_size_5)
                left_overs = left_overs[sum_bigger_5:]
            else:
                ids_lo = left_overs
                diff = len(hh_size_5) - len(left_overs)
                add_zeros = np.zeros(diff, dtype=int)
                ids_to_add = np.concatenate((add_zeros, ids_lo), axis=0)
                assert len(ids_to_add) == len(hh_size_5)
                left_overs = np.arange(0)
            contacts[(hh_sizes == 5) & (hh_landsc == landsc), i] = ids_to_add
            agents_data, data_mbr = super(
                Households, self
            ).split_sampled_data_from_agents_data(
                agents_data[cs.ID], ids_lo, agents_data
            )
            i += 1
        return agents_data

    def transfer_data_to_sub_nets(self):
        # ["id", "sub id", "type", "size", "mean contacts", "year", "weekday"]
        # TODO: (Jan) wäre leichter nachzuvollziehen, ob die Reihenfolge richtig ist, wenn axis0->id , axis7->weekday etc.
        axis0 = self.ids
        # TODO:(jan) axis0=axis1? warum duplicated
        axis1 = self.ids
        axis2 = np.full(self.ids.shape, fill_value=np.nan, dtype=float)
        axis3 = self.sizes
        # TODO:(jan)?
        axis4 = np.where(
            (self.sizes - 1) > 5, (self.sizes - 1) * 0.8, (self.sizes - 1)
        ).astype(int)
        axis5 = np.full(self.ids.shape, fill_value=np.nan, dtype=float)
        # TODO:(jan) axis6 = axis5.copy() ?
        axis6 = np.full(self.ids.shape, fill_value=np.nan, dtype=float)
        # TODO:(Jan) fehlt im Docstring
        axis7 = self.landsc
        self.sub_nets = np.stack(
            (axis0, axis1, axis2, axis3, axis4, axis5, axis6, axis7), axis=1
        )

    def delete_too_many_household_ids(self, df_region, agents_data, landsc):
        """Delete households from xls only if more required household
        members_institutions than available people.

        :param df_region: xls on household data
        :param agents_data: information on agents
        :param landsc: 30 city, 60 landscape
        :return: df_region with deletions
        """
        ids, ages, sexes, z_codes, landscapes, sociable = super(
            Households, self
        ).extract_agents_data(agents_data)
        try:
            region = df_region.loc[df_region.index[0], cs.Z_CODE]
        except BaseException:
            return df_region
        # TODO:(jan) maybe else
        sum_hh_mbr = df_region.loc[df_region["hh_landscape"] == landsc, "hh_size"].sum()
        sum_people = ((z_codes == region) & (landscapes == landsc)).sum()
        # TODO: (jan) verwirrend, dass left_overs negativ und dann ansteigt; besser levt_overs = sum_hh_mbr - sum_people;
        # TODO:(Jan) ggf. left_over -> somthing like empty_housholds?
        # TODO:(Jan) scheint richtig zu sein, aber nicht ganz intuitiv, was hier passiert; Wäre aber vermutlich bereits deutlich leichter, wen left_overs nicht richtung 0 hochzählen würde
        left_overs = sum_people - sum_hh_mbr
        i = 1
        # if more household members_institutions than people, delete households one by one
        while left_overs < 0:
            # delete 1-size, 2-size, ... 6-size, 1-size,....6-size etc.
            size = i % 6
            filt = (df_region["hh_landscape"] == landsc) & (
                df_region["hh_size"] == size
            )
            df_region_sub = df_region.loc[
                filt,
            ]
            if df_region_sub.size > 0:
                df_region.drop(
                    index=df_region_sub.iloc[
                        -1,
                    ].name,
                    inplace=True,
                )

            i += 1
            left_overs += size
        # TODO:(jan) mabe finally?
        return df_region

    def sample_couple(
        self,
        age_min,
        region,
        landsc,
        sum_hh,
        df_age_gr,
        agents_data,
        age_max=99,
    ):
        # TODO:(Jan) ist gefährlich, dass age_min nicht direkt vor age_max; ggf. einfach "agents_data, *, age_max=99"
        """Sample ids from the pool for two household members_institutions
        within specific age groups.

        First sample the female according to specified ages. Then count age groups of female members_institutions and determine
        possible age groups for the other member. Then sample second member.

        Args:
            region(string): key for region
            age_min (int): min age of first member
            age_max (int): max age of first member
            landsc (int): city (30) or rural area (60)
            sum_hh (int): total number of households to sample the couple for
            df_age_gr (DataFrame): Hold translation age to age group
            agents_data (dict(ndarray)): data of agents in the pool to choose from (ids, ages, sexes, z_codes, landsc)

        Returns:
            ids_couple (ndarray(sum,2)): Hold ids for couples
            data_mbr_1: (dict(ndarray)): data of sampled member (ids, ages, sexes, z_codes, landsc)
            agents_data (dict(ndarray)): data of agents in the pool after deletion of sampled agents (ids, ages, sexes, z_codes, landsc)
        """
        # create array to save ids of couples
        ids_couple = np.zeros((2, sum_hh), dtype="U20")
        # TODO:(Jan) Age-groups sind Grupen von 5 Jahren? oder gehts hier doch nur bis 17 Jahre?
        # TODO:(Jan) Übersichtlicher, wenn age_gr&age_dist eine gemeinsame Tabelle bzw. Dictonrary wären
        # TODO:(Jan) Auch diese Daten müssten egtl. einfach über eine csv datei eingelesen werden
        age_gr = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        age_dist = np.array(
            [
                0.12,
                0.12,
                0.10,
                0.06,
                0.06,
                0.06,
                0.06,
                0.06,
                0.07,
                0.08,
                0.08,
                0.05,
                0.04,
                0.04,
            ]
        )

        # sample the first household member (female) from pool
        data_mbr_1, agents_data = self.determine_age_groups_and_sample_member(
            age_gr,
            age_dist,
            region,
            landsc,
            sum_hh,
            df_age_gr,
            agents_data,
            1,
            member="couple",
        )

        # add first members_institutions to array
        ids_couple[0] = data_mbr_1[cs.ID]

        # sort first members_institutions by age group
        (
            ages_mbr_1_count,
            ages_mbr_1_indices,
        ) = self.sort_and_count_ages_by_age_group(data_mbr_1[cs.AGE], df_age_gr)

        # sample partner for each age group
        for a in df_age_gr["age group"]:
            sum = ages_mbr_1_count[a]
            if sum > 0:
                data_mbr_2, agents_data = self.sample_mate(
                    a,
                    region,
                    landsc,
                    sum,
                    df_age_gr,
                    agents_data,
                    member="mate",
                )
                ids_couple[1, ages_mbr_1_indices[a]] = data_mbr_2[cs.ID]

        return ids_couple, data_mbr_1, agents_data

    def sample_family(
        self, nmb_par, nmb_child, region, landsc, sum_hh, df_age_gr, agents_data
    ):
        """Sample ids from the pool for two household members_institutions
        within specific age groups.

        First sample the female according to specified ages. Then count age groups of female members_institutions and determine
        possible age groups for the other member. Then sample second and other member.

        Args:
            # TODO:(jan) not uptodate
            region(string): key for region
            age_min (int): min age of first parent (head of family)
            age_max (int): max age of first parent (head of family)
            landsc (int): city (30) or rural area (60)
            sum_hh (int): total number of households to sample the family type for
            df_age_gr (DataFrame): Hold translation age to age group
            agents_data (dict(ndarray)): data of agents in the pool to choose from (ids, ages, sexes, z_codes, landsc)

        Returns:
            ids_family (ndarray(sum_hh,x)): Hold ids for family members_institutions
            agents_data (dict(ndarray)): data of agents in the pool after deletion of sampled agents (ids, ages, sexes, z_codes, landsc)
        """
        # TODO: (Jan) <0 -> <=0 OR if nmb_par not in [1,2]
        if (nmb_par > 2) or (nmb_par < 0):
            raise AssertionError("Number of parents must be one or two")
        if nmb_child < 1:
            raise AssertionError("Number of children one at least")

        # create array to save family ids
        size = nmb_par + nmb_child
        ids_family = np.zeros((size, sum_hh), dtype="U20")
        # sample first child from pool
        data_child_1, agents_data = self.sample_child(
            region, landsc, sum_hh, df_age_gr, agents_data
        )
        ids_family[0] = data_child_1[cs.ID]

        # sort first child by age group
        (
            ages_child_1_count,
            ages_child_1_indices,
        ) = self.sort_and_count_ages_by_age_group(data_child_1[cs.AGE], df_age_gr)
        # iterate age groups
        # for each a_gr there is a limited variety of possible a_gr for parents / sisters and brothers
        for a_gr in df_age_gr["age group"]:
            sum_a_gr = ages_child_1_count[a_gr]
            if sum_a_gr > 0:
                # sample sisters and brothers
                nmb_sis = nmb_child - 1
                if nmb_child > 1:
                    data_sisters, agents_data = self.sample_mate(
                        a_gr,
                        region,
                        landsc,
                        sum_a_gr * nmb_sis,
                        df_age_gr,
                        agents_data,
                        a_gr_min=0,
                        a_gr_max=5,
                        member="child",
                    )
                    data_sisters[cs.ID] = data_sisters[cs.ID].reshape(
                        nmb_sis, int(len(data_sisters[cs.ID]) / nmb_sis)
                    )
                    ids_family[
                        1 : 1 + nmb_sis, ages_child_1_indices[a_gr]
                    ] = data_sisters[cs.ID]
                # sample parents
                data_parent_1, data_parent_2, agents_data = self.sample_parents(
                    a_gr,
                    region,
                    landsc,
                    nmb_par,
                    sum_a_gr,
                    df_age_gr,
                    agents_data,
                )
                ids_family[1 + nmb_sis, ages_child_1_indices[a_gr]] = data_parent_1[
                    cs.ID
                ]
                if data_parent_2 is not None:
                    ids_family[2 + nmb_sis, ages_child_1_indices[a_gr]] = data_parent_2[
                        cs.ID
                    ]

        return ids_family, agents_data

    def sample_parents(
        self, a_gr, region, landsc, nmb_par, sum_a_gr, df_age_gr, agents_data
    ):
        if nmb_par == 2:
            """# TODO: (Jan) simplyrfy for sex in (0,1): _, _ =
            self.sample_mate(.., sex=sex, ...)

            ODER vgl. TODO inside initialize_1_size_hh()
            """
            data_parent_1, agents_data = self.sample_mate(
                a_gr,
                region,
                landsc,
                sum_a_gr,
                df_age_gr,
                agents_data,
                sex=1,
                member="parent",
            )
            data_parent_2, agents_data = self.sample_mate(
                a_gr,
                region,
                landsc,
                sum_a_gr,
                df_age_gr,
                agents_data,
                sex=0,
                member="parent",
            )

        else:
            if sum_a_gr == 1:  # special case
                data_parent_1, agents_data = self.sample_mate(
                    a_gr,
                    region,
                    landsc,
                    sum_a_gr,
                    df_age_gr,
                    agents_data,
                    sex=1,
                    member="parent",
                )  # 1 for female
            else:  # if single parent, distinguish male (16 %) and female (84 %) single parent
                sum_a_gr_f = int(sum_a_gr * 0.84)
                sum_a_gr_m = sum_a_gr - sum_a_gr_f
                data_parent_1_f, agents_data = self.sample_mate(
                    a_gr,
                    region,
                    landsc,
                    sum_a_gr_f,
                    df_age_gr,
                    agents_data,
                    sex=1,
                    member="parent",
                )
                data_parent_1_m, agents_data = self.sample_mate(
                    a_gr,
                    region,
                    landsc,
                    sum_a_gr_m,
                    df_age_gr,
                    agents_data,
                    sex=0,
                    member="parent",
                )
                data_parent_1 = self.concatenate_data_dictionary(
                    data_parent_1_f, data_parent_1_m
                )
            data_parent_2 = None

        return data_parent_1, data_parent_2, agents_data

    def sample_shared_flat(self, size, region, landsc, sum_hh, df_age_gr, agents_data):
        # create array to save ids
        ids_other = np.zeros((size, sum_hh), dtype="U20")

        # TODO:(Jan) see sample_couple(); hier letztes mal angemerkt
        age_gr = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        age_dist = np.array(
            [
                0.25,
                0.29,
                0.25,
                0.15,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.00,
                0.00,
            ]
        )
        # sample first member from pool
        data_mbr_1, agents_data = self.determine_age_groups_and_sample_member(
            age_gr,
            age_dist,
            region,
            landsc,
            sum_hh,
            df_age_gr,
            agents_data,
            1,
            member="flatmate",
        )

        # sample first member from pool
        # data_mbr_1, agents_data = self.sample_member_from_pool(18, region, landsc, sum_hh, agents_data,
        # age_max=80)
        ids_other[0] = data_mbr_1[cs.ID]

        # sort first members_institutions by age group
        (
            ages_mbr_1_count,
            ages_mbr_1_indices,
        ) = self.sort_and_count_ages_by_age_group(data_mbr_1[cs.AGE], df_age_gr)

        # sample flatmates for each age group
        for a in df_age_gr["age group"]:
            sum = ages_mbr_1_count[a]
            if sum > 0:
                data_other, agents_data = self.sample_mate(
                    a,
                    region,
                    landsc,
                    sum * (size - 1),
                    df_age_gr,
                    agents_data,
                    a_gr_min=4,
                    member="flatmate",
                )
                data_other[cs.ID] = data_other[cs.ID].reshape(
                    size - 1, int(len(data_other[cs.ID]) / (size - 1))
                )
                ids_other[1:size, ages_mbr_1_indices[a]] = data_other[cs.ID]

        return ids_other, agents_data

    def sample_mate(
        self,
        a_gr,
        region,
        landsc,
        sum_hh,
        df_age_gr,
        agents_data,
        sex=None,
        a_gr_min=4,
        a_gr_max=19,
        member=None,
    ):
        """Sample mates for a specific number and according to age gr of the
        first person.

        The age_distr to choose the mate from is relative to the
        mean_contacts_p_age of the first person. The age_distr is
        phantasized and not based on real data.

        :param a_gr: age gr of first member
        :param region: code
        :param landsc: 30 (city), 60 (landscape)
        :param sum_hh: total number to sample a mate for :param
            df_age_gr (DataFrame): Hold translation age to age group
        :param agents_data (dict(ndarray)): data of agents in the pool
            to choose from (ids, ages, sexes, z_codes, landsc)
        :param sex: 1 female, 0 male
        :param a_gr_min: not under mean_contacts_p_age 4 (>18)
        :param a_gr_max: not bigger mean_contacts_p_age 19 (<99)
        :param member:
        :return: sampled_data (ndarray): hold people ids for mates
        :return: agents_data (dict(ndarray)): data of agents in the pool
            after deletion of sampled agents (ids, ages, sexes, z_codes,
            landsc)
        """
        # set range of age group for the second member, a_gr is always the mean
        if member == "parent":
            # TODO:(Jan) liegt der mean in Deutschland nicht ehr bei 35 als bei 30?
            # set mean age group parent 30 years (=6 age groups) older than age group child, range +- 10 years(= +-2 age groups)
            orig_age_dist = np.array([0.125, 0.75, 0.125])
            a_gr += 6
            low_rg = up_rg = 1
        if member == "child":
            # set mean age group parent 30 years (=6 age groups) older than age group child, range +- 10 years(= +-2 age groups)
            orig_age_dist = np.array([0.10, 0.80, 0.10])
            low_rg = up_rg = 1

        else:
            # TODO:(Jan) ? das bezieht sich wieder auf parent? Ich glaub es sollte "elif member=='child' .. sein
            orig_age_dist = np.array([0.05, 0.25, 0.40, 0.25, 0.05])
            low_rg = up_rg = 2
        # select range of mean_contacts_p_age and distr according to mean_contacts_p_age of first member
        # if necessary, cut the age groups and age dist
        age_dist, age_gr = self.set_possible_age_groups_for_other_members(
            a_gr, a_gr_min, a_gr_max, low_rg, up_rg, orig_age_dist, df_age_gr
        )
        # now sample partner from the pool of people considering sampled age groups
        sampled_data, agents_data = self.determine_age_groups_and_sample_member(
            age_gr,
            age_dist,
            region,
            landsc,
            sum_hh,
            df_age_gr,
            agents_data,
            sex,
            member,
        )
        return sampled_data, agents_data

    def sample_child(self, region, landsc, sum_hh, df_age_gr, agents_data):
        # age gr 0: 0-4 and age gr 5 = 25-29
        age_gr = np.array([0, 1, 2, 3, 4, 5])
        age_dist = np.array([0.20, 0.20, 0.20, 0.20, 0.15, 0.05])

        # now sample children from the pool of people considering sampled age groups
        (data_children, agents_data,) = self.determine_age_groups_and_sample_member(
            age_gr,
            age_dist,
            region,
            landsc,
            sum_hh,
            df_age_gr,
            agents_data,
            member="child",
        )
        return data_children, agents_data

    def extract_hh_data(self, hh_data):
        z_codes = hh_data[0]
        hh_sizes = hh_data[1]
        landscapes = hh_data[2]

        return z_codes, hh_sizes, landscapes

    # def save_networks_to_file(self):
    #     self.save_current_mean_contacts()
    #     # transform data
    #     t1 = time.time()
    #     path = f"{self.abs_path}/inputs"
    #     # save members to npy file
    #     net_type = type(self).__name__.lower()
    #     members = self.members_groups
    #     np.save(f"{path}/{net_type}_networks_{self.scale}_subnet.npy", members)
    #
    #     # save info to dataframe
    #     hh_data = self.stack_arrays()
    #     df_data = pd.DataFrame(hh_data)  # to .csv
    #     df_data.columns = ["sub id", "z_code", "size", "landscape"]
    #     df_data.set_index("sub id", inplace=True)
    #     df_data.to_csv(f"{path}/households_{self.scale}_subnet.csv")

    def stack_arrays(self):
        self.ids = self.ids.reshape(len(self.ids), 1)
        self.z_codes = self.z_codes.reshape(len(self.ids), 1)
        self.sizes = self.sizes.reshape(len(self.ids), 1)
        self.landsc = self.landsc.reshape(len(self.ids), 1)
        hh_data = np.hstack(
            (
                self.ids,
                self.z_codes,
                self.sizes,
                self.landsc,
            )
        )
        return hh_data
