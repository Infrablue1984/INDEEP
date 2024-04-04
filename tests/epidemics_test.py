#!/venv/bin/python3
# -*- coding: utf-8 -*-

__author__ = "Sarah Albrecht"
__created__ = "2020"
__date_modified__ = "2023/03/03"
__version__ = "1.0"

import unittest
from datetime import date

import numpy as np
import pandas as pd

from calculation import epidemics
from calculation.network_manager import NetworkManager
from calculation.people import People


class TestEpidemicSpreader(unittest.TestCase):
    def setUp(self):
        regions = [5162]
        start_date = date(2020, 9, 30)
        scale = 100
        self.seed = np.random.default_rng(2)

        self.epidemic_data = {}

        self.people = People(regions, scale)
        self.net_manager = NetworkManager(
            self.people, np.asarray(regions), scale, start_date, self.seed
        )

        self.population_size = self.people.get_pop_size()

        self.epi_user_input = pd.read_csv(
            "../data/inputs/scenario/COVID_default.csv",
            usecols=["parameter", "value", "type"],
            dtype={"parameter": str, "value": float, "type": str},
            index_col="parameter",
        )

        self.epi_spreader = epidemics.AirborneVirusSpreader(
            self.people, self.net_manager, self.epi_user_input, self.seed
        )

        self.ref_epi_user_input = dict()
        with open("../data/inputs/scenario/COVID_default.csv") as fp:
            lines = fp.readlines()
        for item in lines:
            temp = item.split(",")
            if not temp[0] == "parameter":
                self.ref_epi_user_input[temp[0]] = float(temp[1])

    def tearDown(self):
        pass

    def test_init(self):
        pass

    def test_generate_duration_parameters(self):
        dur_param = self.epi_spreader._generate_duration_parameters(self.epi_user_input)
        self.assertEqual(len(dur_param), 11)
        for item in dur_param:
            if item == "susceptible_to_exposed":
                self.assertEqual(dur_param[item], {})
            else:
                self.assertEqual(
                    dur_param[item]["mean"],
                    self.ref_epi_user_input["{}_mean".format(item)],
                )
                self.assertEqual(
                    dur_param[item]["std"],
                    self.ref_epi_user_input["{}_std".format(item)],
                )

    def test_generate_epidemic_parameters(self):
        epi_data = self.epi_spreader._generate_epidemic_parameters(self.epi_user_input)
        for item in epi_data:
            self.assertEqual(epi_data[item], self.ref_epi_user_input[item])

    def test_set_status_fields(self):
        res = dict()
        res["susceptible"] = np.ones(self.population_size, dtype=bool)
        false_statuses = [
            "exposed",
            "asymptomatic",
            "presymptomatic",
            "mild",
            "severe",
            "critical",
            "recovered",
            "dead",
            "next_status_set",
            "infected",
            "infectious",
            "symptomatic",
        ]
        for status in false_statuses:
            res[status] = np.zeros(self.population_size, dtype=bool)
        self.epi_spreader._set_status_fields(self.epidemic_data, self.population_size)
        self.assertEqual(self.epidemic_data.keys(), res.keys())
        for key in self.epidemic_data:
            self.assertTrue(np.allclose(self.epidemic_data[key], res[key]))

    def test_generate_age_groups_of_agents(self):
        age_groups = self.epi_spreader._generate_age_groups_of_agents()
        agents_data = self.people.get_agents_data()
        ages = agents_data["age"]
        for i in range(10):
            self.assertEqual(age_groups[i], int(ages[i] / 10))

    def test_get_progression_probabilities_by_age_group(self):
        prog_proba = self.epi_spreader._get_progression_probabilities_by_age_group(
            self.epi_user_input
        )
        for key in prog_proba:
            temp = key.split("_")
            for i in range(10):
                self.assertEqual(
                    prog_proba[key][i],
                    self.ref_epi_user_input[
                        "{}_probability_from_".format(temp[0]) + str(i * 10)
                    ],
                )

    def test_set_initially_infected(self):
        self.epi_spreader._set_status_fields(self.epidemic_data, self.population_size)
        self.epi_spreader._set_initially_infected(
            self.epidemic_data, self.epi_user_input
        )
        ref_temp = round(
            self.ref_epi_user_input["reported_infected_percentage"]
            / 100
            * self.population_size
        )
        ref_inf = int(ref_temp * self.ref_epi_user_input["unreported_factor"])
        sus, inf = 0, 0
        for i in range(self.population_size):
            if not self.epidemic_data["susceptible"][i]:
                sus += 1
            if self.epidemic_data["infected"][i]:
                inf += 1
        self.assertTrue(sus >= ref_inf)
        self.assertEqual(inf, ref_inf)

    def test_set_initially_recovered(self):
        self.epi_spreader._set_status_fields(self.epidemic_data, self.population_size)
        self.epi_spreader._set_date_fields(self.epidemic_data, self.population_size)
        self.epi_spreader._set_initially_recovered(
            self.epidemic_data, self.epi_user_input
        )
        ref_rec = round(
            self.ref_epi_user_input["initially_recovered_percentage"]
            / 100
            * self.population_size
        )
        sus, rec = 0, 0
        for i in range(self.population_size):
            if not self.epidemic_data["susceptible"][i]:
                sus += 1
            if self.epidemic_data["recovered"][i]:
                rec += 1
        self.assertTrue(sus >= ref_rec)
        self.assertEqual(rec, ref_rec)

    def test_get_mean_transition_durations(self):
        self.epi_spreader._generate_duration_parameters(self.epi_user_input)
        means = dict()
        for key in self.ref_epi_user_input.keys():
            item = key.split("_")
            if len(item) >= 4:
                if item[1] == "to" and item[3] == "mean":
                    if item[0] in means:
                        means[item[0]].append(self.ref_epi_user_input[key])
                    else:
                        means[item[0]] = [self.ref_epi_user_input[key]]
        for item in means:
            mean_trans = self.epi_spreader._get_mean_transition_durations(item)
            self.assertEqual(mean_trans, means[item])

    def test_set_epidemic_data_for_initially_infected(self):
        self.epi_spreader._set_status_fields(self.epidemic_data, self.population_size)
        self.epi_spreader._set_date_fields(self.epidemic_data, self.population_size)
        init_stat_ids = {
            "exposed": np.array([1588, 1592, 1739, 2043]),
            "asymptomatic": np.array([1159, 1299, 1332, 1473]),
            "presymptomatic": np.array([549, 855, 937]),
            "mild": np.array([2634, 2667, 2781]),
            "severe": np.array([1525, 2850, 3500]),
            "critical": np.array([3210, 4439]),
        }
        self.epi_spreader._set_epidemic_data_for_initially_infected(
            self.epidemic_data, init_stat_ids
        )
        for status in init_stat_ids:
            res = np.nonzero(self.epidemic_data[status])
            self.assertEqual(len(init_stat_ids[status]), len(res[0]))

    def test_set_new_status(self):
        self.epi_spreader._set_status_fields(self.epidemic_data, self.population_size)
        self.epi_spreader._set_date_fields(self.epidemic_data, self.population_size)
        self.epi_spreader._set_probability_fields(
            self.epidemic_data, self.epi_user_input, self.population_size
        )
        indices = np.array([101, 287, 446, 2593, 3334])
        statuses = [
            "exposed",
            "asymptomatic",
            "presymptomatic",
            "mild",
            "severe",
            "critical",
            "recovered",
            "dead",
            "susceptible",
        ]
        index = [(0, 1), (0, 2), (3, 4), (3, 6), (4, 5), (4, 6), (5, 6), (5, 7), (8, 0)]
        for j in range(len(index)):
            self.epi_spreader._set_new_status(
                self.epidemic_data,
                statuses[index[j][0]],
                statuses[index[j][1]],
                indices,
            )
            self.assertTrue(self.epidemic_data["next_status_set"][indices].all())

    def test_modify_progression_probabilities(self):
        indices = np.array([528, 1034, 1203, 1342, 2323])
        probabilities = np.array([0, 0.2, 0.5, 0.7, 1])
        res = [0, 0.2, 0.5, 0.7, 1]
        for status in [
            "exposed",
            "asymptomatic",
            "presymptomatic",
            "mild",
            "severe",
            "critical",
        ]:
            prob_res = self.epi_spreader._modify_progression_probabilities(
                status, probabilities, indices
            )
            if status == "critical":
                res = [0, 0.4, 1, 1.4, 2]
            self.assertListEqual(prob_res.tolist(), res)

    def test_determine_hospital_factors(self):
        self.epi_spreader._initialize_epidemic_data(self.epi_user_input)
        indices = np.array([101, 287, 446, 528, 1034, 1203, 1342, 2323, 2593, 3334])
        for item in indices[:5]:
            self.people._public_data["hospitalized"][item] = True
        hosp = self.people.get_data_for("hospitalized", indices)
        hosp_fac = self.epi_spreader._determine_hospital_factors("critical", indices)
        for i in range(10):
            if hosp[i]:
                self.assertTrue(hosp_fac[i] == 1)
            else:
                self.assertTrue(
                    hosp_fac[i] == self.ref_epi_user_input["not_hospitalized_factor"]
                )

    def test_determine_vaccination_factors(self):
        self.epi_spreader._initialize_epidemic_data(self.epi_user_input)
        indices = np.array([101, 287, 446, 528, 1034, 1203, 1342, 2323, 2593, 3334])
        for item in indices[:5]:
            self.people._public_data["vaccinated"][item] = True
        vacc = self.people.get_data_for("vaccinated", indices)
        vacc_fac = self.epi_spreader._determine_vaccination_factors(indices)
        for i in range(10):
            if vacc[i]:
                self.assertTrue(
                    vacc_fac[i]
                    == self.ref_epi_user_input["vaccination_factor_progression"]
                )
            else:
                self.assertTrue(vacc_fac[i] == 1)

    def test_exit_old_status(self):
        self.epi_spreader._initialize_epidemic_data(self.epi_user_input)
        self.epi_spreader._set_status_fields(self.epidemic_data, self.population_size)
        indices = np.array([101, 287, 446, 528, 1034, 1203, 1342, 2323, 2593, 3334])
        self.epi_spreader._exit_old_status(self.epidemic_data, "exposed", indices)
        for i in indices:
            self.assertFalse(self.epidemic_data["exposed"][i])

    def test_enter_new_status(self):
        self.epi_spreader._initialize_epidemic_data(self.epi_user_input)
        self.epi_spreader._set_status_fields(self.epidemic_data, self.population_size)
        self.epi_spreader._set_date_fields(self.epidemic_data, self.population_size)
        indices = np.array([101, 287, 446, 528, 1034, 1203, 1342, 2323, 2593, 3334])
        self.epi_spreader._enter_new_status(self.epidemic_data, "mild", indices, 5)
        for i in indices:
            self.assertTrue(self.epidemic_data["mild"][i])
            self.assertTrue(self.epidemic_data["symptomatic"][i])
            self.assertEqual(self.epidemic_data["date_symptomatic"][i], 5)

    def test_get_possible_next_statuses(self):
        self.assertEqual(
            self.epi_spreader._get_possible_next_statuses("susceptible"), ["exposed"]
        )
        self.assertEqual(
            self.epi_spreader._get_possible_next_statuses("exposed"),
            ["presymptomatic", "asymptomatic"],
        )
        self.assertEqual(
            self.epi_spreader._get_possible_next_statuses("presymptomatic"), ["mild"]
        )
        self.assertEqual(
            self.epi_spreader._get_possible_next_statuses("asymptomatic"), ["recovered"]
        )
        self.assertEqual(
            self.epi_spreader._get_possible_next_statuses("mild"),
            ["severe", "recovered"],
        )
        self.assertEqual(
            self.epi_spreader._get_possible_next_statuses("severe"),
            ["critical", "recovered"],
        )
        self.assertEqual(
            self.epi_spreader._get_possible_next_statuses("critical"),
            ["dead", "recovered"],
        )
        self.assertEqual(self.epi_spreader._get_possible_next_statuses("recovered"), [])
        self.assertEqual(self.epi_spreader._get_possible_next_statuses("dead"), [])

    def test_set_infection_transmission_fields(self):
        self.epi_spreader._initialize_epidemic_data(self.epi_user_input)
        self.epi_spreader._set_infection_transmission_fields(
            self.epidemic_data, self.epi_user_input
        )
        self.assertEqual(len(self.epidemic_data["viral_load"]), self.population_size)
        self.assertEqual(
            len(self.epidemic_data["droplet_volume_concentration"]),
            self.population_size,
        )
        self.assertEqual(
            len(self.epidemic_data["relative_sensitivity"]), self.population_size
        )
        self.assertEqual(
            len(self.epidemic_data["viral_load_evolution_start"]), self.population_size
        )

    def test_evolve_viral_load(self):
        self.epi_spreader._initialize_epidemic_data(self.epi_user_input)
        self.epi_spreader._set_infection_transmission_fields(
            self.epidemic_data, self.epi_user_input
        )
        res = self.epi_spreader._evolve_viral_load(self.epidemic_data, 124, 16, 7)
        viral_load = self.epidemic_data["viral_load"][124]
        epi_param = self.epi_spreader._generate_epidemic_parameters(self.epi_user_input)
        vaccination_factor = epi_param["vaccination_factor_viral_load"]
        self.assertTrue(viral_load <= res or res <= 2 * vaccination_factor * viral_load)

    def test_evolve_relative_sensitivity(self):
        self.epi_spreader._initialize_epidemic_data(self.epi_user_input)
        epi_parameters = self.epi_spreader._generate_epidemic_parameters(
            self.epi_user_input
        )
        self.epi_spreader._set_infection_transmission_fields(
            self.epidemic_data, self.epi_user_input
        )
        ids = np.array([111, 528, 1034, 1342, 3334])
        for item in ids[:3]:
            self.people._public_data["vaccinated"][item] = True
        vacc_factor = epi_parameters["vaccination_factor_relative_sensitivity"]
        res = self.epi_spreader._evolve_relative_sensitivity(self.epidemic_data, ids, 5)
        for i in range(5):
            if i < 3:
                self.assertEqual(res[i], vacc_factor)
            else:
                self.assertEqual(res[i], 1.0)

    def test_calculate_quanta_emission_load(self):
        self.epi_spreader._initialize_epidemic_data(self.epi_user_input)
        epi_parameters = self.epi_spreader._generate_epidemic_parameters(
            self.epi_user_input
        )
        self.epi_spreader._set_infection_transmission_fields(
            self.epidemic_data, self.epi_user_input
        )
        agent_data = self.people.get_agents_data()
        for id in [0, 5, 100, 234]:
            res = self.epi_spreader._calculate_quanta_emission_load(
                self.epidemic_data, agent_data["id"][id], 5
            )
            quanta_conversion_factor = epi_parameters["quanta_conversion_factor"]
            viral_load = self.epidemic_data["viral_load"][id]
            droplet_volume_concentration = self.epidemic_data[
                "droplet_volume_concentration"
            ][id]
            load = viral_load * droplet_volume_concentration / quanta_conversion_factor
            self.assertEqual(res, load)


if __name__ == "__main__":
    unittest.main()
