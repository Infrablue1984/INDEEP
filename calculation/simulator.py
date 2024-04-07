#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""This module is responsible to organize and implement the simulation.

This module consists of two big classes, the SimManager and SimRunner. The SimManager
pre-checks the input data for correctness. It also generates key data, tracks the
results and initializes one or multiple SimRunner objects. The SimRunner initializes
other simulation objects that are responsible to do subtasks. It can run the
simulation one time for a given number of time steps. In each time step it calls the
objects to fulfill their tasks.

Typical usage example:
    sim = SimManager(
        start_date,
        end_date,
        intervention_list,
        epi_user_input,
        regions,
        scale,
    )
    sim.run(5)
    sim.plot_results()
"""

__author__ = "Inga Franzen"
__created__ = "2023"
__date_modified__ = "2023/05/31"
__version__ = "1.0"

import datetime as dt
import multiprocessing

# import profile
import time
from copy import deepcopy as deep
from multiprocessing import Condition, Lock, Manager, Process
from typing import List

import numpy as np
import pandas as pd

from calculation import (
    epidemics,
    network_manager,
    people,
    plots,
    public_regulator,
    tracker,
)
from calculation.interventions import Intervention
from calculation.interventions import InterventionMaker as IM
from data_factory import data_manager
from synchronizer import constants as cs


class SimManager:
    """Manage simulation run.

    Communicate with the GUI (here: StartUp.py) and organize simulation by calling
    itself or other classes. Pre-Check input data, produce key data and initialize
    SimRunner objects to run the simulation one or multiple times. Also have
    a method to perform the reset button, which will delete key data.

    Attributes:
        start_date: DateTime object
        end_date: DateTime object
        intervention_list: A list with objects of intervention type1 and type2
        epi_user_input: A Dataframe with scenario data.
        regions: A numpy array with integers 1000 -17000, standing for german region.
        scale: Number of people represented by a single agent.
        num_time_steps: An integer.
        plotter: A plotter class object.
    """

    sim_progress = {"text": "", "progress": 0}

    def __init__(
        self,
        start_date: dt.date,
        end_date: dt.date,
        intervention_list: List[Intervention],
        epi_user_input: pd.DataFrame,
        regions: List[int],
        scale: int = 100,
    ):
        """Set parameters and objects needed for the simulation.

        Args:
            start_date: Date to start the simulation from.
            end_date: Date at which simulation ends.
            intervention_list: Interventions to mitigate the disease spread.
            epi_user_input: User input to specify the disease scenario.
            regions: Codes of the selected regions.
            scale: Number of people represented by a single agent.

        Raises:
            AssertionError: An argument does not have the correct type.
        """
        if not isinstance(start_date, dt.date):
            raise AssertionError("Start date must be specified as a date.")
        if not isinstance(end_date, dt.date):
            raise AssertionError("End date must be specified as a date.")
        if not isinstance(regions, list):
            raise AssertionError("Regions must be specified as list.")
        if not isinstance(epi_user_input, pd.DataFrame):
            raise AssertionError("Epidemic data must be specified as DataFrame.")
        if not isinstance(intervention_list, list):
            raise AssertionError("Intervention data must be specified as list.")
        for intervention_object in intervention_list:
            if (type(intervention_object).__name__ != "InterventionType1") and (
                type(intervention_object).__name__ != "InterventionType2"
            ):
                raise AssertionError(
                    "Intervention list must contain objects of type Intervention."
                )
        self.start_date = start_date
        self.end_date = end_date
        self.intervention_list = intervention_list
        self.epi_user_input = epi_user_input
        self.regions = np.array(regions)
        self.scale = scale
        self.num_time_steps = (end_date - start_date).days
        self.plotter = plots.Plotter()

    def run(self, number_of_runs: int = 1):
        """Run the simulation one or multiple times.

        Pre-check input data for correctness and organize generation of
        key data. Run the simulation one or multiple times depending on the user
        input. After all simulation runs are done, export the results to a .csv file.

        Args:
            number_of_runs: Total number of simulation runs.
        """
        SimManager.pre_check_start_and_end_date(self.start_date, self.end_date)
        IM.pre_check_intervention_user_input(self.intervention_list)
        data_manager.DataManager.organize_generation_of_key_data_for_selected_regions(
            self.regions, self.scale, SimManager.sim_progress
        )
        if number_of_runs == 1:
            result_dict = self._organize_simple_run()
        else:
            result_dict = self._organize_multiple_runs(number_of_runs)
        tracker.TrackResults.export_results(result_dict)
        SimManager.sim_progress = None

    def _organize_simple_run(self) -> dict:
        """Run the simulation once.

        Initialize SimRunner object and run the simulation on that object.

        Returns:
            result_dict: Dict with run-count as key and pd.Dataframe with results as
            value.
        """
        t1 = time.time()
        result_dict = Manager().dict()
        my_seed = np.random.default_rng(10)
        simulation_runner = self._initialize_simulation_runner()
        simulation_runner.run_simulation(my_seed, result_dict, SimManager.sim_progress)
        t2 = time.time()
        print(
            f"Run simulation once with {self.num_time_steps} steps:"
            f" {round(t2 - t1, 2)} s"
        )
        return result_dict

    def _organize_multiple_runs(self, number_of_runs: int):
        """Run the simulation multiple times.

        Perform multiprocessing and therefore create lock object, condition object and
        other essentials. Loop over number_of_runs and create processes with own seed
        and SimRunner objects for each run. Then start and join processes to run
        all simulations in parallel.

        Args:
            number_of_runs: Total number of simulation runs.

        Returns:
            result_dict: Dict with run-count as key and pd.Dataframe with results as
            value.
        """
        t1 = time.time()
        lock = Lock()
        condition = Condition()
        manager = Manager()
        result_dict = manager.dict()
        SimManager.sim_progress = manager.dict()
        initialize_progress_bar(SimManager.sim_progress)
        processes = []
        for run_count in range(1, number_of_runs + 1):
            my_seed = np.random.default_rng(run_count)
            simulation_runner = self._initialize_simulation_runner()
            processes.append(
                Process(
                    target=simulation_runner.run_simulation,
                    args=(
                        my_seed,
                        result_dict,
                        SimManager.sim_progress,
                        run_count,
                        number_of_runs,
                        lock,
                        condition,
                    ),
                )
            )
        for process in processes:
            process.start()
        SimManager.update_progress_bar_while_running(processes)
        for process in processes:
            process.join()
        t2 = time.time()
        print(
            f"Run simulation {number_of_runs} times with {self.num_time_steps} steps:"
            f" {round(t2 - t1, 2)} s"
        )
        return result_dict

    def _initialize_simulation_runner(self):
        """Initialize a SimRunner object from instance variables.

         Take a deep copy of variables that are also used by other objects.

        Returns:
            A SimRunner object.
        """
        return SimRunner(
            self.start_date,
            self.end_date,
            deep(self.intervention_list),
            self.epi_user_input,
            self.regions,
            self.scale,
        )

    def plot_results(self):
        """Call ClassPlatter to plot the results of the whole simulation."""
        self.plotter.plot_results()

    @staticmethod
    def reset_progress(text: str = ""):
        SimManager.sim_progress = {"text": text, "progress": 0}

    @staticmethod
    def update_progress_bar_while_running(processes):
        """Update progress bar showing the progress of the simulation in
        percent.

        Print progress on the console.

        Args:
            processes (ClassProcess).
        """

        still_running = True
        while still_running:
            still_running = False
            for process in processes:
                if process.is_alive():
                    still_running = True
            print(
                f"{SimManager.sim_progress['text']}:"
                f" {round(SimManager.sim_progress['progress'], 0)}"
            )
            time.sleep(0.1)  # for i in range(100):
        return

    @staticmethod
    def pre_check_start_and_end_date(start_date: dt.date, end_date: dt.date):
        """Check if start and end date are specified correctly.

        Args:
            start_date: Starting date of the simulation.
            end_date: End date of the simulation.
        """
        if (end_date - start_date).days < 0:
            raise BaseException(
                cs.LANGUAGE_DICT[cs.START_END_DATE_EXCEPTION][cs.GERMAN],
            )

    @staticmethod
    def reset_data():
        """Reset all files with key data.

        This resembles a reset button in the tool GUI. The button can be
        helpful when simulation was interrupted during saving process.
        Generally error messages would occur and can be easily removed
        with this button. Key data will automatically be reproduced next
        run.
        """
        scales = [10, 100]
        for scale in scales:
            data_manager.DataManager.reset_data(scale)


class SimRunner:
    """Run simulation.

    Initialize other classes with specific tasks for the simulation.
    Run the simulation for a given number of time steps. Call classes
    with specific tasks in any time step. Ensure succession.

    Attributes:
        regions: A numpy array with integers 1000 -17000, standing for german region.
        intervention_list: A list with objects of intervention type1 and type2
        epi_user_input: A Dataframe with scenario data.
        scale: Number of people represented by a single agent.
        start_date: DateTime object
        end_date: DateTime object
        num_time_steps: An integer.
        people: A People class object.
        net_manager: A NetworkManager class object.
        epi_spreader: An EpidemicSpreader class object.
        pub_reg: A PublicHealthRegulator class object.
        tracker: A tracker class object.
    """

    # @profile
    def __init__(
        self,
        start_date,
        end_date,
        intervention_list,
        epi_user_input,
        regions,
        scale=100,
    ):
        """Initialize parameters and classes needed for the simulation.

        Args:
            start_date (datetime.date): Date to start simulation from.
            end_date (datetime.date): Date at which simulation ends.
            intervention_list (list of objects): Intervention objects to mitigate the disease.
            epi_user_input (Pandas Dataframe): Table with scenario data.
            regions (list with int): Hold key of region.
            scale (int): ratio between agents in the model and reality.

        Raises:
            AssertionError: If type of Args do not fit.
        """
        if not isinstance(start_date, dt.date):
            raise AssertionError("Start date must be specified as a date.")
        if not isinstance(end_date, dt.date):
            raise AssertionError("End date must be specified as a date.")
        if not isinstance(epi_user_input, pd.DataFrame):
            raise AssertionError("Epidemic data must be specified as DataFrame.")
        if not isinstance(intervention_list, list):
            raise AssertionError("Intervention data must be specified as list.")

        self.regions = regions
        self.intervention_list = intervention_list
        self.epi_user_input = epi_user_input
        self.scale = scale
        self.start_date = start_date
        self.end_date = end_date
        self.num_time_steps = (end_date - start_date).days
        self.people = None
        self.net_manager = None
        self.epi_spreader = None
        self.pub_reg = None
        self.tracker = None

    def _initialize_simulation(
        self,
        progress_bar: dict,
        number_of_runs: int,
        lock=None,
        condition=None,
        seed=None,
    ):
        initialize_progress_bar(progress_bar, "Initialize Simulation")
        self.intervention_list = self.intervention_list
        self.tracker = self._initialize_tracker()
        self.people = people.People(self.regions, self.scale)
        refresh_progress_bar(5 / number_of_runs, progress_bar, lock)
        self.net_manager = self._initialize_networks(seed)
        refresh_progress_bar(50 / number_of_runs, progress_bar, lock)
        self.epi_spreader = self._initialize_epidemic_spreader(seed)
        refresh_progress_bar(2 / number_of_runs, progress_bar, lock)
        self.pub_reg = self._initialize_public_regulator(seed)
        refresh_progress_bar(43 / number_of_runs, progress_bar, lock)
        if condition:
            SimRunner.wait_for_all_processes(progress_bar, condition)
        time.sleep(0.5)

    def _initialize_tracker(self):
        """Initialize the tracker object of the simulation.

        Specify the keys that should be tracked throughout the
        simulation and create a tracker object for those keys.

        Returns:
            The tracker object of the simulation created for the
            specified keys.
        """
        track_keys = [
            cs.SUSCEPTIBLE,
            cs.EXPOSED,
            cs.DIAGNOSED,
            cs.QUARANTINED,
            cs.INFECTED,
            cs.RECOVERED,
            cs.SEVERE,
            cs.CRITICAL,
            cs.DEAD,
        ]
        return tracker.TrackResults(
            self.start_date, self.num_time_steps, track_keys, self.scale
        )

    def _initialize_public_regulator(
        self,
        seed: np.random.Generator,
    ) -> public_regulator.PublicHealthRegulator:
        """Initialize the public regulator object of the simulation.

        Args:
            seed: Random number generator with specified seed.

        Returns:
            The public regulator object of the simulation.
        """
        public = public_regulator.PublicHealthRegulator(
            self.intervention_list,
            self.people,
            self.net_manager,
            self.start_date,
            seed,
        )
        epi_user_input = deep(self.epi_user_input)
        public.initialize_public_data(epi_user_input)
        return public

    def _initialize_networks(
        self,
        seed: np.random.Generator,
    ) -> network_manager.NetworkManager:
        """Initialize the networks of the simulation.

        Args:
            seed: Random number generator with specified seed.

        Returns:
            The network manager object of the simulation.
        """
        return network_manager.NetworkManager(
            self.people,
            self.regions,
            self.scale,
            self.start_date,
            seed,
        )

    def _initialize_epidemic_spreader(
        self,
        seed: np.random.Generator,
    ) -> epidemics.EpidemicSpreader:
        """Initialize the epidemic spreader object of the simulation.

        Args:
            seed: Random number generator with specified seed.

        Returns:
            The epidemic spreader object of the simulation.
        """
        # TODO(Felix): Disease name has to be handed over in some way when several
        #   diseases are integrated.
        disease = "covid19"
        epi_user_input = deep(self.epi_user_input)
        # TODO: add other spreader types
        if disease == "covid19":
            epi_spreader = epidemics.AirborneVirusSpreader(
                self.people,
                self.net_manager,
                epi_user_input,
                seed,
            )
        else:
            epi_spreader = None
        return epi_spreader

    # @profile
    def run_simulation(
        self,
        seed: np.random.Generator,
        result_dict,
        progress_bar,
        run_count: int = 1,
        number_of_runs: int = 1,
        lock: multiprocessing.Lock = None,
        condition: multiprocessing.Condition = None,
    ):
        """Perform simulation run.

        Initialize objects and run through all time steps of the
        simulation run. Call and coordinate other classes. Finally,
        process and store the simulation data in the result_dict.

        Args:
            seed: Random number generator with specified seed.
            result_dict (dict): Sort and store results.
            progress_bar (dict): Show percentage of progress.
            run_count: Count for the number of simulation run.
            number_of_runs: Total number of simulation runs.
            lock (ClassLock): ensure processes run parallel
            condition (ClassCondition): Ensure all initializations are finished.
        """
        self._initialize_simulation(progress_bar, number_of_runs, lock, condition, seed)
        self._run_time_steps(result_dict, progress_bar, number_of_runs, run_count, lock)

    def _run_time_steps(
        self, result_dict, progress_bar, number_of_runs, run_count, lock
    ):
        """Perform all time steps of simulation run.

        Iterate time steps, coordinate and call other classes to perform subtasks.
        Track results for each time step and for the whole simulation in the end.

        Args:
            result_dict (dict): Sort and store results.
            progress_bar (dict): Show percentage of progress.
            run_count: Count for the number of simulation run.
            number_of_runs: Total number of simulation runs.
            lock (ClassLock): ensure processes run parallel.
        """

        initialize_progress_bar(progress_bar, "Run Simulation")
        t3 = time.time()
        self.tracker.setup_run()
        for time_step in range(self.num_time_steps):
            date_of_time_step = self.start_date + dt.timedelta(days=time_step)
            if time_step > 0:
                self.net_manager.update_networks(self, time_step, date_of_time_step)
                self.epi_spreader.infect_agents(time_step)
                self.pub_reg.regulate(time_step, date_of_time_step)
            self.tracker.track(self._get_track_data(), time_step, date_of_time_step)
            value = 100 / self.num_time_steps / number_of_runs
            refresh_progress_bar(value, progress_bar, lock)
        result_dict[run_count] = self.tracker.track_run()
        t4 = time.time()
        print(f"Running {self.num_time_steps} time steps: {round(t4 - t3, 2)}s")

    def _get_track_data(self) -> dict:
        """Collect all data of the simulation.

        Returns:
            The union of the simulation's epidemic, agents and public
            data in the form of a dictionary.
        """
        track_data1 = self.people.get_epidemic_data()
        track_data2 = self.people.get_agents_data()
        track_data3 = self.people.get_public_data()
        return {**track_data1, **track_data2, **track_data3}

    @staticmethod
    def wait_for_all_processes(progress_bar, condition):
        """Time out to wait for all SimRunner classes to accomplish
        initialization.

        Args:
            progress_bar (dict): Show percentage of progress.
            condition (ClassCondition): Ensure all initializations are finished.
        """
        with condition:
            if progress_bar["progress"] > 97:
                condition.notify_all()
            else:
                condition.wait()


def refresh_progress_bar(value, progress_bar, lock=None):
    """Refresh progress bar after performing time step.

    Args:
        value (float): percentage to add
        progress_bar (dict): Show percentage of progress.
        lock (ClassLock): ensure processes run parallel.
    """
    if lock is None:
        progress_bar["progress"] += value
        print(f"{progress_bar['text']}: {round(progress_bar['progress'], 0)}")
    else:
        lock.acquire()
        progress_bar["progress"] += value
        lock.release()


def initialize_progress_bar(progress_bar, text: str = ""):
    """Initialize progress bar on Zero percentage and optional Text.

    Args:
        progress_bar (dict): Show percentage of progress.
        text (str): Optional text.
    """
    progress_bar["text"] = text
    progress_bar["progress"] = 0
    time.sleep(0.25)
