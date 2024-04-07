#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Set GUI data, create simulation object and start the simulation.

Be the mediator between user interface and programming code. A real GUI
is part of an extended project. This is the simple code of the agent
based model.
"""

__author__ = "Inga Franzen"
__created__ = "2020"
__date_modified__ = "2023/06/06"
__version__ = "1.0"

import time

# import built-ins
from datetime import date
from datetime import timedelta as td

# import packages
import numpy as np
import pandas as pd

# import modules
from calculation.interventions import InterventionMaker as IM
from calculation.simulator import SimManager
from synchronizer.synchronizer import PathManager as PM


def _extract_counties_code_and_scale_from_input_date(my_regional_code):
    """Extract code of counties and scale from regional code.

    The regional code can symbolize one or more federal states, the whole of Germany or
    single counties. To find specific regional code, go to data/inputs/social_data:
    - pop_cities
    - pop_counties
    - pop_cities
    Case 1:
        E.g. [5, 7] with 5 for 'Nordrhein-Westfalen' and 7 for 'Rheinland-Pfalz'.
        Then all county codes from these states will be extracted.
    Case 2:
        [0] means calculation for whole Germany and all county codes will be extracted.
    Case 3:
        E.g: [5162, 1002, 1003] are county codes and will just be returned.
    Args:
        my_regional_code (list[int]): List of regional codes, e.g. [5, 7], [0] or
        [5162, 1002, 1003]

    Returns:
        counties (list[int]): List of codes of counties, e.g. [5162, 1002, 1003].
        scale (int): Number of people represented by a single agent.
    """
    df_counties = pd.read_excel(PM.get_path_counties())
    if my_regional_code[0] == 0:
        counties = df_counties["Schlüsselnummer"].tolist()
    elif my_regional_code[0] <= 16:
        filt = df_counties["Landesschlüssel"] == -1
        for region in my_regional_code:
            filt = (filt) | (df_counties["Landesschlüssel"] == region)
        counties = df_counties.loc[filt, "Schlüsselnummer"].tolist()
    else:
        counties = my_regional_code
    scale = 100
    return counties, scale


def set_GUI():
    """Be a static GUI.

    Set start data and read tables with start data. To manipulate epidemic and
    intervention data, go into the file and change input manually. A real GUI is
    part of an extended project. This is simply the code of the agent based model.

    Global variables:
    intervention_list (PandasDataFrame): Contains type and degree of different
    public interventions, such as mask wearing or home office.
    epidemic_user_input (PandasDataFrame): Contains input of the disease,
    such as number of initial cases or epidemic parameters.
    start_date: Date to start the simulation from.
    end_date: Date at which simulation ends.
    days (int): Number of time steps that the simulation will run.
    regions (list[int]): List with regional code.
    scale (int): Number of people represented by a single agent.
    """
    global intervention_list, epidemic_user_input, start_date, days, end_date
    global regional_code, scale, number_of_parallel_runs

    # read table with interventions and make a list of intervention objects.
    # data/inputs/social_data/interventions.xls --> Type1, Type2
    intervention_list = IM.initialize_interventions(PM.get_path_interventions())

    # create data to simulate epidemic
    # data/inputs/scenario/COVID_default.xls
    epidemic_user_input = pd.read_csv(
        "data/inputs/scenario/COVID_default.csv",
        usecols=["parameter", "value", "type"],
        dtype={"parameter": str, "value": float, "type": str},
        index_col="parameter",
    )

    # manually create dates
    start_date = date(2020, 9, 30)
    days = 60
    end_date = start_date + td(days=days)

    # manually choose regions
    # Example 1: my_regional_code = [5162, 1002, 1003, 1004, 1051, 3256]
    # Example 2: my_regional_code = [7]
    # To understand regional code, see docstring of method
    # _extract_counties_code_and_scale_from_input_date
    regional_code = [1002]
    # transfer regional code
    regional_code, scale = _extract_counties_code_and_scale_from_input_date(
        regional_code
    )
    number_of_parallel_runs = 5


def simulate():
    """Set seed, create ClassSimManager and run the simulation with global input data."""
    np.random.seed(10)
    set_GUI()
    print(f"Run simulation for my_regional_code: {regional_code}")
    sim = SimManager(
        start_date,
        end_date,
        intervention_list,
        epidemic_user_input,
        regional_code,
        scale,
    )
    t1 = time.time()
    # reset data when unexpected errors occur with:
    sim.reset_data()
    sim.run(number_of_parallel_runs)
    t2 = time.time()
    print(f"Run simulation in: {round(t2 - t1, 2)} s")
    sim.plot_results()


if __name__ == "__main__":
    simulate()
