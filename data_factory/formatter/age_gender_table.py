#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Transform age and gender data from census into code friendly format.

A new pandas dataframe is created and the original data from census are
processed and written into the new dataframe. The original data are
taken from the folder downloads and the new dataframe is saved into the
folder inputs.
"""

__author__ = "Inga Franzen"
__created__ = "2020"
__date_modified__ = "2022/11/08"
__version__ = "1.0"

import os
import warnings

# import packages
import numpy as np
import pandas as pd

from synchronizer.config import Config
from synchronizer.synchronizer import FileLoader

# ToDo: occupy on warning regarding xlwt package


def read_table(path, file):
    # read original file
    # TODO: path ist bereits abs_file_name
    abs_file_name = os.path.join(path, file)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df = pd.read_excel(abs_file_name, dtype=str, na_values=["NA"])
    return df


def create_age_dist_table(abs_file_name, out_path):
    """Create table for age distribution."""
    df_ages = FileLoader.read_excel_table(abs_file_name, dtype=str)
    # drop disturbing first lines und column
    df_ages.drop(
        columns=["Bevölkerung: Bundesländer, Stichtag, Altersjahre"], inplace=True
    )
    df_ages.drop(index=[0, 1, 2, 3, 4], inplace=True)
    df_ages.drop(index=[96, 97, 98, 99, 100, 101, 102], inplace=True)

    # change name of states to federal key and re-sort
    df_ages.rename(
        columns={
            "Unnamed: 1": "08",
            "Unnamed: 2": "09",
            "Unnamed: 3": "11",
            "Unnamed: 4": "12",
            "Unnamed: 5": "04",
            "Unnamed: 6": "02",
            "Unnamed: 7": "06",
            "Unnamed: 8": "13",
            "Unnamed: 9": "03",
            "Unnamed: 10": "05",
            "Unnamed: 11": "07",
            "Unnamed: 12": "10",
            "Unnamed: 13": "14",
            "Unnamed: 14": "15",
            "Unnamed: 15": "01",
            "Unnamed: 16": "16",
        },
        inplace=True,
    )
    df_ages = df_ages.reindex(sorted(df_ages.columns), axis=1)

    # add column for ages and set to index
    df_ages["Alter"] = np.arange(len(df_ages))
    df_ages.set_index("Alter", inplace=True)

    # add ages 91 to 99 and distribute age > 90 evenly to these ages
    data_bigger_90 = {
        "01": int(df_ages.iloc[90, 0]) / 10,
        "02": int(df_ages.iloc[90, 1]) / 10,
        "03": int(df_ages.iloc[90, 2]) / 10,
        "04": int(df_ages.iloc[90, 3]) / 10,
        "05": int(df_ages.iloc[90, 4]) / 10,
        "06": int(df_ages.iloc[90, 5]) / 10,
        "07": int(df_ages.iloc[90, 6]) / 10,
        "08": int(df_ages.iloc[90, 7]) / 10,
        "09": int(df_ages.iloc[90, 8]) / 10,
        "10": int(df_ages.iloc[90, 9]) / 10,
        "11": int(df_ages.iloc[90, 10]) / 10,
        "12": int(df_ages.iloc[90, 11]) / 10,
        "13": int(df_ages.iloc[90, 12]) / 10,
        "14": int(df_ages.iloc[90, 13]) / 10,
        "15": int(df_ages.iloc[90, 14]) / 10,
        "16": int(df_ages.iloc[90, 15]) / 10,
    }
    df_bigger_90 = pd.DataFrame(data=data_bigger_90, index=range(9))
    df_ages = pd.concat([df_ages, df_bigger_90], ignore_index=True)
    df_ages.loc[90] = df_ages.loc[91]

    # change total numbers to percentage
    for column in df_ages:
        df_ages[column] = df_ages[column].astype("int")
        df_ages[column] = [int(x) for x in df_ages[column]]
        total = df_ages[column].sum(0)
        df_ages[column] = [x / total for x in df_ages[column]]

    # add title for index again
    df_ages.index.name = "Alter"
    df_ages.columns = df_ages.columns.astype(int)

    # export formatted table
    abs_file_name = os.path.join(out_path, "age_distribution.xlsx")
    df_ages.to_excel(abs_file_name, index=True)


def create_gender_table(abs_file_name, out_path):
    """Create table for gender distribution."""
    df_gender = FileLoader.read_excel_table(abs_file_name, dtype=str)
    # drop disturbing first lines und column
    df_gender.drop(
        columns=[
            "Bevölkerung: Deutschland, Stichtag, Altersjahre,\nNationalität,"
            " Geschlecht/Familienstand",
            "Unnamed: 3",
            "Unnamed: 4",
        ],
        inplace=True,
    )
    df_gender.drop(index=[0, 1, 2, 3, 4, 5], inplace=True)
    df_gender.drop(index=[92, 93, 94, 95, 96, 97], inplace=True)

    # rename columns
    df_gender.rename(
        columns={"Unnamed: 1": "männlich", "Unnamed: 2": "weiblich"}, inplace=True
    )

    # add column for ages and set to index
    df_gender["Alter"] = np.arange(len(df_gender))
    df_gender.set_index("Alter", inplace=True)

    # take values of last row and add values till age 99
    male_last = df_gender.loc[len(df_gender) - 1, "männlich"]
    female_last = df_gender.loc[len(df_gender) - 1, "weiblich"]
    counter = len(df_gender)
    data_set = pd.DataFrame(
        [[int(male_last), int(female_last)]], columns=["männlich", "weiblich"]
    )
    while counter < 100:
        df_gender = pd.concat([df_gender, data_set], ignore_index=True)
        counter += 1

    # change total numbers to percentage
    for i in df_gender.index:
        df_gender.loc[i] = [int(x) for x in df_gender.loc[i]]
        total = df_gender.loc[i].sum(0)
        df_gender.loc[i] = [x / total for x in df_gender.loc[i]]

    # add title for index again
    df_gender.index.name = "Alter"

    # export formatted table
    abs_file_name = os.path.join(out_path, "gender_distribution.xlsx")
    df_gender.to_excel(abs_file_name, index=True)


if __name__ == "__main__":
    abs_path = Config.BUNDLE_DIR
    abs_file_name1 = os.path.join(
        abs_path, "data", "downloads", "Alle", "12411-0012.xlsx"
    )
    abs_file_name2 = os.path.join(
        abs_path, "data", "downloads", "Alle", "12411-0007.xlsx"
    )
    out_path = os.path.join(abs_path, "data", "inputs", "social_data")
    create_age_dist_table(abs_file_name1, out_path)
    create_gender_table(abs_file_name2, out_path)
