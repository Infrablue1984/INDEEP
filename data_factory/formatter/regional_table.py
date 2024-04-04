#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Create people ids from census data such as county id or rural/urban.

A new pandas dataframe is created and the original data from census are
processed and written into the new dataframe. The original data are
taken from the folder downloads and the new dataframe is saved into the
folder inputs.
"""

__author__ = "Inga Franzen"
__created__ = "2020"
__date_modified__ = "2022/11/09"
__version__ = "1.0"

# import packages

import os
import numpy as np
import pandas as pd

from synchronizer.config import Config

"""Script to format original download files of regional statical data and save them 
into new tables"""


def create_city_table(abs_file_name: str, out_path: str):
    # read data
    sheet = "Städte"
    df_cities = pd.read_excel(abs_file_name, sheet, dtype=str, na_values=["NA"])
    # drop unnessecary lines and columns
    df_cities.drop(index=df_cities.index[range(2061, 2071)], inplace=True)
    df_cities.drop(index=df_cities.index[range(6)], inplace=True)
    df_cities.drop(columns=df_cities.columns[[4, 5, 8, 12]], inplace=True)
    df_cities.dropna(axis="index", how="all", inplace=True)
    column_names = [
        "Schlüsselnummer",
        "Landesschlüssel",
        "RB-Schlüssel",
        "Kreisschlüssel",
        "Stadt",
        "PLZ",
        "Bevölkerung",
        "männlich",
        "weiblich",
    ]
    df_cities = df_cities.set_axis(column_names, axis=1, copy=False)
    # create full key by concatenating single keys in df_cities
    df_cities["Schlüsselnummer"] = (
        df_cities["Landesschlüssel"]
        + df_cities["RB-Schlüssel"]
        + df_cities["Kreisschlüssel"]
    )
    # create integers
    my_columns1 = ["Schlüsselnummer", "Bevölkerung", "weiblich", "männlich"]
    my_columns2 = ["Bevölkerung", "weiblich", "männlich"]
    df_cities.dropna(axis="index", how="all", inplace=True)
    # df_cities[my_columns1] = df_cities[my_columns1].fillna(-1)
    for column in my_columns1:
        df_cities[column] = [int(x) for x in df_cities[column]]
    df_cities = add_sums_for_columns(df_cities, my_columns2)
    # refresh line numeration and export data
    df_cities.index = np.arange(len(df_cities))
    save_pop_data(df_cities, out_path, "pop_cities.xlsx")


def create_county_tables(abs_file_name: str, out_path: str):
    """Read and format original tables df_cities and df_counties."""

    # read raw data from download folder
    sheet = "Kreisfreie Städte u. Landkreise"
    df_counties = pd.read_excel(abs_file_name, sheet, dtype=str, na_values=["NA"])

    # drop some lines and columns and rename
    df_counties.drop(index=[0, 1, 2, 3, 4, 5], inplace=True)
    df_counties.drop(columns=df_counties.columns[[3, 4, 8]], inplace=True)
    subset = df_counties.columns[[1, 2]]
    df_counties.dropna(axis="index", how="all", subset=subset, inplace=True)
    column_names = [
        "Schlüsselnummer",
        "Regionale Bezeichnung",
        "Kreis / Landkreis",
        "Bevölkerung",
        "männlich",
        "weiblich",
    ]
    df_counties = df_counties.set_axis(column_names, axis=1, copy=False)

    # create single keys by splitting full key in df_counties
    df_counties["Landesschlüssel"] = df_counties["Schlüsselnummer"].str[0:2]
    df_counties["RB-Schlüssel"] = df_counties["Schlüsselnummer"].str[2]
    df_counties["Kreisschlüssel"] = df_counties["Schlüsselnummer"].str[3:]

    # create integers
    my_columns = ["Schlüsselnummer", "Bevölkerung", "weiblich", "männlich"]
    df_counties[my_columns] = df_counties[my_columns].fillna(0)
    for column in my_columns:
        df_counties[column] = [int(x) for x in df_counties[column]]
    # df_counties["Schlüsselnummer"] = [int(x) for x in df_counties["Schlüsselnummer"]]

    # create columns for rural and urban in df_counties
    abs_file_name = os.path.join(out_path, "pop_cities.xlsx")
    df_cities = pd.read_excel(abs_file_name)
    for i in df_counties.index:
        regional_key = df_counties.loc[i, "Schlüsselnummer"]
        number_of_people = df_cities.loc[
            df_cities["Schlüsselnummer"] == regional_key, "Bevölkerung"
        ]
        df_counties.loc[i, "städtisch"] = sum(int(y) for y in number_of_people)
        df_counties.loc[i, "ländlich"] = (
            df_counties.loc[i, "Bevölkerung"] - df_counties.loc[i, "städtisch"]
        )

    # filter keys for states and create new file 'df_states' with equivalent columns
    create_states_table(df_counties, out_path)
    # drop lines with keys for states in the original file
    filt = [x < 1000 for x in df_counties["Schlüsselnummer"]]
    df_counties.drop(df_counties[filt].index, inplace=True)

    # Export data
    df_counties.index = np.arange(len(df_counties))
    save_pop_data(df_counties, out_path, "pop_counties.xlsx")


def create_states_table(df_counties, out_path):
    """Create table df_states and overtake important information."""
    filt = [0 < x < 17 for x in df_counties["Schlüsselnummer"]]
    df_states = df_counties.loc[filt, ["Schlüsselnummer", "Regionale Bezeichnung"]]
    # format and clean data
    df_states = sum_population_for_big_key(df_counties, df_states, out_path)
    # rename column
    df_states.rename(columns={"Regionale Bezeichnung": "Bundesland"}, inplace=True)
    save_pop_data(df_states, out_path, "pop_states.xlsx")


def sum_population_for_big_key(df_counties, df_other, out_path):
    """Take and add numbers of total population, female, male, urban, and rural
    according to regional key."""
    df = df_other
    df.index = np.arange(len(df))
    # get cities
    abs_file_name = os.path.join(out_path, "pop_cities.xlsx")
    df_cities = pd.read_excel(abs_file_name)
    for i in df.index:
        # get total number, female, male
        big_key = df.loc[i, "Schlüsselnummer"]
        counties = [int(x / 1000) == big_key for x in df_counties["Schlüsselnummer"]]
        df.loc[i, "Bevölkerung"] = sum_column(df_counties, counties, "Bevölkerung")
        df.loc[i, "weiblich"] = sum_column(df_counties, counties, "weiblich")
        df.loc[i, "männlich"] = sum_column(df_counties, counties, "männlich")
        # get total urban / rural
        cities = [int(x / 1000) == big_key for x in df_cities["Schlüsselnummer"]]
        df.loc[i, "städtisch"] = sum_column(df_cities, cities, "Bevölkerung")
        df.loc[i, "ländlich"] = df.loc[i, "Bevölkerung"] - df.loc[i, "städtisch"]
    my_columns = ["Bevölkerung", "städtisch", "ländlich", "weiblich", "männlich"]
    df = add_sums_for_columns(df, my_columns)
    return df


def add_sums_for_columns(df, columns):
    # append new line for whole Germany with total sums
    data = {"Schlüsselnummer": 0, df.columns[1]: "Gesamt"}
    for column in columns:
        column_sum = df[column].sum(0)
        data[column] = column_sum
    new_line = pd.DataFrame(data=data, index=[0])
    df = pd.concat([df, new_line], ignore_index=True)
    return df


def sum_column(df_counties, indices, column):
    number_of_people = df_counties.loc[indices, column]
    number_of_people.dropna(axis="index", how="any", inplace=True)
    my_sum = sum(int(y) for y in number_of_people)
    return my_sum


def save_pop_data(df, out_path, file_name):
    """Update index to Schlüsselnummer."""
    df.set_index("Schlüsselnummer", inplace=True)
    abs_file_name = os.path.join(out_path, file_name)
    df.to_excel(abs_file_name, index=True)


if __name__ == "__main__":
    abs_path = Config.BUNDLE_DIR
    abs_file_name1 = os.path.join(
        abs_path, "data", "downloads", "Alle", "05-staedte.xlsx"
    )
    abs_file_name2 = os.path.join(
        abs_path, "data", "downloads", "Alle", "04-kreise.xlsx"
    )
    out_path = os.path.join(abs_path, "data", "inputs", "social_data")
    create_city_table(abs_file_name1, out_path)
    create_county_tables(abs_file_name2, out_path)
