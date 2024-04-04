#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Docstring."""

__author__ = "Inga Franzen"
__created__ = "2021"
__date_modified__ = "2023/05/31"
__version__ = "1.0"


import os
import warnings
from os.path import exists

# import packages
import numpy as np
import pandas as pd
from pandas import ExcelWriter

from synchronizer.config import Config
from synchronizer.synchronizer import FileLoader, FileSaver


def read_excel_table(abs_file_name, sheet_name=0, dtype=None):
    # read original file
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df = pd.read_excel(
            abs_file_name, dtype=dtype, sheet_name=sheet_name, na_values=["NA"]
        )
    return df


def save_excel_table(path, file, df, sheet_name):
    abs_file_name = os.path.join(path, file)
    if exists(abs_file_name):
        with ExcelWriter(
            abs_file_name,
            mode="a",
            engine="openpyxl",
            if_sheet_exists="replace",
        ) as writer:
            df.to_excel(writer, sheet_name)
    else:
        writer = ExcelWriter(abs_file_name, engine="openpyxl")
        df.to_excel(writer, sheet_name)
        writer.close()


def format_federal_to_columns(df_schools):

    # modify data, delete unnecessary rows and columns, name columns
    df_schools.drop(index=[0, 1, 2, 3, 4, 5], inplace=True)
    df_schools.drop(columns=df_schools.columns[2:4], inplace=True)

    # rename columns and add columns
    df_schools.columns = ["type", "year", "total"]
    df_schools.index = range(0, len(df_schools))

    # create new table
    federals = [
        "Baden-Württemberg",
        "Bayern",
        "Berlin",
        "Brandenburg",
        "Bremen",
        "Hamburg",
        "Hessen",
        "Mecklenburg-Vorpommern",
        "Niedersachsen",
        "Nordrhein-Westfalen",
        "Rheinland-Pfalz",
        "Saarland",
        "Sachsen",
        "Sachsen-Anhalt",
        "Schleswig-Holstein",
        "Thüringen",
    ]

    filt = df_schools["type"] == "Bayern"
    t_len = df_schools.loc[
        filt,
    ].index[0]
    index = range(0, t_len)
    df_format = pd.DataFrame(index=index, columns=["type", "year"] + federals)

    # copy federals form lines into columns
    df_format["type"] = df_schools.loc[1:t_len, "type"]
    df_format["year"] = df_schools.loc[1:t_len, "year"]
    df_schools.set_index("type", inplace=True)
    for federal in range(0, len(federals)):
        this = federals[federal]
        if federal < 15:
            next = federals[federal + 1]
            begin_index = df_schools.index[df_schools.index == this][
                0
            ]  # [0] is to get index value instead of object
            end_index = df_schools.index[df_schools.index == next][0]
        else:
            begin_index = df_schools.index[df_schools.index == this][
                0
            ]  # [0] is to get index value instead of object
            end_index = df_schools.index[len(df_schools) - 3]
        df_fed = df_schools.loc[
            begin_index:end_index,
        ]
        df_fed = df_fed[:-1]["total"].tolist()
        assert len(df_format) == len(df_fed)
        df_format[this] = df_fed

    # drop school with not listet in "Statistisches Bundesamt, Schulen auf einen Blick, 2018"
    filt = df_format["type"] == "Abendhauptschulen"
    begin_index = df_format.loc[
        filt,
    ].index[0]
    filt = df_format["type"] == "Insgesamt"
    end_index = df_format.loc[
        filt,
    ].index[0]
    df_format.drop(index=range(begin_index, end_index), inplace=True)

    # rename lines
    df_format.index = range(0, len(df_format))

    return df_format


def calculate_percentage_for_type_and_year(df_format):
    # create new table for summed values
    df_format.replace("-", np.nan, inplace=True)
    s_types = df_format.loc[df_format["type"].notna(), "type"].to_numpy()
    year_gr = np.array(["5-10", "5-10 %", "11-13", "11-13 %"])
    federals = np.array(df_format.columns[2:])
    columns = pd.MultiIndex.from_product(
        [federals, year_gr], names=("Bundesland", "Jahrgangsstufe")
    )
    df_format2 = pd.DataFrame(index=s_types, columns=columns, dtype=float)
    # iterate school type in old table
    for idx in range(0, len(s_types)):
        s_type = s_types[idx]
        # filter data for each school type
        begin_index = df_format.loc[
            df_format["type"] == s_types[idx],
        ].index[0]
        if idx < len(s_types) - 1:
            end_index = (
                df_format.loc[
                    df_format["type"] == s_types[idx + 1],
                ].index[0]
                - 1
            )
        else:
            end_index = df_format.index[len(df_format) - 1]
        type_series = df_format.loc[
            begin_index:end_index,
        ]

        # set indices for iteration of federals
        begin_5 = type_series.loc[type_series["year"] == "Klassenstufe 5"].index[0]
        end_11 = type_series.loc[
            type_series["year"] == "Jahrgangsstufe 13 / Qualifizierungsphase Q2"
        ].index[0]
        # G8 with 5-9 and 11-13 is missing 10'th year. It is considered 5-10 and 11-12 due to real ages
        if s_type == "Gymnasien (G8)":
            end_5 = type_series.loc[
                type_series["year"] == "Jahrgangsstufe 11 / Einführungsphase E"
            ].index[0]
            begin_11 = type_series.loc[
                type_series["year"] == "Jahrgangsstufe 12 / Qualifizierungsphase Q1"
            ].index[0]
        else:
            end_5 = type_series.loc[type_series["year"] == "Klassenstufe 10"].index[0]
            begin_11 = type_series.loc[
                type_series["year"] == "Jahrgangsstufe 11 / Einführungsphase E"
            ].index[0]

        # iterate federals
        for federal in federals:
            # sum pupil for relevant years
            df_format2.at[s_type, (federal, "5-10")] = type_series.loc[
                begin_5:end_5, federal
            ].sum()
            df_format2.at[s_type, (federal, "11-13")] = type_series.loc[
                begin_11:end_11, federal
            ].sum()

    df_format2.drop(index="Insgesamt", inplace=True)

    # calculate %
    df_format2.fillna(0, inplace=True)
    for federal in federals:
        for line in df_format2.index:
            df_format2.at[line, (federal, "5-10 %")] = (
                df_format2.loc[line, (federal, "5-10")]
                / df_format2.loc[:, (federal, "5-10")].sum()
            )
            df_format2.at[line, (federal, "11-13 %")] = (
                df_format2.loc[line, (federal, "11-13")]
                / df_format2.loc[:, (federal, "11-13")].sum()
            )
        assert df_format2.loc[:, (federal, "5-10 %")].sum() < 1.000000001
        assert df_format2.loc[:, (federal, "5-10 %")].sum() > 0.99999999
        assert df_format2.loc[:, (federal, "11-13 %")].sum() < 1.00000001
        assert df_format2.loc[:, (federal, "11-13 %")].sum() > 0.99999999

    return df_format2


def integrate_and_drop_school_types(df):
    # now add school types if considered necessary
    df.loc["mit Oberstufe",] = (
        df.loc[
            "Schulartunabhängige Orientierungsstufen",
        ]
        + df.loc[
            "Integrierte Gesamtschulen",
        ]
        + df.loc[
            "Freie Waldorfschulen",
        ]
        + df.loc[
            "Gymnasien (G9)",
        ]
        + df.loc[
            "Gymnasien (G8)",
        ]
    )
    df.loc["ohne Oberstufe, klein",] = (
        df.loc[
            "Förderschulen",
        ]
        + df.loc[
            "Hauptschulen",
        ]
    )
    df.loc["ohne Oberstufe, groß",] = (
        df.loc[
            "Realschulen",
        ]
        + df.loc[
            "Schularten mit mehreren Bildungsgängen",
        ]
    )

    # drop unnecessary rows
    list_idx = [
        "Schulartunabhängige Orientierungsstufen",
        "Freie Waldorfschulen",
        "Gymnasien (G9)",
        "Gymnasien (G8)",
        "Vorklassen",
        "Schulkindergärten",
        "Grundschulen",
        "Förderschulen",
        "Schularten mit mehreren Bildungsgängen",
        "Hauptschulen",
        "Integrierte Gesamtschulen",
        "Realschulen",
    ]
    df.drop(index=list_idx, inplace=True)
    # drop unnecessary columns
    col4 = df.columns[3:-1:4]
    df.drop(columns=col4, inplace=True)
    col3 = df.columns[2:-1:3]
    df.drop(columns=col3, inplace=True)
    col1 = df.columns[0:-1:2]
    df.drop(columns=col1, inplace=True)
    col0 = df.columns[-1]
    df.drop(columns=col0, inplace=True)

    return df


def reformat_table(df):
    sorted_federal_list = [
        "Schleswig-Holstein",
        "Hamburg",
        "Niedersachsen",
        "Bremen",
        "Nordrhein-Westfalen",
        "Hessen",
        "Rheinland-Pfalz",
        "Baden-Württemberg",
        "Bayern",
        "Saarland",
        "Berlin",
        "Brandenburg",
        "Mecklenburg-Vorpommern",
        "Sachsen",
        "Sachsen-Anhalt",
        "Thüringen",
    ]
    new_name = dict(
        zip(
            sorted_federal_list,
            np.arange(1, 17),
        )
    )
    df = df.rename(columns=new_name, level=0)
    # df = df.sort_values(by=df.columns.names[0])
    df.columns = [t[0] for t in df.columns]
    df = df.T
    df.sort_index(inplace=True)
    df["Bundesland"] = sorted_federal_list
    df["Schlüsselnummer"] = df.index
    df.set_index("Schlüsselnummer", inplace=True)
    return df


def create_school_size_table(abs_file_name: str, out_path: str):
    df_sizes = FileLoader.read_excel_table(abs_file_name, sheet_name=0)
    df_sizes.rename(columns={"Größe": "sizes"}, inplace=True)
    df_sizes.drop(columns=df_sizes.columns[3:], inplace=True)
    df_sizes.set_index("Schlüsselnummer", inplace=True)
    save_excel_table(out_path, "meta_school.xlsx", df_sizes, "federal_sizes")
    df_sizes = FileLoader.read_excel_table(abs_file_name, sheet_name=1)
    df_sizes.drop(columns=df_sizes.columns[6:], inplace=True)
    my_columns = {
        "Typ": "type",
        "Größe": "size",
        "Klassengröße": "class_size",
        "Alter min": "min_age",
        "Alter max": "max_age",
        "Schüler je Lehrer": "pupil_per_teacher",
    }
    my_index = {
        "Grundschule": "primary",
        "Oberschule groß": "secondary big",
        "Oberschule mittel": "secondary medium",
        "Oberschule klein": "secondary small",
        "Kita": "kita",
    }
    df_sizes.rename(columns=my_columns, inplace=True)
    df_sizes.set_index("type", inplace=True)
    df_sizes.rename(index=my_index, inplace=True)
    FileSaver.save_excel_table(out_path, "meta_school.xlsx", df_sizes, "type_sizes")


def create_high_school_tables(abs_file_name: str, out_path: str):
    # read data
    df_schools = FileLoader.read_excel_table(abs_file_name)
    df_1 = format_federal_to_columns(df_schools)
    df_2 = calculate_percentage_for_type_and_year(df_1)
    df_3 = integrate_and_drop_school_types(df_2)
    df_4 = reformat_table(df_3)
    FileSaver.save_excel_table(out_path, "meta_school.xlsx", df_4, "pupil_5_10")
    # save_excel_table(out_path, "schools_1.xlsx", df_1, "Sheet")
    # save_excel_table(out_path, "schools_2.xlsx", df_2, "Sheet")
    # save_excel_table(out_path, "schools_3.xlsx", df_3, "Sheet")


if __name__ == "__main__":
    abs_path = Config.BUNDLE_DIR
    abs_file_name1 = os.path.join(
        abs_path, "data", "downloads", "Alle", "21111-0011.xlsx"
    )
    abs_file_name2 = os.path.join(
        abs_path, "data", "downloads", "Alle", "Schulgroessen.xlsx"
    )
    out_path = os.path.join(abs_path, "data", "inputs", "social_data")
    create_high_school_tables(abs_file_name1, out_path)
    create_school_size_table(abs_file_name2, out_path)
