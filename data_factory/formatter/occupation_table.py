#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Transform occupation data from census into code friendly format.

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

# import built-ins
import warnings
from os.path import exists
from pathlib import Path, PurePath

# import packages
import numpy as np
import pandas as pd
from pandas import ExcelWriter

from synchronizer.config import Config
from synchronizer.synchronizer import FileLoader


def read_table(path, file):
    # read original file
    abs_file_name = os.path.join(path, file)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df = pd.read_excel(abs_file_name, na_values=["NA"])
    return df


def create_base_occupations(abs_file_name, out_path: str):
    """Read and format data out of original dataframe for general
    occupations."""
    # read data
    df_occ = FileLoader.read_excel_table(abs_file_name)
    # modify data sheets, delete unnecessary rows and columns, name columns
    df_occ.drop(index=[0, 1, 2, 3], inplace=True)
    df_occ.drop(
        index=[
            726,
            727,
            728,
            729,
            730,
            731,
            732,
            733,
            734,
            735,
            736,
            737,
            738,
            739,
            740,
            741,
            742,
        ],
        inplace=True,
    )
    df_occ.columns = ["year", "total", "employed", "unemployed"]
    df_occ.replace("x", 0, inplace=True)
    df_occ.replace("/", 0, inplace=True)
    df_occ.fillna(0, inplace=True)
    df_occ.index = np.arange(len(df_occ))

    # split into female and male
    index = df_occ.loc[
        df_occ["year"] == "weiblich",
    ].index
    df_female = df_occ.iloc[df_occ.index[index[0] :]]
    df_male = df_occ.iloc[df_occ.index[: index[0]]]
    df_end = pd.DataFrame({"year": "Ende Altersgruppe"}, index=[999])
    df_female = pd.concat([df_female, df_end], ignore_index=True)
    df_male = pd.concat([df_male, df_end], ignore_index=True)
    df_female.index = np.arange(len(df_female))
    """Create new dataframe for occupations."""

    # create new dataframes for occupations
    # age groups are modified form 7,14 to 7,15 and 15,19 to 16,19 as people under 16 count as pupil
    data = {
        "Min bracket": [0, 1, 3, 7, 17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        "Max bracket": [0, 2, 6, 16, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 99],
    }
    df_occ = pd.DataFrame(data=data)
    # set columns
    df_occ["employed"] = 0.0
    df_occ["pupil"] = 0.0
    df_occ["kita"] = 0.0
    df_occ["retired"] = 0.0
    df_occ["None"] = 0.0
    # set data for kita child, values from destatis Statistisches Bundesamt
    df_occ.at[0, "kita"] = 0.02  # 1,8 % Pressemitteilung Nr. 380 vom 30. September 2020
    df_occ.at[0, "None"] = 0.98
    df_occ.at[
        1, "kita"
    ] = 0.34  # Kindertagesbetreuung Betreuungsquote von Kindern unter 6 Jahren nach Bundesländern
    df_occ.at[1, "None"] = 0.66
    df_occ.at[
        2, "kita"
    ] = 0.92  # Betreuungsquoten der Kinder unter 6 Jahren in Kindertagesbetreuung am 01.03.2021 nach Ländern
    df_occ.at[2, "None"] = 0.08

    df_occ_female = pd.DataFrame(df_occ.copy())
    df_occ_male = pd.DataFrame(df_occ.copy())

    # Define Age groups, the age groups are shifted by one age group
    # For Example: the value above '25 bis unter 30 Jahre' is the value for '15 bis unter 20 Jahre'
    # This is because the last value in the table, which we take, is above the next age group, but still belongs to
    # the age group before.
    age_groups = [
        "15 bis unter 20 Jahre",
        "20 bis unter 25 Jahre",
        "25 bis unter 30 Jahre",
        "30 bis unter 35 Jahre",
        "35 bis unter 40 Jahre",
        "40 bis unter 45 Jahre",
        "45 bis unter 50 Jahre",
        "50 bis unter 55 Jahre",
        "55 bis unter 60 Jahre",
        "60 bis unter 65 Jahre",
        "65 Jahre und mehr",
        "Ende Altersgruppe",
    ]

    # fill in the percentage of employed people according to age group
    # line = 4 in the target table equals age group 17 to 24
    line = 3
    begin_index = 0
    for age in age_groups:
        end_index = df_female.loc[
            df_female["year"] == age,
        ].index[0]
        df_female_age_group = df_female.loc[
            df_female.index[
                begin_index:end_index,
            ]
        ]
        df_occ_female.at[line, "employed"] = (
            df_female_age_group[-1:]["employed"] / df_female_age_group[-1:]["total"]
        )
        df_occ_female.at[line, "None"] = (
            df_female_age_group[-1:]["unemployed"] / df_female_age_group[-1:]["total"]
        )
        # subtable male with data for (the previous) age group
        df_male_age_group = df_male.loc[
            df_male.index[
                begin_index:end_index,
            ]
        ]
        df_occ_male.at[line, "employed"] = (
            df_male_age_group[-1:]["employed"] / df_male_age_group[-1:]["total"]
        )
        df_occ_male.at[line, "None"] = (
            df_male_age_group[-1:]["unemployed"] / df_male_age_group[-1:]["total"]
        )
        begin_index = end_index + 1
        line += 1

    # now estimate the percentages of occupations other than employed
    for line in range(3, len(df_occ_female)):
        # estimate, that all younger than 30 are pupil/students, if not employed or unemployed (=None)
        if df_occ_female.loc[line, "Min bracket"] < 30:
            df_occ_female.at[line, "pupil"] = (
                1.0
                - df_occ_female.loc[line, "employed"]
                - df_occ_female.loc[line, "None"]
            )
            df_occ_male.at[line, "pupil"] = (
                1.0 - df_occ_male.loc[line, "employed"] - df_occ_male.loc[line, "None"]
            )
        # estimate, that all 30 < people < 40 are either pupil/students or at home for other reasons (maybe parental care),
        # if not employed
        elif df_occ_female.loc[line, "Min bracket"] < 40:
            df_occ_female.at[line, "pupil"] = (
                1.0 - df_occ_female.loc[line, "employed"]
            ) * 0.5
            df_occ_female.at[line, "None"] = (
                1.0 - df_occ_female.loc[line, "employed"]
            ) * 0.5
            df_occ_male.at[line, "pupil"] = (
                1.0 - df_occ_male.loc[line, "employed"]
            ) * 0.5
            df_occ_male.at[line, "None"] = (
                1.0 - df_occ_male.loc[line, "employed"]
            ) * 0.5
        # if not employed, 40 < people < 65 must be either pupil/students, retired or None (big ???)
        # estimate 10 % pupil, 80% None and 10% retired
        elif df_occ_female.loc[line, "Min bracket"] < 65:
            df_occ_female.at[line, "pupil"] = (
                1.0 - df_occ_female.loc[line, "employed"]
            ) * 0.1
            df_occ_female.at[line, "retired"] = (
                1.0 - df_occ_female.loc[line, "employed"]
            ) * 0.1
            df_occ_female.at[line, "None"] = (
                1.0 - df_occ_female.loc[line, "employed"]
            ) * 0.8
            df_occ_male.at[line, "pupil"] = (
                1.0 - df_occ_male.loc[line, "employed"]
            ) * 0.1
            df_occ_male.at[line, "retired"] = (
                1.0 - df_occ_male.loc[line, "employed"]
            ) * 0.1
            df_occ_male.at[line, "None"] = (
                1.0 - df_occ_male.loc[line, "employed"]
            ) * 0.8
        # estimate, that all older 65 are retired, if not employed
        else:
            df_occ_female.at[line, "retired"] = (
                1.0
                - df_occ_female.loc[line, "employed"]
                - df_occ_female.loc[line, "None"]
            )
            df_occ_male.at[line, "retired"] = (
                1.0 - df_occ_male.loc[line, "employed"] - df_occ_male.loc[line, "None"]
            )
    df_occ_female.set_index("Min bracket", inplace=True)
    df_occ_male.set_index("Min bracket", inplace=True)
    abs_file_name = os.path.join(out_path, "occupations.xlsx")
    if exists(abs_file_name):
        with ExcelWriter(
            abs_file_name,
            mode="a",
            engine="openpyxl",
            if_sheet_exists="replace",
        ) as writer:
            df_occ_female.to_excel(writer, "female occupations")
            df_occ_male.to_excel(writer, "male occupations")
    else:
        writer = ExcelWriter(abs_file_name, engine="openpyxl")
        df_occ_female.to_excel(writer, "female occupations")
        df_occ_male.to_excel(writer, "male occupations")
        writer.close()


def create_special_occupations(abs_file_name: str, out_path: str):
    """Read and format data out of original dataframe for special
    occupations."""
    # read data sheets
    SVB = pd.read_excel(f"{abs_file_name}", "SVB - Tabelle I", na_values=["NA"])
    GB = FileLoader.read_excel_table(abs_file_name, sheet_name="GB - Tabelle I")

    # load tables to to list
    data_sheets = [SVB, GB]

    # modify data sheets, delete unnecessary rows and columns, name columns
    for sheet in data_sheets:
        if len(sheet.columns) == 14:
            sheet.columns = [
                "Tätigkeit",
                "Gesamt",
                "Männer",
                "Frauen",
                "Deutsche",
                "Ausländer",
                "unter 25",
                "25-55",
                "55-65",
                "älter 65",
                "Regelgrenze",
                "Azubi Gesamt",
                "Azubi Männer",
                "Azubi Frauen",
            ]
            sheet.drop(
                columns=[
                    "Azubi Gesamt",
                    "Azubi Männer",
                    "Azubi Frauen",
                    "Deutsche",
                    "Ausländer",
                ],
                inplace=True,
            )
        if len(sheet.columns) == 11:
            sheet.columns = [
                "Tätigkeit",
                "Gesamt",
                "Männer",
                "Frauen",
                "Deutsche",
                "Ausländer",
                "unter 25",
                "25-55",
                "55-65",
                "älter 65",
                "Regelgrenze",
            ]
            sheet.drop(columns=["Deutsche", "Ausländer"], inplace=True)

        sheet.drop(index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 886, 887], inplace=True)
        sheet.drop(sheet[sheet["Tätigkeit"] == "davon:"].index)
        sheet.replace(["*", "-"], np.nan, inplace=True)
        sheet.set_index("Tätigkeit", inplace=True)

    # built sum out of full- and part-time tables
    total = pd.DataFrame(SVB.copy())
    for column in total.columns:
        total[column] = SVB[column] + GB[column]

    """create new dataframes for special groups of occupation"""

    # create new dataframe for female special groups of occupation
    data = {
        "Min bracket": [17, 25, 55, 65, 67, 17],
        "Max bracket": [24, 54, 64, 66, 99, 99],
    }
    df_female = pd.DataFrame(data=data)

    # set columns
    df_female["medical"] = 0.0
    df_female["elderly care"] = 0.0
    df_female["education"] = 0.0
    df_female["gastronomy"] = 0.0
    df_female["employed"] = 0.0
    df_female["total"] = 0.0

    # create new dataframe for male special groups of occupation
    df_male = pd.DataFrame(df_female.copy())

    # get medical
    med_filt = (
        (total.index == "811 Arzt- und Praxishilfe")
        | (total.index == "813 Gesundh.,Krankenpfl.,Rettungsd.Geburtsh.")
        | (total.index == "814 Human- und Zahnmedizin")
        | (total.index == "816 Psychologie, nichtärztl. Psychotherapie")
        | (total.index == "817 Nicht ärztliche Therapie und Heilkunde")
    )
    df_med = total.loc[
        med_filt,
    ].copy()
    df_med.index = [0, 1, 2, 3, 4]
    for column in df_med:
        df_med.at[0, column] = df_med[column].sum()
    df_med.drop(index=[1, 2, 3, 4], inplace=True)

    # get education
    edu_filt = (total.index == "831 Erziehung,Sozialarb.,Heilerziehungspfl.") | (
        total.index == "84 Lehrende und ausbildende Berufe"
    )
    df_edu = total.loc[
        edu_filt,
    ].copy()
    df_edu.index = [0, 1]
    for column in df_edu:
        df_edu.at[0, column] = df_edu[column].sum()
    df_edu.drop(index=[1], inplace=True)

    # get gastronomy
    df_gast = total.loc[
        total.index == "633 Gastronomie",
    ]
    df_gast.index = [0]

    # get elderly care
    df_elder = total.loc[
        total.index == "821 Altenpflege",
    ]
    df_elder.index = [0]

    # get total employed
    df_total_empl = total.loc[
        total.index == "Insgesamt",
    ]
    df_total_empl.index = [0]

    # load sub-dataframes of special groups to a_gr list
    df_groups = [df_med, df_elder, df_edu, df_gast]

    # iterate list and write values into new dataframe
    column = 0
    for frame in df_groups:
        # % male/ female according to occupation divided by % male/ female according to total employed
        per_male = (frame.loc[0, "Männer"] / frame.loc[0, "Gesamt"]) / (
            total.loc["Insgesamt", "Männer"] / total.loc["Insgesamt", "Gesamt"]
        )
        per_female = (
            frame.loc[0, "Frauen"]
            / frame.loc[0, "Gesamt"]
            / (total.loc["Insgesamt", "Frauen"] / total.loc["Insgesamt", "Gesamt"])
        )
        df_female.iat[0, 2 + column] = (
            frame.loc[0, "unter 25"] / total.loc["Insgesamt", "unter 25"] * per_female
        )
        df_male.iat[0, 2 + column] = (
            frame.loc[0, "unter 25"] / total.loc["Insgesamt", "unter 25"] * per_male
        )
        df_female.iat[1, 2 + column] = (
            frame.loc[0, "25-55"] / total.loc["Insgesamt", "25-55"] * per_female
        )
        df_male.iat[1, 2 + column] = (
            frame.loc[0, "25-55"] / total.loc["Insgesamt", "25-55"] * per_male
        )
        df_female.iat[2, 2 + column] = (
            frame.loc[0, "55-65"] / total.loc["Insgesamt", "55-65"] * per_female
        )
        df_male.iat[2, 2 + column] = (
            frame.loc[0, "55-65"] / total.loc["Insgesamt", "55-65"] * per_male
        )
        df_female.iat[3, 2 + column] = (
            frame.loc[0, "Regelgrenze"]
            / total.loc["Insgesamt", "Regelgrenze"]
            * per_female
        )
        df_male.iat[3, 2 + column] = (
            frame.loc[0, "Regelgrenze"]
            / total.loc["Insgesamt", "Regelgrenze"]
            * per_male
        )
        df_female.iat[4, 2 + column] = (
            (frame.loc[0, "älter 65"] - frame.loc[0, "Regelgrenze"])
            / (
                total.loc["Insgesamt", "älter 65"]
                - total.loc["Insgesamt", "Regelgrenze"]
            )
            * per_female
        )
        df_male.iat[4, 2 + column] = (
            (frame.loc[0, "älter 65"] - frame.loc[0, "Regelgrenze"])
            / (
                total.loc["Insgesamt", "älter 65"]
                - total.loc["Insgesamt", "Regelgrenze"]
            )
            * per_male
        )
        df_female.iat[5, 2 + column] = (
            frame.loc[0, "Gesamt"] / total.loc["Insgesamt", "Gesamt"] * per_female
        )
        df_male.iat[5, 2 + column] = (
            frame.loc[0, "Gesamt"] / total.loc["Insgesamt", "Gesamt"] * per_male
        )
        column += 1

    # add employed and total
    for line in range(6):
        df_female.at[line, "employed"] = (
            1
            - df_female.loc[line, "medical"]
            - df_female.loc[line, "elderly care"]
            - df_female.loc[line, "education"]
            - df_female.loc[line, "gastronomy"]
        )
        df_male.at[line, "employed"] = (
            1
            - df_male.loc[line, "medical"]
            - df_male.loc[line, "elderly care"]
            - df_male.loc[line, "education"]
            - df_male.loc[line, "gastronomy"]
        )
        df_female.at[line, "total"] = (
            df_female.loc[line, "medical"]
            + df_female.loc[line, "elderly care"]
            + df_female.loc[line, "education"]
            + df_female.loc[line, "gastronomy"]
            + df_female.loc[line, "employed"]
        )
        df_male.at[line, "total"] = (
            df_male.loc[line, "medical"]
            + df_male.loc[line, "elderly care"]
            + df_male.loc[line, "education"]
            + df_male.loc[line, "gastronomy"]
            + df_male.loc[line, "employed"]
        )

    df_female.set_index("Min bracket", inplace=True)
    df_male.set_index("Min bracket", inplace=True)
    # export dataframe
    abs_file_name = os.path.join(out_path, "occupations.xlsx")
    if exists(abs_file_name):
        with ExcelWriter(
            abs_file_name,
            mode="a",
            engine="openpyxl",
            if_sheet_exists="replace",
        ) as writer:
            df_female.to_excel(writer, "special female")
            df_male.to_excel(writer, "special male")
    else:
        writer = ExcelWriter(abs_file_name, engine="openpyxl")
        df_female.to_excel(writer, "special female")
        df_male.to_excel(writer, "special male")
        writer.close()


if __name__ == "__main__":
    abs_path = Config.BUNDLE_DIR
    abs_file_path1 = os.path.join(
        abs_path, "data", "downloads", "Alle", "12211-0003.xlsx"
    )
    abs_file_path2 = os.path.join(
        abs_path, "data", "downloads", "Alle", "bo-heft-d-0-201912-xlsx.xlsx"
    )
    out_path = os.path.join(abs_path, "data", "inputs", "social_data")
    create_base_occupations(abs_file_path1, out_path)
    create_special_occupations(abs_file_path2, out_path)
    # just to test something
    file = "bo-heft-d-0-201912-xlsx.xlsx"
    pattern = "bo-heft-d-0-*-xlsx.xlsx"
    test = PurePath(file).match(pattern)
