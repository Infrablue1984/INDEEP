#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""Docstring."""

__author__ = "Inga Franzen"
__created__ = "2021"
__date_modified__ = "2023/05/31"
__version__ = "1.0"


import os

from synchronizer.config import Config
from synchronizer.synchronizer import FileLoader


def create_table_with_mean_contacts(abs_file_name: str, out_path: str):
    df_contact = FileLoader.read_excel_table(abs_file_name)
    df_contact.drop(columns=df_contact.columns[2:], inplace=True)
    columns = [
        "age",
        "mean contacts",
    ]
    df_contact.columns = columns
    df_contact.index = range(10)
    df_contact.index.name = "age group"

    # export formatted table
    abs_file_name = os.path.join(out_path, "mean_contacts.xlsx")
    df_contact.to_excel(abs_file_name, index=True)


if __name__ == "__main__":
    # TODO: Test different sizes of mean contacts and throw exceptions.
    abs_path = Config.BUNDLE_DIR
    abs_file_name = os.path.join(abs_path, "data", "downloads", "Alle", "Kontakte.xlsx")
    out_path = os.path.join(abs_path, "data", "inputs", "social_data")
    create_table_with_mean_contacts(abs_file_name, out_path)
