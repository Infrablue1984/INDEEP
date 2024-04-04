#!/usr/bin/env python
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


def create_work_table(abs_file_name: str, out_path: str):
    df_work = FileLoader.read_excel_table(abs_file_name)
    df_work.drop(columns=df_work.columns[8:], inplace=True)
    columns = [
        "type",
        "min_size",
        "max_size",
        "mean_size",
        "%_people",
        "%_work",
        "alpha",
        "beta",
    ]
    df_work.columns = columns
    df_work.set_index("type", inplace=True)
    index = ["mini", "small", "medium", "big"]
    df_work.index = index
    df_work.index.name = "type"

    # export formatted table
    abs_file_name = os.path.join(out_path, "meta_work.xlsx")
    df_work.to_excel(abs_file_name, index=True)


if __name__ == "__main__":
    abs_path = Config.BUNDLE_DIR
    abs_file_name = os.path.join(
        abs_path, "data", "downloads", "Alle", "Firmengroessen.xlsx"
    )
    out_path = os.path.join(abs_path, "data", "inputs", "social_data")
    create_work_table(abs_file_name, out_path)
