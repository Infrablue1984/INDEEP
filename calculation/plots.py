#!/venv/bin/python3
# -*- coding: utf-8 -*-
"""This module is responsible for the Plotter class.

Plot data of the simulation. # TODO(Felix): Finish this.
"""

__author__ = "Inga Franzen, Felix Rieth"
__created__ = "2023"
__date_modified__ = "2023/05/31"
__version__ = "1.0"

import os
from abc import ABC, abstractmethod

import pandas as pd

from synchronizer.synchronizer import PathManager as PM
from synchronizer.synchronizer import Synchronizer as Sync


class PlotInterface(ABC):
    """Plots the results of the simulation."""

    @abstractmethod
    def plot_results(self, *args, **kwargs):
        """Plot the results of the simulation.

        Args:
            args: Arguments of the plot_results method.
            kwargs: Keyword arguments of the plot_results method.
        """


class Plotter(PlotInterface):
    """Plots the results of the simulation."""

    def plot_results(self):
        """Plot the results of the simulation on a bokeh server."""
        server = PM.get_path_bokeh_server()
        os.system(f"bokeh serve --show {server}")
