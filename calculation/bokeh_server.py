#!/venv/bin/python3
# -*- coding: utf-8 -*-

__author__ = "Felix Rieth"
__created__ = "2023"
__date_modified__ = "2023/05/31"
__version__ = "1.0"

import datetime as dt
import types

import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (
    Band,
    BoxZoomTool,
    Button,
    CheckboxButtonGroup,
    CheckboxGroup,
    ColumnDataSource,
    Div,
    HoverTool,
    NumeralTickFormatter,
    PanTool,
    RangeSlider,
    ResetTool,
    SaveTool,
    Select,
    Toggle,
)
from bokeh.palettes import Sunset
from bokeh.plotting import curdoc, figure

from synchronizer import constants as cs
from synchronizer.synchronizer import PathManager as PM
from synchronizer.synchronizer import Synchronizer

########################################################################################

########################################################################################
# Configuration and properties of bokeh plots.
KEYS = [
    cs.SUSCEPTIBLE,
    cs.INFECTED,
    cs.RECOVERED,
    cs.SEVERE,
    cs.CRITICAL,
    cs.DEAD,
]

KEYS_C = [
    cs.INFECTED,
    cs.RECOVERED,
    cs.SEVERE,
    cs.CRITICAL,
    cs.DEAD,
]

COLORS = types.MappingProxyType(
    {
        cs.SUSCEPTIBLE: Sunset[10][5],
        cs.INFECTED: Sunset[10][0],
        cs.RECOVERED: Sunset[10][3],
        cs.SEVERE: Sunset[10][7],
        cs.CRITICAL: Sunset[10][9],
        cs.DEAD: "black",
    }
)

WIDTH = 1000
HEIGHT = 500

BUTTON_HEIGHT = 25
BUTTON_WIDTH = 100
BUTTON_MARGIN = (10, 0, 10, 5)

SMALL_BUTTON_HEIGHT = 25
SMALL_BUTTON_WIDTH = 55
SMALL_BUTTON_MARGIN = (-5, 0, 8, 5)

DIV_MARGIN = (10, 0, -3, 5)

SELECT_MARGIN = (10, 0, 0, 5)

AGE_GROUPS = Synchronizer.AGE_CUTOFFS[cs.AGE_GR]
AGE_GROUP_MINIMUMS = Synchronizer.AGE_CUTOFFS[cs.MIN]
AGE_GROUP_MAXIMUMS = Synchronizer.AGE_CUTOFFS[cs.MAX]
AGE_GROUP_LABELS = [
    "{minimum}-{maximum}".format(
        minimum=AGE_GROUP_MINIMUMS[age_group],
        maximum=AGE_GROUP_MAXIMUMS[age_group],
    )
    for age_group in AGE_GROUPS
]

TOOLS_A = (
    PanTool(),
    BoxZoomTool(),
    HoverTool(
        tooltips=[
            ("people", "$y{0,0}"),
            ("date", "@date{%F}"),
        ],
        formatters={
            "@date": "datetime",
        },
        mode="mouse",
    ),
    SaveTool(),
    ResetTool(),
)

TOOLS_B = (
    PanTool(),
    BoxZoomTool(),
    HoverTool(
        tooltips=[
            ("people", "$y{0,0}"),
            ("date", "@date{%F}"),
        ],
        formatters={
            "@date": "datetime",
        },
        mode="mouse",
    ),
    SaveTool(),
    ResetTool(),
)

TOOLS_C = (
    PanTool(),
    BoxZoomTool(),
    HoverTool(
        tooltips=[
            ("people", "@value{0,0}"),
            ("date", "@left{%F}"),
        ],
        formatters={
            "@left": "datetime",
        },
        mode="mouse",
    ),
    SaveTool(),
    ResetTool(),
)


########################################################################################

########################################################################################
# Utility functions.
def read_source_data():
    data_path = PM.get_path_results()
    source_data = pd.read_csv(data_path, index_col=[0, 1], header=[0, 1, 2])
    source_data.index = source_data.index.set_levels(
        pd.to_datetime(source_data.index.levels[0]),
        level=0,
    )
    return source_data


def generate_source_a(age_groups: list = None) -> ColumnDataSource:
    """Docstring."""
    if age_groups is None:
        age_groups = AGE_GROUPS
    source_data = SOURCE_DATA.loc[(slice(None), age_groups), :]
    source_data = source_data.droplevel(cs.AGE_GROUP).groupby(cs.DATE).sum()
    return ColumnDataSource(source_data)


def generate_source_b(status: str) -> ColumnDataSource:
    """Docstring."""
    source_data = SOURCE_DATA.loc[:, ("mean", "active", status)].unstack(cs.AGE_GROUP)
    source_data.columns = source_data.columns.values.astype(str)
    return ColumnDataSource(source_data)


def generate_source_c(
    status: str,
    age_groups: list = None,
    cumulative: bool = False,
) -> ColumnDataSource:
    """Docstring."""
    if age_groups is None:
        age_groups = AGE_GROUPS
    source_data = SOURCE_DATA.loc[(slice(None), age_groups), :]
    source_data = source_data["mean"]["new"]
    source_data = source_data.droplevel(cs.AGE_GROUP).groupby(cs.DATE).sum()
    if cumulative:
        source_data = source_data.cumsum()
    status_values = source_data[status].values
    if status_values.size == 0:
        return ColumnDataSource(
            {"value": np.array([]), "left": [], "right": []},
        )
    dates = source_data.index
    edge_date_end = source_data.index[-1] + dt.timedelta(days=1)
    edges = source_data.index.tolist()
    edges.append(edge_date_end)
    return ColumnDataSource(
        {"value": status_values, "left": edges[:-1], "right": edges[1:]},
    )


########################################################################################

########################################################################################
# Read source data.
SOURCE_DATA = read_source_data()

########################################################################################

########################################################################################
# Plot A: Plot of active cases for combined age groups.
source_a = generate_source_a()

plot_a = figure(
    title="ACTIVE CASES",
    x_axis_label="date",
    y_axis_label="number of persons",
    x_axis_type="datetime",
    width=WIDTH,
    height=HEIGHT,
    tools=TOOLS_A,
)

lines_dict_a = {}
bands_dict_a = {}
for key in KEYS:
    lines_dict_a[key] = plot_a.line(
        x=cs.DATE,
        y="mean_active_{key}".format(key=key),
        source=source_a,
        legend_label=key,
        color=COLORS[key],
        line_width=2,
    )

    bands_dict_a[key] = Band(
        base=cs.DATE,
        lower="min_active_{key}".format(key=key),
        upper="max_active_{key}".format(key=key),
        source=source_a,
        fill_alpha=0.3,
        fill_color=COLORS[key],
        line_color=COLORS[key],
    )
    plot_a.add_layout(bands_dict_a[key])

plot_a.legend.location = "top_left"
plot_a.yaxis[0].formatter = NumeralTickFormatter(format="0,0")


########################################################################
# Plot A: CheckboxButtonGroup widget for statuses.
def update_status_a(attr, old, new):
    for status_index in old:
        status = buttongroup_status_a.labels[status_index]
        if status_index not in new:
            lines_dict_a[status].visible = False
            bands_dict_a[status].visible = False
    for status_index in new:
        status = buttongroup_status_a.labels[status_index]
        lines_dict_a[status].visible = True
        if toggle_bands_a.active:
            bands_dict_a[status].visible = True


buttongroup_status_a = CheckboxButtonGroup(
    labels=KEYS,
    active=list(range(len(KEYS))),
    orientation="vertical",
    width=BUTTON_WIDTH,
)
buttongroup_status_a.on_change("active", update_status_a)

div_status_a = Div(
    text="""Statuses:
    """,
    margin=DIV_MARGIN,
)


########################################################################
# Plot A: CheckboxGroup for age group filtering.
def update_age_groups_a(attr, old, new):
    new_source = generate_source_a(new)
    source_a.data.update(new_source.data)


checkboxes_age_group_a = CheckboxGroup(
    labels=AGE_GROUP_LABELS,
    active=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
)
checkboxes_age_group_a.on_change("active", update_age_groups_a)

div_age_groups_a = Div(
    text="""Age groups:
    """,
    margin=DIV_MARGIN,
)


########################################################################
# Plot A: Button widget to select all age groups.
def select_all_a():
    checkboxes_age_group_a.active = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


button_select_all_a = Button(
    label="all",
    height=SMALL_BUTTON_HEIGHT,
    width=SMALL_BUTTON_WIDTH,
    margin=SMALL_BUTTON_MARGIN,
    button_type="success",
)
button_select_all_a.on_click(select_all_a)


########################################################################
# Plot A: Button widget to clear selection of age groups.
def clear_selection_a():
    checkboxes_age_group_a.active = []


button_clear_selection_a = Button(
    label="clear",
    height=SMALL_BUTTON_HEIGHT,
    width=SMALL_BUTTON_WIDTH,
    margin=SMALL_BUTTON_MARGIN,
    button_type="danger",
)
button_clear_selection_a.on_click(clear_selection_a)


########################################################################
# Plot A: Toggle button for showing bands.
def update_bands_a():
    if toggle_bands_a.active:
        for status_index in buttongroup_status_a.active:
            bands_dict_a[buttongroup_status_a.labels[status_index]].visible = True
    else:
        for status_index in buttongroup_status_a.active:
            bands_dict_a[buttongroup_status_a.labels[status_index]].visible = False


toggle_bands_a = Toggle(
    label="Show bands",
    active=True,
    height=BUTTON_HEIGHT,
    width=BUTTON_WIDTH,
    margin=BUTTON_MARGIN,
    button_type="primary",
)

toggle_bands_a.on_event("button_click", update_bands_a)


########################################################################################

########################################################################################
# Plot B: Plot of active cases for separated age groups.
initial_status_b = cs.INFECTED
source_b = generate_source_b(initial_status_b)

plot_b = figure(
    title="ACTIVE CASES PER AGE GROUP",
    x_axis_label="date",
    y_axis_label="number of persons",
    x_axis_type="datetime",
    width=WIDTH,
    height=HEIGHT,
    tools=TOOLS_B,
)
lines_dict_b = {}
for age_group in AGE_GROUPS:
    lines_dict_b[age_group] = plot_b.line(
        x=cs.DATE,
        y="{age_group}".format(age_group=age_group),
        source=source_b,
        legend_label="{minimum}-{maximum}".format(
            minimum=(age_group * 10),
            maximum=(age_group * 10 + 9),
        ),
        color=Sunset[10][age_group],
        line_width=4,
    )

plot_b.legend.location = "top_left"
plot_b.yaxis[0].formatter = NumeralTickFormatter(format="0,0")


########################################################################
# Plot B: Select widget for statuses.
def update_status_b(attr, old, new):
    new_source = generate_source_b(new)
    source_b.data.update(new_source.data)


select_status_b = Select(
    title="Status:",
    options=KEYS,
    value=initial_status_b,
    margin=SELECT_MARGIN,
)

select_status_b.on_change("value", update_status_b)


########################################################################
# Plot B: CheckboxGroup for age group filtering.
def update_age_groups_b(attr, old, new):
    for age_group in old:
        if age_group not in new:
            lines_dict_b[age_group].visible = False
    for age_group in new:
        lines_dict_b[age_group].visible = True


checkboxes_age_group_b = CheckboxGroup(
    labels=AGE_GROUP_LABELS,
    active=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
)
checkboxes_age_group_b.on_change("active", update_age_groups_b)

div_age_groups_b = Div(
    text="""Age groups:
    """,
    margin=DIV_MARGIN,
)


########################################################################
# Plot B: Button widget to select all age groups.
def select_all_b():
    checkboxes_age_group_b.active = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


button_select_all_b = Button(
    label="all",
    height=SMALL_BUTTON_HEIGHT,
    width=SMALL_BUTTON_WIDTH,
    margin=SMALL_BUTTON_MARGIN,
    button_type="success",
)
button_select_all_b.on_click(select_all_b)


########################################################################
# Plot B: Button widget to clear selection of age groups.
def clear_selection_b():
    checkboxes_age_group_b.active = []


button_clear_selection_b = Button(
    label="clear",
    height=SMALL_BUTTON_HEIGHT,
    width=SMALL_BUTTON_WIDTH,
    margin=SMALL_BUTTON_MARGIN,
    button_type="danger",
)
button_clear_selection_b.on_click(clear_selection_b)

########################################################################################

########################################################################################
# Plot C: Plot of new cases for combined age groups.
initial_status_c = cs.INFECTED
source_c = generate_source_c(initial_status_c)


plot_c = figure(
    title="NEW CASES",
    x_axis_label="date",
    y_axis_label="number of persons",
    x_axis_type="datetime",
    width=WIDTH,
    height=HEIGHT,
    tools=TOOLS_C,
)

bars_c = plot_c.quad(
    top="value",
    bottom=0,
    left="left",
    right="right",
    source=source_c,
    line_color="white",
    fill_color=Sunset[10][3],
)

plot_c.yaxis[0].formatter = NumeralTickFormatter(format="0,0")


########################################################################
# Plot C: Select widget for statuses.
def update_status_c(attr, old, new):
    active_age_groups = checkboxes_age_group_c.active
    cumulative = toggle_cumulative_c.active
    new_source = generate_source_c(new, active_age_groups, cumulative)
    source_c.data.update(new_source.data)


select_status_c = Select(
    title="Status:",
    options=KEYS_C,
    value=initial_status_c,
    margin=SELECT_MARGIN,
)

select_status_c.on_change("value", update_status_c)


########################################################################
# Plot C: CheckboxGroup for age group filtering.
def update_age_groups_c(attr, old, new):
    status = select_status_c.value
    cumulative = toggle_cumulative_c.active
    new_source = generate_source_c(status, new, cumulative)
    source_c.data.update(new_source.data)


checkboxes_age_group_c = CheckboxGroup(
    labels=AGE_GROUP_LABELS,
    active=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
)
checkboxes_age_group_c.on_change("active", update_age_groups_c)

div_age_groups_c = Div(
    text="""Age groups:
    """,
    margin=DIV_MARGIN,
)


########################################################################
# Plot C: Button widget to select all age groups.
def select_all_c():
    checkboxes_age_group_c.active = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


button_select_all_c = Button(
    label="all",
    height=SMALL_BUTTON_HEIGHT,
    width=SMALL_BUTTON_WIDTH,
    margin=SMALL_BUTTON_MARGIN,
    button_type="success",
)
button_select_all_c.on_click(select_all_c)


########################################################################
# Plot C: Button widget to clear selection of age groups.
def clear_selection_c():
    checkboxes_age_group_c.active = []


button_clear_selection_c = Button(
    label="clear",
    height=SMALL_BUTTON_HEIGHT,
    width=SMALL_BUTTON_WIDTH,
    margin=SMALL_BUTTON_MARGIN,
    button_type="danger",
)
button_clear_selection_c.on_click(clear_selection_c)


########################################################################
# Plot C: Toggle button for cumulative cases.
def update_cumulative_c():
    active_age_groups = checkboxes_age_group_c.active
    status = select_status_c.value
    cumulative = toggle_cumulative_c.active
    new_source = generate_source_c(status, active_age_groups, cumulative)
    source_c.data.update(new_source.data)


toggle_cumulative_c = Toggle(
    label="Cumulative",
    active=False,
    height=BUTTON_HEIGHT,
    width=BUTTON_WIDTH,
    margin=BUTTON_MARGIN,
    button_type="primary",
)

toggle_cumulative_c.on_event("button_click", update_cumulative_c)


########################################################################################

########################################################################################
# Show plots
curdoc().add_root(
    column(
        row(
            plot_a,
            column(
                div_status_a,
                buttongroup_status_a,
                toggle_bands_a,
            ),
            column(
                div_age_groups_a,
                checkboxes_age_group_a,
                button_select_all_a,
                button_clear_selection_a,
            ),
        ),
        row(
            plot_b,
            column(
                select_status_b,
            ),
            column(
                div_age_groups_b,
                checkboxes_age_group_b,
                button_select_all_b,
                button_clear_selection_b,
            ),
        ),
        row(
            plot_c,
            column(
                select_status_c,
                toggle_cumulative_c,
            ),
            column(
                div_age_groups_c,
                checkboxes_age_group_c,
                button_select_all_c,
                button_clear_selection_c,
            ),
        ),
    ),
)
