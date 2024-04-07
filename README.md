# INDEEP (**In**fectious **D**is**e**ases - **E**mpidemiological **P**reestimation) - Agent-based model

## Description

The INDEEP agent-based model is developed to model the spread of diseases among people with a focus on transfer in public and private locations. Any agent is modelled with personal data (e.g. age, profession, region) and also personal health data (e.g. certain probabilities of susceptibility, infectivity). The spread of the disease takes place in different locations (e.g. schools, households or freetime activites). Any spreading event is modelled seperately, which allows for considering local parameters (e.g. time spent together, room size, air conditions). The spreading formula contains specific biological parameters (e.g. inhaltion rate, quantum doses, personal distance).

This agent-based model is part of the modeling tool INDEEP (**In**fectious **D**is**e**ases - **E**mpidemiological **P**reestimation) developed by the research group BIGAUGE as part of the ZNF, Universität Hamburg. INDEEP also contains a classical compartement model and user interface, which are not presented here.

## Installation

1. clone the repository
2. Recommended: create virtual environment (and activate)
3. Install packages

```bash
make install
```

Alternatively type the commands:

```bash
pip install -r requirements.txt
pre-commit install
```

## Usage

### Basic

A simple way of running the simulation is  with pre-defined values and produce visual output in form of some plots. Use the command:

```bash
make run
```

Alternatively type the commands:

```bash
 python3 StartUp.py
```

### Advanced

A more complicated way of running the simulation is via defining own parameter values. Therefore modify the input data manually in three different places:

- StartUp.py &rarr; _set_GUI function
- [COVID_default.csv ](./data/inputs/scenario)
- [interventions.xls](./data/inputs/social_data/interventions.xls)

The simulation can be run by either of the two commands or from an IDE.

```bash
1. make run
2. python3 StartUp.py
```

### Set Parameter

In order to modify the epidemiological data, open the [COVID_default.csv ](./data/inputs/scenario) and see for details on the parameters [here](./documentation/epidemic_parameters.md). To modify interventions, open [interventions.xls](./data/inputs/social_data/interventions.xls) and see for details on the interventions [here](./documentation/intervention_adjustment.md). Set basic parameters, e.g. start date and regions in the StartUp.py within the function _set_GUI.

### Update statistical data

1. Go to the folder ./data/downloads/Alle/

2. Search for the table in google and download data from the [Statistisches Bundesamt](https://www.destatis.de)

3. ```bash
   make table-update
   ```

## File Structure

StartUp.py &rarr; Module collecting the GUI data and start the simulation.

bokeh_server.py &rarr; module to present the results in a bokeh server.

### calculation folder

Modules to run the simulation.

### data_factory folder

Modules to transfer basic statistical data into locations (e.g. households, class rooms) with people assigned.

### data folder

Contains original and modified statistical data tables.

### synchronizer folder

Modules to define constants, create directories, synchronize variable names, file pathes, age cutoffs and numbers assigned to specific locations (e.g. 1 is for household, 2 for workplace) or table columns.

### tests

Unittests for each module and/or function.

## Disclaimer

The model is **not scientifically validated** and not yet meant to predict scenarios!

## Contributers

The code was developed by Inga Franzen and Felix Rieth with contributions from Jan Bürger and Sarah Albrecht.
