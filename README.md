# INDEEP (agent-based model)

## Description

The INDEEP agent-based model is developed to model the spread of diseases among



This agent-based modek is part of the modeling tool INDEEP (**In**fectious **D**is**e**ases - **E**mpidemiological **P**reestimation) developed by the research group BIGAUGE as part of the ZNF, Univerität Hamburg.

## Installation

1. clone the repository
2. Recommended: create virtual environment (and activate)
3. Install packages

```bash
make install
```

## Usage

### Basic

A simple way of running the simulation is via the command:

```bash
make run
```

This will run the simulation with pre-defined values and produce visual output in form of some plots.

### Advanced

A more complicated way of running the simulation is via defining own parameter values. Therefore open the file StartUp.py and modify the input data in the function _set_GUI manually. The simulation can be run by either of the two commands:

```bash
1. make run
2. python3 StartUp.py
```

### Set Parameter

In order to modify the epidemiological data, open the [COVID_default.csv ](./data/inputs/scenario) and see for details on the parameters [here](./documentation/epidemic_parameters.md). To modify interventions, open [interventions.xls](./data/inputs/social_data/interventions.xls) and see for details on the interventions [here](./documentation/intervention_adjustment.md). Set basic parameters, such as start and end date or the regions that should be modeled in the StartUp.py manually.

## Disclaimer

The model is **not scientifically validated** and not yet meant to predict scenarios.

## Contributers

The code was developed by Inga Franzen and Felix Rieth with contributions from Jan Bürger and Sarah Albrecht.
