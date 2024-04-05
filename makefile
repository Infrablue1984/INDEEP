DATA = data_factory.formatter
UNITTEST = python -W ignore -m unittest

.PHONY: install
install:
	pip install -r requirements.txt
	pre-commit install


.PHONY: table-update
table-update:
	python -W ignore -m $(DATA).age_gender_table
	python -W ignore -m $(DATA).contact_table
	python -W ignore -m $(DATA).occupation_table
	python -W ignore -m $(DATA).regional_table
	python -W ignore -m $(DATA).school_table
	python -W ignore -m $(DATA).work_table

.PHONY: test
test:
	$(UNITTEST) tests.epidemics_test
	$(UNITTEST) tests.functions_test
	$(UNITTEST) tests.graphs_test
	$(UNITTEST) tests.interventions_test
	$(UNITTEST) tests.network_manager_test
	$(UNITTEST) tests.networks_test
	$(UNITTEST) tests.people_test
	$(UNITTEST) tests.public_regulator_test
	$(UNITTEST) tests.simulator_test
	$(UNITTEST) tests.tracker_test
