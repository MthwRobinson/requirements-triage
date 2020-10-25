#######################
# Linting and Testing
#######################

lint:
	black reqs_triage --check
	flake8

lint-black:
	black reqs_triage --check
	black test_reqs_triage --check

tidy:
	black reqs_triage
	black test_reqs_triage

test:
	pytest test_reqs_triage --cov=reqs_triage -vv

################
# Install
################

pip-compile:
	pip-compile requirements/base.in
	pip-compile requirements/dev.in
	pip-compile requirements/test.in

pip-install:
	pip install -r requirements/base.txt
	pip install -r requirements/dev.txt
	pip install -r requirements/test.txt
	pip install -e .
