# HW2 is installed into the UniLab checkout's virtual environment. Point
# UNILAB_DIR at that checkout (students: the sibling clone of UniLab).
UNILAB_DIR ?= ../UniLab
HW2_DIR := $(abspath .)
# A VIRTUAL_ENV inherited from another project would shadow the UniLab venv.
UV := env -u VIRTUAL_ENV uv

.PHONY: install
install:
	cd $(UNILAB_DIR) && $(UV) pip install -e $(HW2_DIR)

.PHONY: lint
lint:
	$(UV) run --project $(UNILAB_DIR) --no-sync ruff check src
	$(UV) run --project $(UNILAB_DIR) --no-sync ruff format --check src

.PHONY: format
format:
	$(UV) run --project $(UNILAB_DIR) --no-sync ruff format src
	$(UV) run --project $(UNILAB_DIR) --no-sync ruff check --fix src
