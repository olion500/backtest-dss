SHELL := /bin/sh

IMAGE_NAME ?= backtest-dss
TAG ?= latest
DOCKER_IMAGE := $(IMAGE_NAME):$(TAG)
CONFIG ?= configs/sample_search_space.json
RUN_ARGS ?=
PRICES ?= data/sample_prices.csv
BACKTEST_ARGS ?=
TICKER ?= SOXL
START ?= 2024-01-01
END ?= 2025-01-01
INTERVAL ?= 1d
FETCH_ARGS ?=
EXEC ?=
DOCKER_RUN := docker run --rm -v "$(CURDIR)":/workspace -w /workspace $(DOCKER_IMAGE)

.PHONY: help build run backtest fetch-prices exec dry-run shell test clean

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  make help           Show this message."
	@echo "  make build          Build the Docker image ($(DOCKER_IMAGE))."
	@echo "  make run            Execute optimize_params.py inside Docker (CONFIG=$(CONFIG))."
	@echo "                      Append RUN_ARGS=\"...\" for additional CLI arguments."
	@echo "  make backtest       Run the local simulator (backtest_local.py)."
	@echo "                      Override PRICES=/path/to.csv and BACKTEST_ARGS=\"...\" as needed."
	@echo "  make fetch-prices   Download OHLCV data via Yahoo Finance."
	@echo "                      Set TICKER, START, END, INTERVAL, FETCH_ARGS for custom runs."
	@echo "  make exec           Run an arbitrary command inside the container."
	@echo "                      Example: make exec EXEC=\"python script.py --help\""
	@echo "  make dry-run        Run the optimiser with --dry-run through Docker."
	@echo "  make shell          Start an interactive shell in the container."
	@echo "  make test          Run the pytest suite inside Docker."
	@echo "  make clean          Remove the Docker image."

build:
	docker build --tag $(DOCKER_IMAGE) .

run:
	$(MAKE) exec EXEC="python optimize_params.py --config \"$(CONFIG)\" $(strip $(RUN_ARGS))"

backtest:
	$(MAKE) exec EXEC="python backtest_local.py --config \"$(CONFIG)\" --prices \"$(PRICES)\" $(strip $(BACKTEST_ARGS))"

fetch-prices:
	$(MAKE) exec EXEC="python fetch_prices.py \"$(TICKER)\" --start \"$(START)\" --end \"$(END)\" --interval \"$(INTERVAL)\" $(strip $(FETCH_ARGS))"

exec: build
	@if [ -z "$(strip $(EXEC))" ]; then \
		echo "Set EXEC=\"command\" to run inside the container"; \
		exit 1; \
	fi
	$(DOCKER_RUN) bash -lc '$(EXEC)'

dry-run: RUN_ARGS := --dry-run $(RUN_ARGS)
dry-run: run

shell:
	$(MAKE) exec EXEC="bash"

test:
	$(MAKE) exec EXEC="python -m pytest -q"

clean:
	- docker image rm $(DOCKER_IMAGE)
