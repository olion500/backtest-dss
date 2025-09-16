SHELL := /bin/sh

IMAGE_NAME ?= backtest-dss
TAG ?= latest
DOCKER_IMAGE := $(IMAGE_NAME):$(TAG)
CONFIG ?= configs/sample_search_space.json
RUN_ARGS ?=
DOCKER_RUN := docker run --rm -it -v "$(CURDIR)":/workspace -w /workspace $(DOCKER_IMAGE)

.PHONY: help build run dry-run shell clean

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  make help           Show this message."
	@echo "  make build          Build the Docker image ($(DOCKER_IMAGE))."
	@echo "  make run            Execute optimize_params.py inside Docker (CONFIG=$(CONFIG))."
	@echo "                      Append RUN_ARGS=\"...\" for additional CLI arguments."
	@echo "  make dry-run        Run the optimiser with --dry-run through Docker."
	@echo "  make shell          Start an interactive shell in the container."
	@echo "  make clean          Remove the Docker image."

build:
	docker build --tag $(DOCKER_IMAGE) .

run: build
	$(DOCKER_RUN) python optimize_params.py --config "$(CONFIG)" $(strip $(RUN_ARGS))

dry-run: RUN_ARGS := --dry-run $(RUN_ARGS)
dry-run: run

shell: build
	$(DOCKER_RUN) bash

clean:
	- docker image rm $(DOCKER_IMAGE)
