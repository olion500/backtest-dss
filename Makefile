SHELL := /bin/sh

APP_NAME ?= dongpa
PORT ?= 8501
DEV_IMAGE := $(APP_NAME)-dev
DOCKER_RUN := docker run --rm -p $(PORT):8501 $(APP_NAME):latest

.PHONY: help install run-local build build-dev run shell dev clean optuna

help:
	@echo "Dongpa backtest helpers"
	@echo "------------------------"
	@echo "make install    Install Python dependencies with uv"
	@echo "make run-local  Launch Streamlit app on localhost:$(PORT) with uv"
	@echo "make build      Build production Docker image ($(APP_NAME):latest)"
	@echo "make run        Run the Dockerised app (maps $(PORT):8501)"
	@echo "make shell      Open a bash shell inside the app container"
	@echo "make build-dev  Force rebuild the live-reload dev image ($(DEV_IMAGE):latest)"
	@echo "make dev        Run dev container (auto-builds image if missing)"
	@echo "make optuna     Run 2-phase Optuna optimizer (300 wide + 900 focused)"
	@echo "make clean      Remove built Docker images"

install:
	uv sync

run-local:
	uv run streamlit run backtest.py --server.address=0.0.0.0 --server.port=$(PORT)

build:
	docker build --file Dockerfile --tag $(APP_NAME):latest .

build-dev:
	@echo "Building dev Docker image..."
	docker build --file Dockerfile.dev --tag $(DEV_IMAGE):latest .

run: build
	$(DOCKER_RUN)

shell: build
	docker run --rm -it -p $(PORT):8501 --entrypoint bash $(APP_NAME):latest

dev:
	@if ! docker image inspect $(DEV_IMAGE):latest > /dev/null 2>&1; then \
		echo "Dev image not found, building..."; \
		$(MAKE) build-dev; \
	fi
	@echo "Starting dev container on port $(PORT)..."
	@docker run --rm -it -p $(PORT):8501 -v "$(CURDIR)":/app $(DEV_IMAGE):latest

clean:
	- docker rmi $(APP_NAME):latest $(DEV_IMAGE):latest

optuna:
	uv run python run_optuna.py $(ARGS)
