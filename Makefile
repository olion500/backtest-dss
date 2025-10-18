SHELL := /bin/sh

APP_NAME ?= dongpa
PORT ?= 8501
DEV_IMAGE := $(APP_NAME)-dev
DOCKER_RUN := docker run --rm -p $(PORT):8501 $(APP_NAME):latest

.PHONY: help install run-local build build-dev run shell dev clean

help:
	@echo "Dongpa backtest helpers"
	@echo "------------------------"
	@echo "make install    Install Python dependencies locally"
	@echo "make run-local  Launch Streamlit app on localhost:$(PORT)"
	@echo "make build      Build production Docker image ($(APP_NAME):latest)"
	@echo "make run        Run the Dockerised app (maps $(PORT):8501)"
	@echo "make shell      Open a bash shell inside the app container"
	@echo "make build-dev  Build the live-reload dev image ($(DEV_IMAGE):latest)"
	@echo "make dev        Run dev container with source mounted"
	@echo "make clean      Remove built Docker images"

install:
	python -m pip install --requirement requirements.txt

run-local:
	streamlit run app_dongpa.py --server.address=0.0.0.0 --server.port=$(PORT)

build:
	docker build --file Dockerfile --tag $(APP_NAME):latest .

build-dev:
	docker build --file Dockerfile.dev --tag $(DEV_IMAGE):latest .

run: build
	$(DOCKER_RUN)

shell: build
	docker run --rm -it -p $(PORT):8501 --entrypoint bash $(APP_NAME):latest

dev: build-dev
	docker run --rm -it -p $(PORT):8501 -v "$(CURDIR)":/app $(DEV_IMAGE):latest

clean:
	- docker rmi $(APP_NAME):latest $(DEV_IMAGE):latest
