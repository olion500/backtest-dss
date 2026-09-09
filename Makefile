SHELL := /bin/sh

APP_NAME ?= dongpa
PORT ?= 8501
DEV_IMAGE := $(APP_NAME)-dev
DOCKER_RUN := docker run --rm -p $(PORT):8501 $(APP_NAME):latest

# Deploy settings live in .env (gitignored); see .env.example.
-include .env
GCP_REGION ?= asia-northeast1
CLOUD_RUN_SERVICE ?= dongpa-viewer
DONGPA_RATE_LIMIT ?= 12
DEPLOY_IMAGE := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/dongpa/$(APP_NAME)-viewer:latest

.PHONY: help install run-local test build build-dev run shell dev clean web-install web-api web-frontend web-build web-run deploy

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
	@echo "make test       Run pytest suite"
	@echo "make web-install Install viewer frontend dependencies"
	@echo "make web-api    Run the FastAPI viewer API on port 8000"
	@echo "make web-frontend Run the Vite viewer on port 5173"
	@echo "make web-build  Build the production viewer image"
	@echo "make web    Build and run the viewer on port 8000"
	@echo "make deploy     Build, push, and deploy the viewer to Cloud Run"
	@echo "make clean      Remove built Docker images"

install:
	uv sync

run-local:
	uv run streamlit run main.py --server.address=0.0.0.0 --server.port=$(PORT)

test:
	uv run --group dev python -m pytest tests/ -v

web-install:
	npm --prefix web/frontend install

web-api:
	uv run uvicorn web.api.app.main:app --host 0.0.0.0 --port 8000 --reload

web-frontend:
	npm --prefix web/frontend run dev

web-build:
	docker build --file Dockerfile.web --tag $(APP_NAME)-viewer:latest .

# data/ is bind-mounted so the startup updater's fetches land in the repo:
# each boot only fetches the increment, and the dataset can be committed.
web: web-build
	docker run --rm -p 8000:8000 -v "$(CURDIR)/data":/app/data $(APP_NAME)-viewer:latest

deploy: web-build
	@test -n "$(GCP_PROJECT)" || { echo "GCP_PROJECT is not set — copy .env.example to .env first"; exit 1; }
	docker tag $(APP_NAME)-viewer:latest $(DEPLOY_IMAGE)
	docker push $(DEPLOY_IMAGE)
	gcloud run deploy $(CLOUD_RUN_SERVICE) \
		--project $(GCP_PROJECT) \
		--image $(DEPLOY_IMAGE) \
		--region $(GCP_REGION) \
		--allow-unauthenticated \
		--memory 1Gi --cpu 1 \
		--min-instances 0 --max-instances 1 \
		--timeout 120 \
		--iap \
		--set-env-vars DONGPA_RATE_LIMIT=$(DONGPA_RATE_LIMIT)

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
	- docker rmi $(APP_NAME):latest $(DEV_IMAGE):latest $(APP_NAME)-viewer:latest
