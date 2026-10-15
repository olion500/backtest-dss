SHELL := /bin/sh

APP_NAME ?= dongpa

# Deploy settings live in .env (gitignored); see .env.example.
-include .env
GCP_REGION ?= asia-northeast1
CLOUD_RUN_SERVICE ?= dongpa-viewer
DONGPA_RATE_LIMIT ?= 12
DEPLOY_IMAGE := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/dongpa/$(APP_NAME)-viewer:latest

.PHONY: help install test web-install web-api web-frontend web-build web deploy clean

help:
	@echo "Dongpa backtest helpers"
	@echo "------------------------"
	@echo "make install       Install Python dependencies with uv"
	@echo "make test          Run pytest suite"
	@echo "make web-install   Install frontend dependencies"
	@echo "make web-api       Run the FastAPI API on port 8000 (reload)"
	@echo "make web-frontend  Run the Vite dev server on port 5173"
	@echo "make web-build     Build the production image"
	@echo "make web           Build and run the app on port 8000"
	@echo "make deploy        Build, push, and deploy to Cloud Run"
	@echo "make clean         Remove the built Docker image"

install:
	uv sync

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

clean:
	- docker rmi $(APP_NAME)-viewer:latest
