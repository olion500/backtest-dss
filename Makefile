SHELL := /bin/sh

APP_NAME ?= dongpa

# Everything personal (GCP project, domain, who may sign in, state bucket) lives in
# .env (gitignored); see .env.example. Nothing in the repo names one person's setup.
-include .env
GCP_REGION ?= asia-northeast1
CLOUD_RUN_SERVICE ?= dongpa-viewer
TRADER_BUCKET ?= $(GCP_PROJECT)-dongpa-trader
# No GCP project: records stay in var/trader (local) instead of a made-up bucket name.
DONGPA_TRADER_RUNS ?= $(if $(GCP_PROJECT),gs://$(TRADER_BUCKET)/runs,)
PULUMI_STATE ?= gs://$(GCP_PROJECT)-pulumi
PULUMI_STACK ?= prod
DEPLOY_IMAGE := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/dongpa/$(APP_NAME)-viewer:latest
# Pulumi CLI (curl -fsSL https://get.pulumi.com | sh). The stack has no secrets, so the passphrase is empty.
# infra/__main__.py reads the .env values from the environment; GOOGLE_* configure the gcp provider.
PULUMI ?= $(HOME)/.pulumi/bin/pulumi
PULUMI_ENV := PULUMI_CONFIG_PASSPHRASE="" PULUMI_SKIP_UPDATE_CHECK=1 PATH="$(HOME)/.pulumi/bin:$$PATH" \
	PULUMI_BACKEND_URL="$(PULUMI_STATE)" GOOGLE_PROJECT="$(GCP_PROJECT)" GOOGLE_REGION="$(GCP_REGION)" \
	GCP_PROJECT="$(GCP_PROJECT)" GCP_REGION="$(GCP_REGION)" CLOUD_RUN_SERVICE="$(CLOUD_RUN_SERVICE)" \
	VIEWER_DOMAIN="$(VIEWER_DOMAIN)" IAP_MEMBERS="$(IAP_MEMBERS)" TRADER_BUCKET="$(TRADER_BUCKET)" \
	VIEWER_IMAGE="$(DEPLOY_IMAGE)" DONGPA_RATE_LIMIT="$(DONGPA_RATE_LIMIT)"

.PHONY: help check-env install test web-install web-api web-frontend web-build web image-push deploy infra-init infra-preview infra-up trader-check clean

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
	@echo "make image-push    Build and push the image to Artifact Registry"
	@echo "make deploy        Build, push, and roll the new image on Cloud Run"
	@echo "make infra-init    First time only: create the Pulumi state bucket and stack"
	@echo "make infra-preview Show what Pulumi would change on GCP (infra/)"
	@echo "make infra-up      Apply infra/ to GCP"
	@echo "make trader-check  Reconcile every account in config/trader (read only)"
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

check-env:
	@test -n "$(GCP_PROJECT)" || { echo "GCP_PROJECT is not set — copy .env.example to .env first"; exit 1; }

image-push: check-env web-build
	docker tag $(APP_NAME)-viewer:latest $(DEPLOY_IMAGE)
	docker push $(DEPLOY_IMAGE)

deploy: image-push
	# Only the image: env, scaling, IAP and IAM belong to infra/ (Pulumi).
	gcloud run deploy $(CLOUD_RUN_SERVICE) \
		--project $(GCP_PROJECT) \
		--region $(GCP_REGION) \
		--image $(DEPLOY_IMAGE)

infra-init: check-env
	gcloud storage buckets describe $(PULUMI_STATE) --project $(GCP_PROJECT) >/dev/null 2>&1 || \
		gcloud storage buckets create $(PULUMI_STATE) --project $(GCP_PROJECT) --location $(GCP_REGION) \
			--uniform-bucket-level-access --public-access-prevention
	gcloud storage buckets update $(PULUMI_STATE) --versioning
	cd infra && $(PULUMI_ENV) $(PULUMI) stack select --create $(PULUMI_STACK)

infra-preview: check-env
	cd infra && $(PULUMI_ENV) $(PULUMI) preview --diff --stack $(PULUMI_STACK)

infra-up: check-env
	cd infra && $(PULUMI_ENV) $(PULUMI) up --stack $(PULUMI_STACK)

trader-check:
	DONGPA_TRADER_RUNS=$(DONGPA_TRADER_RUNS) uv run python -m trader check --all

clean:
	- docker rmi $(APP_NAME)-viewer:latest
