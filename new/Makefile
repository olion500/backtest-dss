APP_NAME := dongpa
PORT := 8501

help:
	@echo ""
	@echo "Dongpa Backtest Makefile Commands"
	@echo "--------------------------------------------"
	@echo "make build   - Build Docker image"
	@echo "make run     - Run app (port $(PORT))"
	@echo "make bash    - Open shell in container"
	@echo "make clean   - Remove Docker image"
	@echo "make dev     - Run development mode with sync"
	@echo "--------------------------------------------"
	@echo "Example: make build && make run"
	@echo ""

build:
	docker build -t $(APP_NAME):latest .

run:
	docker run -it --rm -p $(PORT):8501 $(APP_NAME):latest

bash:
	docker run -it --rm $(APP_NAME):latest bash

clean:
	docker rmi $(APP_NAME):latest || true

DEV_IMAGE := $(APP_NAME)-dev

dev:
	docker run -it --rm -p $(PORT):8501 -v "$(shell pwd)":/app $(DEV_IMAGE):latest
