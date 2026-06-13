# Makefile for running the mula alphazero package remotely via the Colab CLI.
# Requires: colab CLI (https://github.com/googlecolab/colabtools)

SESSION_NAME ?= mula-alphazero
GPU          ?=
TPU          ?=
LOCAL_SRC    := mula/models/alphazero
REMOTE_DIR   := /content/mula/models/alphazero

.PHONY: help \
        colab-new colab-new-gpu colab-status \
        colab-upload colab-install \
        colab-run colab-run-exec colab-download \
        colab-logs colab-stop colab-clean colab-all

help:
	@echo "Colab remote execution targets:"
	@echo "  colab-new          Create a new Colab session (CPU by default)"
	@echo "  colab-new-gpu      Create a new Colab session with --gpu T4"
	@echo "  colab-status       Show session status"
	@echo "  colab-upload       Upload mula/models/alphazero/ to $(REMOTE_DIR) on the VM"
	@echo "  colab-install      Install dependencies on the VM"
	@echo "  colab-run          Run via piped colab console"
	@echo "  colab-run-exec     Run via colab exec with 1-hour timeout (recommended)"
	@echo "  colab-download     Pull checkpoints/plots from the VM"
	@echo "  colab-logs         Show last 20 log entries"
	@echo "  colab-stop         Stop the session (saves compute units)"
	@echo "  colab-clean        Stop session and delete local artifacts"
	@echo "  colab-all          Upload + install + run (uses colab-run-exec)"

## Provisioning

colab-new:
ifeq ($(GPU),)
ifeq ($(TPU),)
	colab new -s $(SESSION_NAME)
else
	colab new -s $(SESSION_NAME) --tpu $(TPU)
endif
else
	colab new -s $(SESSION_NAME) --gpu $(GPU)
endif
	colab status -s $(SESSION_NAME)

colab-new-gpu:
	$(MAKE) colab-new GPU=T4

colab-status:
	colab status -s $(SESSION_NAME)

## Sync & dependencies

colab-mkdir:
	@echo "mkdir -p $(REMOTE_DIR)" | colab console -s $(SESSION_NAME)

colab-upload: colab-mkdir
	colab upload -s $(SESSION_NAME) $(LOCAL_SRC)/__init__.py $(REMOTE_DIR)/__init__.py
	colab upload -s $(SESSION_NAME) $(LOCAL_SRC)/params.py   $(REMOTE_DIR)/params.py
	colab upload -s $(SESSION_NAME) $(LOCAL_SRC)/modeling.py $(REMOTE_DIR)/modeling.py
	colab upload -s $(SESSION_NAME) $(LOCAL_SRC)/train.py    $(REMOTE_DIR)/train.py
	colab upload -s $(SESSION_NAME) $(LOCAL_SRC)/run_model.py $(REMOTE_DIR)/run_model.py
	colab upload -s $(SESSION_NAME) $(LOCAL_SRC)/README.md   $(REMOTE_DIR)/README.md

colab-install:
	@echo "import subprocess; subprocess.check_call(['python', '-m', 'pip', 'install', '-q', 'jax==0.10.0', 'jaxlib==0.10.0', 'jax-cuda12-plugin==0.10.0', 'jax-cuda12-pjrt==0.10.0', 'flax>=0.12.7', 'pgx', 'mctx', 'matplotlib', 'omegaconf', 'pydantic'])" | colab exec -s $(SESSION_NAME) --timeout 300

## Execution

colab-run: colab-upload colab-install
	@echo "cd $(REMOTE_DIR)/../.. && python -m mula.models.alphazero.train env_id=othello seed=0" | colab console -s $(SESSION_NAME)

colab-run-exec: colab-upload colab-install
	colab exec -s $(SESSION_NAME) --timeout 3600 -f scripts/remote_train.py

## Artifacts

colab-download:
	mkdir -p output
	colab download -s $(SESSION_NAME) /content/checkpoints output/checkpoints 2>/dev/null || true
	colab download -s $(SESSION_NAME) /content/plots output/plots 2>/dev/null || true
	@echo "Artifacts saved to output/"

colab-logs:
	colab log -s $(SESSION_NAME) -n 20

## Teardown

colab-stop:
	colab stop -s $(SESSION_NAME)

colab-clean: colab-stop
	rm -rf output/

## Convenience combo

colab-all: colab-upload colab-install colab-run-exec
