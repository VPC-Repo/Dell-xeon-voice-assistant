# Repo-local: virtualenv in .venv, models in ./models, pids and logs in
# ./run. Nothing needs sudo and nothing is installed system-wide.

.PHONY: setup run stop status test lint secrets

setup:                 ## one-time: virtualenv, models, browser client, .env
	bash scripts/setup.sh

run:                   ## start the assistant (MOCK=1 for no models)
	bash scripts/run.sh start

stop:                  ## stop everything run started
	bash scripts/run.sh stop

status:                ## health of each part
	bash scripts/run.sh status

test:                  ## unit + integration; needs dev dependencies
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check gateway tts-server tests scripts

secrets:
	git ls-files -z | xargs -0 .venv/bin/detect-secrets-hook --no-verify  # pragma: allowlist secret
