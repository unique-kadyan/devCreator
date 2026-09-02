# Convenience targets. Everything here is just `asa` underneath.
.PHONY: help doctor test research episode dashboard backup lint clean

help:
	@echo "  make doctor     check this machine is ready"
	@echo "  make test       run the test suite (no network, no keys)"
	@echo "  make research   collect and score topics"
	@echo "  make episode    queue and run one episode end to end"
	@echo "  make dashboard  review queue at http://127.0.0.1:8420"
	@echo "  make backup     snapshot the database and irreplaceable assets"

doctor:
	.venv/bin/asa doctor

test:
	.venv/bin/python -m pytest tests/ -q

research:
	.venv/bin/asa research

# TOPIC="a clever fox opens a village bakery" make episode
episode:
	.venv/bin/asa job new $(if $(TOPIC),--topic "$(TOPIC)",)
	.venv/bin/asa run

dashboard:
	.venv/bin/asa dashboard

backup:
	./scripts/backup.sh

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
