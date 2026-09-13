PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup data golden eval reproduce verify test demo clean

setup:
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]" -c constraints.txt

data:
	$(PY) -m support_agent.data.download
	$(PY) -m support_agent.data.build_pairs

golden:
	$(PY) -m support_agent.data.sample_golden

eval:
	$(PY) -m support_agent.eval.run --out results

reproduce:
	LLM_OFFLINE=1 $(PY) -m support_agent.eval.run --out results

# Reproduce, then fail loudly if anything moved. This is the claim CI checks: the committed
# results are exactly what the committed cache and the seeded bootstraps produce.
verify:
	@t=$$(mktemp); cp results/summary.md $$t; \
	LLM_OFFLINE=1 $(PY) -m support_agent.eval.run --out results >/dev/null 2>&1; \
	if diff -u $$t results/summary.md; then \
		echo "OK: summary.md and every bootstrap interval reproduced byte-identically"; \
	else \
		echo "FAIL: results moved; the diff above is what changed"; rm -f $$t; exit 1; \
	fi; rm -f $$t

test:
	$(PY) -m pytest -q

demo:
	LLM_OFFLINE=0 $(PY) -m support_agent.cli demo "$(MSG)"

clean:
	rm -rf results/*.tmp .pytest_cache
