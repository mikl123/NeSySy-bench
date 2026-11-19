# Contributing to NeSySy

Bug reports, questions and pull requests are welcome.

## Before you start

- **Bugs and questions:** open an issue. Include the command you ran, what you expected, what happened, your OS and Python version, and how you installed NeSySy.
- **Larger changes** (a new constraint type, learner or generator, or a change to existing behaviour): open an issue first, so we can agree on the approach before you spend time on it.
- **Small fixes** (typos, documentation, obvious bugs) can go straight to a pull request.

## Setup

```bash
./install.sh                                         # pinned environment in .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/pre-commit install                         # run the hooks on every commit
```

## Before opening a pull request

1. Run the formatters and checks: `.venv/bin/pre-commit run --all-files`
2. Run the tests: `.venv/bin/python -m pytest nesy_data_gen/tests nesy_approaches/tests`
3. Update `docs/` and the README if behaviour or options change.
4. Keep each pull request to one change.

## Changes that affect the paper's results

The version used for the paper is frozen at tag `v1.0.0`. `main` can move on, but a change that alters generated constraints or datasets, the metrics, or the numbers produced by `run.py` must be explained in the pull request: what changes, why, and the output of `./smoke_test.sh` before and after. Changes to the pinned versions in `requirements.txt` or `constraints.txt` count as such changes.

## Review and merging

Pull requests need one approving review from a maintainer and passing CI (checks and tests). They are squash-merged.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
