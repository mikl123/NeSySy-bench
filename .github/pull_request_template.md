## What and why

<!-- What does this pull request change, and why? Link the issue it addresses, e.g. "Closes #12". -->

## Checklist

- [ ] Tests pass: `.venv/bin/python -m pytest nesy_data_gen/tests nesy_approaches/tests`
- [ ] `pre-commit run --all-files` is clean
- [ ] `docs/` and the README are updated if behaviour or options changed
- [ ] **Does this change the paper's results** (generated constraints or datasets, metrics, pinned versions)?
  - [ ] No
  - [ ] Yes: explained below, with the output of `./smoke_test.sh` before and after
