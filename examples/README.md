# Examples

Each pipeline script is a self-contained ~60-line file: build a constraint, generate a dataset, train a method, print metrics. Shared configs live in `example_setting/`.

## CNF (boolean / k-SAT)

| Script | Method tested |
|--------|---------------|
| `cnf_pipeline.py` | plain NN baseline |
| `cnf_pishield_pipeline.py` | PiShield (CCN+) propositional layer |
| `cnf_sequential_pipeline.py` | sequential autoregressive model |
| `cnf_distant_supervision_dataset.py` | dataset generation only (distant supervision) |
| `cnf_distant_supervision_deepproblog_pipeline.py` | distant supervision + DeepProbLog semantic loss |

## SMT (continuous)

| Script | Method tested |
|--------|---------------|
| `smt_bounded_interval_pipeline.py` | plain NN baseline |
| `smt_bounded_interval_pishield_pipeline.py` | PiShield (DRL) QFLRA layer |
| `smt_linear_inequality_pipeline.py` | plain NN baseline |

## Tutorials

| Script | Topic |
|--------|-------|
| `custom_constraint_example.py` | subclass `Constraint` with your own predicate |
| `custom_generator_function_example.py` | plug your own `y = f(x)` callable into `generate_dataset` |
