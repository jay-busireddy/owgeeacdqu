# OWGE-EAC-DQU Integration Study

This package contains a prospective paper draft and two frozen confirmatory test suites.

## Windows setup

```powershell
cd OWGE_EAC_DQU_Integration_Study

py -m venv .venv_integration
.\.venv_integration\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Syntax check

```powershell
python -m py_compile study1_factorial_pipeline.py
python -m py_compile study2_priming_metareasoning.py
python -m py_compile combine_confirmatory.py
```

## Smoke tests

```powershell
python study1_factorial_pipeline.py `
  --preset smoke `
  --output smoke_study1

python study2_priming_metareasoning.py `
  --preset smoke `
  --output smoke_study2
```

Smoke results are software checks only.

## Frozen confirmatory runs

```powershell
python study1_factorial_pipeline.py `
  --preset confirmatory `
  --output confirmatory_study1

python study2_priming_metareasoning.py `
  --preset confirmatory `
  --output confirmatory_study2

python combine_confirmatory.py `
  --study1 confirmatory_study1 `
  --study2 confirmatory_study2 `
  --output confirmatory_combined
```

Do not alter seeds, coefficients, thresholds, endpoints, learning rates, thought cost, or sample sizes after inspecting partial confirmatory results.

## Return the result package

```powershell
Compress-Archive `
  -Path .\confirmatory_study1\*,.\confirmatory_study2\*,.\confirmatory_combined\* `
  -DestinationPath .\owge_eac_dqu_confirmatory_results.zip
```

The paper draft intentionally contains no confirmatory result claims. When you return the ZIP, the result tables, discussion, and conclusion can be updated from the frozen data.
