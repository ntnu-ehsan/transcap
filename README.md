# Transcap Transformer Reactance Estimator

This project estimates transformer sizing and equivalent reactance for transmission substations using CSV inputs.

## What It Does

- Reads transformer records from a CSV file.
- Detects apparent power from one of:
  - `s_max`
  - `s_nom`
  - `Max_Apparent_Power`
  - or computes from `Max_Active_Power` and `Max_Reactive_Power`.
- Normalizes voltage levels to catalog classes (range-based bucketing).
- Selects transformer configuration under normal + N-1 constraints:
  - prioritizes fewer transformers first,
  - then lower installed MVA.
- Computes reactance outputs (`X_pu` and `X_ohm`) for feasible rows.
- Parses `LINESTRING` geometry and exports bus/HV/LV coordinates.

## Main Script

- `main.py`

## Input CSV Formats Supported

The script supports both legacy and PyPSA-like formats.

Required base columns:
- `transformer_id`
- `bus0`
- `bus1`
- `voltage_bus0`
- `voltage_bus1`

Power columns (any one option):
1. `s_max`
2. `s_nom`
3. `Max_Apparent_Power`
4. `Max_Active_Power` + `Max_Reactive_Power`

Optional:
- `geometry` as WKT `LINESTRING(...)` for coordinate extraction.

## How To Run

```bash
python main.py
```

Current default input path in code:
- `data/pypsa_eur_transformers.csv`

## Outputs

- Intermediate sizing/reactance table:
  - `data/transformers_reactance.csv`
- Final HL export:
  - `data/pypsa_transformers_reactance_HL.csv`
  - if target file is locked, fallback:
    - `data/pypsa_transformers_reactance_HL_new.csv`

## Status Values

- `OK`: feasible sizing found and reactance computed.
- `UNSUPPORTED_VOLTAGE`: voltage pair not covered by current catalog classes.
- `INFEASIBLE_MAX_N`: no feasible design with `n <= max_n` and available catalog ratings.
- `BAD_INPUT`: missing/invalid required numeric input (e.g., power).

## Notes

- Catalog and HV-side Xpu assumptions are configured in `main.py`.
- Reactance lookup uses snapped HV class for robust handling (e.g., 380 kV mapped to 400 kV class).
