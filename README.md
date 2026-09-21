# ÆSOP Early Warning Detection Models

Code and data to compare and evaluate the synchronicity and detection performance of models implemented by the ÆSOP teams.

# Moving Epidemic Method (MEM) – Python implementation for early warning surveillance

**Author:** Juliane Oliveira **Affiliation:** Centre for Data and Knowledge Integration for Health (CIDACS), Gonçalo Moniz Institute, Oswaldo Cruz Foundation (FIOCRUZ), Brazil

This repository contains a **Python implementation of the Moving Epidemic Method (MEM)** to be used in the context of early warning detection of epidemic surges, developed as part of the **AESOP Project**.

The original MEM methodology was proposed and widely implemented in **R**. This work provides a **fully Python-based implementation**, designed for large-scale, scalable and municipality-level surveillance analyses, particularly within the AESOP Project.

---

## Methodological reference

The implementation follows the principles described in:

> Vega T, Lozano JE, Meerhoff T, Snacken R, Mott J, Ortiz de Lejarazu R, Nunes B. *Influenza surveillance in Europe: establishing epidemic thresholds by the moving epidemic method.* **Influenza and Other Respiratory Viruses**, 2013; 7(4): 546–558.

Key adaptations include:

- a Python-native implementation,
- flexible seasonal definitions at the municipality level,
- integration with rolling averages and automated parameter selection,
- scalability for nationwide surveillance systems.

---

## Workflow overview

The MEM pipeline is executed in **three main steps**, each corresponding to an executable script.

```
┌──────────────────────────────┐
│  Municipal weekly data       │
│  (ILI / IVAS counts)         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│  Script 1: def_sea_peri_MEM_all_cities.py        │
│  Select season definition & delta                │
│  (per municipality)                              │
└──────────────┬───────────────────────────────────┘
               │  def_sea_MEM_out_<date>.parquet
               ▼
┌──────────────────────────────────────────────────┐
│  Script 2: mem_thresholds_all_cities.py          │
│  Estimate MEM baselines, epidemic thresholds,    │
│  and intensity levels                            │
│  (per municipality)                              │
└──────────────┬───────────────────────────────────┘
               │  mem_output_<date>.parquet
               ▼
┌──────────────────────────────────────────────────┐
│  Script 3: surge_aps_with_MEM.py                 │
│  Define epidemic surges and warning signals      │
│  (MEM-based + fallback strategy)                 │
└──────────────┬───────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Final AESOP-ready dataset                   │
│  Weekly MEM-based warning indicators         │
│. aesop_02_02_2026_with_MEM.parquet           │
└──────────────────────────────────────────────┘
```

---

## 1. Select seasonal definition and delta (per municipality)

**Script:** `def_sea_peri_MEM_all_cities.py`

This script:

- loads municipal-level surveillance data,
- applies MEM using multiple seasonal definitions (e.g. calendar year or
  influenza-style seasons),
- searches over a range of `delta` values,
- and selects the configuration that satisfies predefined constraints on
  epidemic onset and epidemic growth.

Each municipality is processed independently.

### Outputs

- `def_sea_MEM_out_<date>.parquet` Selected seasonal definition and MEM parameter (`delta`) for each municipality.

- `set_bad_cities_<date>.csv` Municipalities for which no valid MEM configuration was found.

### Input data requirements

The input dataset must contain, at minimum:

- `co_ibge` – municipality code
- `epiyear` – epidemiological year
- `epiweek` – epidemiological week
- `atend_ivas` – weekly ILI/IVAS counts

Additional metadata columns (state, municipality name, rates) may be included.

---

## 2. Estimate MEM baselines and epidemic thresholds (per municipality)

**Script:** `baseline_epi_thresh_mem_all_city.py`

This script:

- loads municipal-level weekly surveillance data,
- loads the selected seasonal definition and `delta` from Script 1,
- applies a 4-week moving average to smooth time series,
- dynamically constructs epidemiological seasons (calendar year or shifted
  influenza-style seasons),
- applies MEM independently to each municipality,
- estimates baselines and epidemic thresholds,
- computes low, medium, and high intensity levels.

The seasonal structure used for each municipality exactly matches the one
selected in the previous step.

### Outputs

- `mem_output_<date>.parquet` Consolidated MEM results per municipality, including:
  
  - baseline and post-baseline values,
  - epidemic and post-epidemic thresholds,
  - intensity levels (low, medium, high),
  - seasonal metadata.

---

## 3. Define epidemic surges and generate warning signals

**Script:** `surge_aps_with_MEM.py`

This script:

- loads MEM thresholds estimated in Script 2,
- applies MEM-based surge definitions to weekly IVAS counts,
- generates:
  - binary epidemic surge indicators (0/1),
  - multi-level surge intensity categories,
- removes isolated one-week signals and enforces temporal consistency,
- applies a fallback statistical definition for municipalities where MEM
  thresholds could not be estimated,
- merges MEM-based and fallback results into a unified dataset.

### Outputs

- `aesop_<date>_with_MEM.parquet` Final dataset containing weekly MEM-based surges indicators per municipality,
  ready for visualization, modeling, and decision-support workflows.

## Notes

- This implementation adapts the original MEM methodology to Python while
  preserving its statistical foundations.
- Seasonal definitions may correspond to the calendar year or to
  municipality-specific epidemic onset weeks.
- All computations are performed independently for each municipality,
  enabling scalable nationwide analyses.
- The code is modular and extensible, and was developed to support real-time
  surveillance workflows within the **AESOP Project**.

---

## Citation

If you use this software, please cite it as:

> Oliveira J. *Moving Epidemic Method (MEM) implementation for early warning detection of epidemic surges.* Centre for Data and Knowledge Integration for Health (CIDACS), Oswaldo Cruz Foundation (FIOCRUZ), 2026.

A `CITATION.cff` file is provided to facilitate citation in software-aware
reference managers.
