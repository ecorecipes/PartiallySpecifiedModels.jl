# Bundled Datasets — Provenance & Checksums

This directory contains the two datasets bundled with `PartiallySpecifiedModels.jl`.

## `cop.dat`

- **SHA-256**: `aec57cd870da71a549c8db2f49faab6fc16a6bb2d391c5c7594c9cbd16d448df`
- **Format**: Whitespace-delimited plain text, 10 rows × 12 columns, no header row.
  - Column 1: Sampling time in days (`0, 10, 20, ..., 90`).
  - Columns 2–7: Naupliar stage abundances `N1`–`N6` (*Calanus finmarchicus*).
  - Columns 8–12: Pre-adult copepodite stage abundances `C1`–`C5` (adult stage `C6` is not included).
- **Provenance**: Simulated 11-stage benchmark dataset distributed with Simon Wood's `DDEfit` (`ddefit504`) software accompanying Wood (2001), *"Partially specified ecological models"*, *Ecological Monographs* 71(1):1–25.
- **Used in**: `examples/copepod.jl`, `vignettes/04_copepod/04_copepod.qmd`.

## `hare_lynx.csv`

- **SHA-256**: `ffa77f35700b534af6c524b3658ca30f3ee61f399f789ac29c8ecc4024a2eb4e`
- **Format**: Comma-separated values with header `year,hare,lynx` (91 annual rows, 1845–1935).
  - `year`: Calendar year (1845–1935).
  - `hare`: Snowshoe hare (*Lepus americanus*) pelts traded (thousands).
  - `lynx`: Canada lynx (*Lynx canadensis*) pelts traded (thousands).
- **Provenance**: Hudson's Bay Company pelt-trading records digitized from MacLulich (1937), *Fluctuations in the Numbers of the Varying Hare (Lepus americanus)* (University of Toronto Studies, Biological Series No. 43) and Odum (1953), *Fundamentals of Ecology*. Public domain historical data.
- **Used in**: `vignettes/03_lotka_volterra/03_lotka_volterra.qmd`, `vignettes/13_shape_constraints/13_shape_constraints.qmd`.
