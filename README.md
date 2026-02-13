# DGOJ_CDB_EXAMPLE
This file should be interpreted as a methodological bridge between:  1. The Spanish regulator’s behavioural feature framework, and   2. The Dutch CDB data infrastructure.  It demonstrates conceptual portability rather than production-ready implementation.


Introduction
============

This file provides a small illustrative example of how the DGOJ features as mentioned 
in https://technical-regulation-information-system.ec.europa.eu/en/notification/27517
can be mapped onto the current Dutch CDB (Controle Data Bank) data structure.

The implementation should be considered a prototype.
It is intended purely to demonstrate feasibility and translation logic.
Errors, simplifications, and imperfect proxies are expected.

Relationship to the broader architecture
----------------------------------------
This standalone example is part of the full open-source
feature engineering framework developed in the ZonMw project, containing:

- standardized data loaders for WOK_* tables
- streaming feature computation
- scenario-based feature windows
- reproducible regulator-grade pipelines

As a result, many imports and structural dependencies will not work in isolation.

Feature feasibility and proxy decisions
---------------------------------------
Limitations of the CDB schema required creative approximations,
performance-driven simplifications, or omission of certain variables.

Key deviations and assumptions:

- **F5 (gender)**  
  Impossible to derive from the CDB dataset.

- **F17 (number of countries)**  
  Approximated using the diversity of popular football leagues involved in
  betting behaviour as a behavioural proxy.

- **F19 (number of competitions)**  
  Approximated by measuring how many distinct betting events occur
  simultaneously within a time window.

- **F20 (domestic vs. foreign betting)**  
  Implemented as the percentage of Dutch football bets versus foreign bets
  (identified through non-recognisable Dutch team names).

- **F21 (currency)**  
  Not observable within the CDB data.

- **F22 / F23 (limit participation)**  
  Often missing in the source data.  
  When absent, alternative limit-related variables were used as proxies.

- **F25 (time-unrestricted indicator)**  
  The original definition is not limited to the reference period, which may
  introduce methodological risk.  
  Implemented here as a binary indicator.

- **F30 vs. F31 (game structure assumptions)**  
  Two alternative interpretations are used:
  
  - *F30*: assumes multiple game transactions exist within a single game.
  - *F31*: assumes multiple rows share the same game session ID.
  
  Further validation is required and will be checked separately in a notebook.

- **F32–F38 (label differences)**  
  Minor naming deviations due to schema alignment with the CDB model.

- **F48 (cash-out proxy via BET_UPDATED)**  
  In the CDB data model, `BET_UPDATED` indicates that a previously placed bet
  was modified. This may reflect:
  
  - a partial payout (cash-out), **but also**
  - cancellation of specific bet components.
  
  Because no explicit cash-out indicator exists in `WOK_Bet`, `BET_UPDATED`
  is used as a proxy.  
  This proxy can therefore produce **false positives**
  (partial cancellation ≠ true cash-out).

