"""
co2_predictor
=============

Modular, OOP implementation of the CO2 Emission prediction project.

This package intentionally mirrors the structure of the original
exploratory notebook (`CO2 Emission.ipynb`), but separates the code
into three concerns that must never be mixed:

1. Training-time code   -> data/, features/, modeling/
2. Inference-time code  -> inference/, persistence.py
3. Analysis-only code   -> evaluation/diagnostics.py
                           (never imported by inference/ or by the
                           production training entrypoint's critical
                           path -- used only from notebooks/scripts
                           for human-facing plots and reports)

See scripts/run_training.py for the end-to-end training entrypoint
that ties every module below together (equivalent to running the
whole notebook top to bottom).
"""
