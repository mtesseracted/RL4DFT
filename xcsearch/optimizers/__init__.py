"""Optimizer adapters, one module per backend (D9).

Each third-party backend gets its own module so that importing one never
requires the others to be installed: only ``optuna_tpe`` imports optuna, only
``cmaes`` imports cma. ``composed`` is backend-free -- it assembles the
two-level decomposition of D11 out of whatever halves it is handed.
"""
