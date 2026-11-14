"""
src/final_model.py - Primary Model Forwarder
============================================
Official Model: The GOAT Model (src/goat_model.py)
Candidate Model (NOT_FINALMODEL): Eemil 3-Sleeve (src/NOT_FINALMODEL_eemil.py)

By default, final_positions routes to the GOAT Model (Option B: Kristofer Stretch Stop + Cross-Sectional Momentum + 10% Deadband).
"""
from __future__ import annotations
import polars as pl
import pandas as pd
from src.goat_model import goat_positions, DATE
import src.NOT_FINALMODEL_eemil as NOT_FINALMODEL_eemil

# Primary official submission engine: The GOAT Model
def final_positions(prices: pl.DataFrame | pd.DataFrame, macro: pl.DataFrame | pd.DataFrame | None = None) -> pl.DataFrame:
    """
    Official Hackathon Submission Positions.
    Runs the GOAT Model (Dual-Engine: Kristofer Stretch Stop + Sector-Neutral Cross-Sectional Momentum).
    Macro parameter is optional and accepted for interface compatibility.
    """
    return goat_positions(prices)

# Exploratory candidate model for research comparison
def not_final_eemil_positions(prices: pl.DataFrame, macro: pl.DataFrame) -> pl.DataFrame:
    """Exploratory candidate (NOT_FINALMODEL): Eemil Trend + CPI + Twins."""
    return NOT_FINALMODEL_eemil.final_positions(prices, macro)
