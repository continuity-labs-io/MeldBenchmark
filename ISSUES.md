# MeldBenchmark Issues & Technical Debt

This document tracks known issues, technical debt, and open TODOs within the codebase.

## 1. Empty Implementations: Hardware Substrate
**Location:** `src/core/substrate.py`
**Description:**
The `HardwareSubstrate` abstract base class and `CPUSubstrate` have methods containing only `pass`.
**Action Item:**
Review if `pass` should be replaced with `NotImplementedError` or if concrete no-op implementations (like `CPUSubstrate.synchronize`) should be documented as intentional.

## 2. Add Neurospike data integration.
FinalSpark Neuroplatform: Standard Intan MEA, Low: 32 to 64 channels per well, 20 kHz to 30 kHz, Continuous 24/7 streaming for months.  Use for Time: Perfect for testing Mamba-2's ability to handle infinite sequence lengths and compute thermodynamic drift over weeks, but it lacks the spatial density to show off Mamba's capacity.

